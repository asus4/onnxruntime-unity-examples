using System;
using System.Collections.ObjectModel;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Unity;
using Microsoft.ML.OnnxRuntime.UnityEx;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.Assertions;

namespace Microsoft.ML.OnnxRuntime.Examples
{
    /// <summary>
    /// Licensed under AGPL-3.0 license
    /// See LICENSE for full license information.
    /// https://github.com/ultralytics/ultralytics/blob/main/LICENSE
    /// 
    /// https://docs.ultralytics.com/tasks/segment/
    /// 
    /// Ported from python code: 
    /// https://github.com/ultralytics/ultralytics/blob/aecf1da32b50e144b226f54ae4e979dc3cc62145/examples/YOLOv8-Segmentation-ONNXRuntime-Python/main.py
    /// </summary>
    public sealed class Yolo11Seg : ImageInference<float>
    {
        [Serializable]
        public class Options : ImageInferenceOptions
        {
            [Header("YOLO options")]
            public TextAsset labelFile;
            [Range(0f, 1f)]
            public float confidenceThreshold = 0.25f;
            [Range(0f, 1f)]
            public float nmsThreshold = 0.45f;
            [Range(5, 50)]
            public int maxSegmentation = 50;
            public ComputeShader visualizeSegmentationShader;
        }

        public readonly struct Detection : IDetection<Detection>
        {
            public readonly int label;
            public readonly Rect rect;
            public readonly float probability;
            public readonly int anchorId;

            public readonly int Label => label;
            public readonly Rect Rect => rect;

            public Detection(int label, Rect rect, float probability, int anchorId)
            {
                this.label = label;
                this.rect = rect;
                this.probability = probability;
                this.anchorId = anchorId;
            }

            public int CompareTo(Detection other)
            {
                // Descending sort
                return other.probability.CompareTo(probability);
            }

            public Color GetColor()
            {
                return Colors[label % Colors.Length];
            }
        }

        public static readonly Color[] Colors = (new uint[]
        {
            0x042AFFFF,
            0x0BDBEBFF,
            0xF3F3F3FF,
            0x00DFB7FF,
            0x111F68FF,
            0xFF6FDDFF,
            0xFF444FFF,
            0xCCED00FF,
            0x00F344FF,
            0xBD00FFFF,
            0x00B4FFFF,
            0xDD00BAFF,
            0x00FFFFFF,
            0x26C000FF,
            0x01FFB3FF,
            0x7D24FFFF,
            0x7B0068FF,
            0xFF1B6CFF,
            0xFC6D2FFF,
            0xA2FF0BFF,
        })
        .Select(hex => new Color32(
            (byte)((hex >> 24) & 0xFF),
            (byte)((hex >> 16) & 0xFF),
            (byte)((hex >> 8) & 0xFF),
            (byte)(hex & 0xFF)))
        .Select(c => (Color)c)
        .ToArray();

        private readonly Options options;
        public readonly int classCount;
        public readonly ReadOnlyCollection<string> labelNames;

        // [0: predictions] shape: 1,116,8400 (Batch_size=1, xywh_conf_cls_nm, Num_anchors)
        // [1: protos] shape: 1,32,160,160
        private readonly int3 output0Shape;
        private readonly NativeArray<float> output0Transposed; // 1, 8400, 116
        private NativeList<Detection> proposalList;
        private NativeList<Detection> detectionList;

        private readonly Yolo11SegVisualize segmentation;

        public NativeArray<Detection>.ReadOnly Detections => detectionList.AsReadOnly();
        public Texture SegmentationTexture => segmentation.Texture;

        // Profilers
        static readonly ProfilerMarker generateProposalsMarker = new($"{typeof(Yolo11Seg).Name}.GenerateProposals");
        static readonly ProfilerMarker segmentationMarker = new($"{typeof(Yolo11Seg).Name}.Segmentation");

        public Yolo11Seg(byte[] model, Options options)
            : base(model, options)
        {
            this.options = options;

            Assert.AreEqual(2, outputs.Count);

            // Output 0
            {
                var info = outputs[0].GetTensorTypeAndShape();
                Assert.AreEqual(3, info.DimensionsCount);
                output0Shape = new int3((int)info.Shape[0], (int)info.Shape[1], (int)info.Shape[2]);
                output0Transposed = new NativeArray<float>((int)info.ElementCount, Allocator.Persistent);

                const int MAX_PROPOSALS = 100;
                proposalList = new NativeList<Detection>(MAX_PROPOSALS, Allocator.Persistent);
                detectionList = new NativeList<Detection>(MAX_PROPOSALS, Allocator.Persistent);

                var labels = options.labelFile.text.Split('\n', StringSplitOptions.RemoveEmptyEntries);
                labelNames = Array.AsReadOnly(labels);
                classCount = labelNames.Count;
            }

            // Output 1
            {
                var info = outputs[1].GetTensorTypeAndShape();
                Assert.AreEqual(4, info.DimensionsCount);
                var shape = new int3((int)info.Shape[1], (int)info.Shape[2], (int)info.Shape[3]);
                segmentation = new Yolo11SegVisualize(
                    shape,
                    options.visualizeSegmentationShader,
                    Colors,
                    options.maxSegmentation);
            }
        }

        public override void Dispose()
        {
            base.Dispose();
            proposalList.Dispose();
            detectionList.Dispose();
            segmentation.Dispose();
            output0Transposed.Dispose();
        }

        protected override void PostProcess()
        {
            var output0 = outputs[0].GetTensorDataAsSpan<float>();

            // 0: Parse predictions
            // [0: predictions] shape: 1,116,8400 (Batch_size=1, xywh+conf_cls(80)+nm(32), Num_anchors)
            generateProposalsMarker.Begin();
            GenerateProposals(output0, proposalList, options.confidenceThreshold);
            generateProposalsMarker.End();

            if (proposalList.Length == 0)
            {
                return;
            }
            proposalList.Sort();
            IDetection<Detection>.NMS(proposalList.AsArray(), detectionList, options.nmsThreshold);

            segmentationMarker.Begin();
            // [1: protos] shape: 1,32,160,160
            var output1 = outputs[1].GetTensorMutableDataAsSpan<float>();
            segmentation.Process(output0Transposed, output1, Detections);
            segmentationMarker.End();
        }

        /// <summary>
        ///  Convert CV rect to Viewport space
        /// </summary>
        /// <param name="rect">A Normalized Rect, input should be 0 - 1</param>
        /// <returns></returns>
        public Rect ConvertToViewport(in Rect rect)
        {
            Rect unityRect = rect.FlipY();
            var mtx = InputToViewportMatrix;
            Vector2 min = mtx.MultiplyPoint3x4(unityRect.min);
            Vector2 max = mtx.MultiplyPoint3x4(unityRect.max);
            return new Rect(min, max - min);
        }



        private void GenerateProposals(ReadOnlySpan<float> tensor, NativeList<Detection> proposals, float confidenceThreshold)
        {
            proposals.Clear();

            Assert.AreEqual(1, output0Shape.x, "Support only batch size 1");

            int classCount = this.classCount;

            // reciprocal width and height
            float rWidth = 1f / width;
            float rHeight = 1f / height;

            // shape: 116,8400 (xywh+conf_cls(80)+nm(32), Num_anchors)
            var tensor2D = tensor.AsSpan2D(output0Shape.yz);

            // shape: 8400,116 (Num_anchors, xywh+conf_cls(80)+nm(32))
            var tensorTransposed = new Span2D<float>(output0Transposed, output0Shape.zy);
            tensor2D.Transpose(tensorTransposed);

            for (int anchorId = 0; anchorId < output0Shape.z; anchorId++)
            {
                ReadOnlySpan<float> anchor = tensorTransposed[anchorId];

                // Find max confidence
                var confidences = anchor.Slice(4, classCount);
                int classId = confidences.ArgMax();
                float maxConfidence = confidences[classId];

                // Filter out low confidence anchors
                if (maxConfidence < confidenceThreshold)
                {
                    continue;
                }

                // Normalize Rect
                float cx = anchor[0] * rWidth;
                float cy = anchor[1] * rHeight;
                float w = anchor[2] * rWidth;
                float h = anchor[3] * rHeight;
                float x = cx - w * 0.5f;
                float y = cy - h * 0.5f;

                proposals.Add(new Detection(
                    classId,
                    new Rect(x, y, w, h),
                    maxConfidence,
                    anchorId)
                );
            }
        }
    }
}
