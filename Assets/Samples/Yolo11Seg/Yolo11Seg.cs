using System;
using System.Collections.ObjectModel;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Unity;
using UnityEngine;
using UnityEngine.Assertions;
using Unity.Collections;
using Unity.Mathematics;

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
            public ComputeShader visualizeSegmentationShader;
        }

        public readonly struct Detection : IComparable<Detection>
        {
            public readonly int label;
            public readonly Rect rect;
            public readonly float probability;

            public Detection(Rect rect, int label, float probability)
            {
                this.rect = rect;
                this.label = label;
                this.probability = probability;
            }

            public int CompareTo(Detection other)
            {
                // Descending sort
                return other.probability.CompareTo(probability);
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
        private readonly int2 output0Shape;
        private NativeArray<Detection> proposalsArray;
        private NativeArray<Detection> detectionsArray;
        private int detectionCount = 0;

        private readonly Yolo11SegVisualize segmentation;

        public ReadOnlySpan<Detection> Detections => detectionsArray.AsReadOnlySpan()[..detectionCount];
        public Texture SegmentationTexture => segmentation.Texture;

        public Yolo11Seg(byte[] model, Options options)
            : base(model, options)
        {
            this.options = options;

            Assert.AreEqual(2, outputs.Count);

            // Output 0
            {
                var output0Shape = Array.ConvertAll(outputs[0].GetTensorTypeAndShape().Shape, x => (int)x);
                this.output0Shape = new int2(output0Shape[1], output0Shape[2]);

                const int MAX_PROPOSALS = 500;
                proposalsArray = new NativeArray<Detection>(MAX_PROPOSALS, Allocator.Persistent);
                detectionsArray = new NativeArray<Detection>(MAX_PROPOSALS, Allocator.Persistent);

                var labels = options.labelFile.text.Split('\n', StringSplitOptions.RemoveEmptyEntries);
                labelNames = Array.AsReadOnly(labels);
                classCount = labelNames.Count;

                Assert.IsTrue(classCount <= output0Shape[1] - 4); // 4:xywh
            }

            {
                var output1Shape = Array.ConvertAll(outputs[1].GetTensorTypeAndShape().Shape, x => (int)x);
                int3 shape = new(output1Shape[1], output1Shape[2], output1Shape[3]);
                segmentation = new Yolo11SegVisualize(shape, options.visualizeSegmentationShader, Colors);
            }
        }

        public override void Dispose()
        {
            base.Dispose();
            proposalsArray.Dispose();
            detectionsArray.Dispose();
            segmentation.Dispose();
        }

        protected override void PostProcess()
        {

            // 0: Parse predictions
            // [0: predictions] shape: 1,116,8400 (Batch_size=1, xywh+conf_cls(80)+nm(32), Num_anchors)
            var output0 = outputs[0].GetTensorDataAsSpan<float>();
            var proposals = GenerateProposals(output0, options.confidenceThreshold);

            if (proposals.Length == 0)
            {
                detectionCount = 0;
                return;
            }
            proposals.Sort();
            detectionCount = NMS(proposals, detectionsArray, options.nmsThreshold);

            // [1: protos] shape: 1,32,160,160
            var output1 = outputs[1].GetTensorDataAsSpan<float>();
            segmentation.Process(output1);
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

        public Color GetColor(in Detection detection)
        {
            return Colors[detection.label % Colors.Length];
        }

        // TODO: consider using Burst
        private NativeSlice<Detection> GenerateProposals(ReadOnlySpan<float> tensor, float confidenceThreshold)
        {
            // [0: predictions] shape: 1,116,8400 (Batch_size=1, xywh+conf_cls(80)+nm(32), Num_anchors)

            int2 shape = output0Shape.yx;
            int cols = shape.x;
            int classCount = this.classCount;

            // reciprocal width and height
            float rWidth = 1f / width;
            float rHeight = 1f / height;

            int proposalsCount = 0;

            // TODO: Should transpose first?
            // cols ->
            for (int anchorId = 0; anchorId < cols; anchorId++)
            {
                int classId = int.MinValue;
                float maxConfidence = float.MinValue;

                // rows
                for (int i = 0; i < classCount; i++)
                {
                    const int RECT_OFFSET = 4;
                    float confidence = tensor.GetValue(anchorId, RECT_OFFSET + i, shape);
                    if (confidence > maxConfidence)
                    {
                        maxConfidence = confidence;
                        classId = i;
                    }
                }

                // Filter out low confidence anchors
                if (maxConfidence < confidenceThreshold)
                {
                    continue;
                }

                // Normalize Rect
                float cx = tensor.GetValue(anchorId, 0, shape) * rWidth;
                float cy = tensor.GetValue(anchorId, 1, shape) * rHeight;
                float w = tensor.GetValue(anchorId, 2, shape) * rWidth;
                float h = tensor.GetValue(anchorId, 3, shape) * rHeight;
                float x = cx - w * 0.5f;
                float y = cy - h * 0.5f;

                proposalsArray[proposalsCount++] = new Detection(
                    new Rect(x, y, w, h),
                    classId,
                    maxConfidence);

                if (proposalsCount >= proposalsArray.Length)
                {
                    break;
                }
            }

            return proposalsArray.Slice(0, proposalsCount);
        }

        private static int NMS(
           in NativeSlice<Detection> proposals,
           NativeArray<Detection> detections,
           float iou_threshold)
        {
            int detectedCount = 0;

            foreach (Detection a in proposals)
            {
                bool keep = true;
                for (int i = 0; i < detectedCount; i++)
                {
                    Detection b = detections[i];

                    // Ignore different classes
                    if (b.label != a.label)
                    {
                        continue;
                    }
                    float iou = a.rect.IntersectionOverUnion(b.rect);
                    if (iou > iou_threshold)
                    {
                        keep = false;
                    }
                }
                if (keep)
                {
                    detections[detectedCount] = a;
                    detectedCount++;
                }
            }

            return detectedCount;
        }
    }
}
