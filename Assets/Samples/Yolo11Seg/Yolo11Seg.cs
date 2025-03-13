using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading;
using Microsoft.ML.OnnxRuntime.Unity;
using Microsoft.ML.OnnxRuntime.UnityEx;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
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
            [Range(320, 1024)]
            public int dynamicMaxSize = 640;

            [Header("Segmentation options")]
            public ComputeShader visualizeSegmentationShader;
            [Range(5, 50)]
            public int maxDetectionCount = 50;
            [Range(0f, 1f)]
            public float maskThreshold = 0.5f;
        }

        public readonly struct Detection : IDetection<Detection>
        {
            public readonly Rect rect;
            public readonly int label;
            public readonly float probability;
            public readonly int anchorId;

            public readonly int Label => label;
            public readonly Rect Rect => rect;

            public Detection(Rect rect, int label, float probability, int anchorId)
            {
                this.rect = rect;
                this.label = label;
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

        // [0: predictions] shape: 1,116,8400 (Batch_size=1, XyWh_conf_cls_nm, Num_anchors)
        // [1: proto] shape: 1,32,160,160
        private int3 output0Shape;
        private NativeArray<float> output0Transposed; // 1, 8400, 116
        private NativeList<Detection> proposalList;
        private NativeList<Detection> detectionList;

        private Yolo11SegVisualize segmentation;

        public NativeArray<Detection>.ReadOnly Detections => detectionList.AsReadOnly();
        public Texture SegmentationTexture => segmentation.Texture;

        // Profilers
        static readonly ProfilerMarker generateProposalsMarker = new($"{typeof(Yolo11Seg).Name}.GenerateProposals");
        static readonly ProfilerMarker segmentationMarker = new($"{typeof(Yolo11Seg).Name}.Segmentation");

        public Yolo11Seg(byte[] model, Options options)
            : base(model, options)
        {
            this.options = options;

            var labels = options.labelFile.text.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            labelNames = Array.AsReadOnly(labels);
            classCount = labelNames.Count;

            if (!isDynamicOutputShape)
            {
                EnsurePostProcessResources(outputs);
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                proposalList.Dispose();
                detectionList.Dispose();
                segmentation?.Dispose();
                output0Transposed.Dispose();
            }
            base.Dispose(disposing);
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

        protected override void PreProcess(Texture texture)
        {
            if (isDynamicInputShape)
            {
                EnsureDynamicInputs(texture);
            }
            base.PreProcess(texture);
        }

        protected override Awaitable PreProcessAsync(Texture texture, CancellationToken cancellationToken)
        {
            if (isDynamicInputShape)
            {
                EnsureDynamicInputs(texture);
            }
            return base.PreProcessAsync(texture, cancellationToken);
        }

        protected override void PostProcess(IReadOnlyList<OrtValue> outputs)
        {
            if (isDynamicOutputShape)
            {
                EnsurePostProcessResources(outputs);
            }

            var output0 = outputs[0].GetTensorDataAsSpan<float>();

            // 0: Parse predictions
            // [0: predictions] shape: 1,116,8400 (Batch_size=1, XyWh+conf_cls(80)+nm(32), Num_anchors)
            generateProposalsMarker.Begin();
            ScheduleGenerateProposalsJob(output0, proposalList, options.confidenceThreshold)
                .Complete();
            generateProposalsMarker.End();

            if (proposalList.Length == 0)
            {
                detectionList.Clear();
                return;
            }

            // Run Non-Maximum Suppression
            proposalList.Sort();
            IDetection<Detection>.NMS(proposalList, detectionList, options.nmsThreshold);

            segmentationMarker.Begin();
            // [0] 1(batch), 8400(anchor), 116(data)
            // [1: proto] shape: 1,32,160,160
            var output0Span = output0Transposed.AsReadOnlySpan();
            var output0Tensor = output0Span.AsSpan2D(output0Shape.zy);
            var output1 = outputs[1].GetTensorDataAsSpan<float>();
            segmentation.Process(output0Tensor, output1, Detections);
            segmentationMarker.End();
        }

        private void EnsureDynamicInputs(Texture texture)
        {
            if (!isDynamicInputShape)
            {
                return;
            }

            int2 texSize = new(texture.width, texture.height);
            // Choose similar aspect ratio to the texture, 
            // But needs to be multiple of 32 for YOLOv11
            const int ALIGNMENT_SIZE = 32;
            int2 dim = MathUtil.ResizeToMaxSize(texSize, options.dynamicMaxSize, ALIGNMENT_SIZE);

            bool needResize = dim.x != Width || dim.y != Height;
            if (!needResize)
            {
                return;
            }

            // Resize input tensor
            textureToTensor.Dispose();
            textureToTensor = CreateTextureToTensor(dim.x, dim.y);

            foreach (var input in inputs)
            {
                input.Dispose();
            }
            var inputMetadata = session.InputMetadata;
            var metadata = inputMetadata.Values.First();
            var ortValue = OrtValue.CreateAllocatedTensorValue(
                OrtAllocator.DefaultInstance,
                metadata.ElementDataType,
                new long[] { 1, 3, dim.y, dim.x });
            inputs = new List<OrtValue>(1) { ortValue }.AsReadOnly();
            Debug.Log($"Resized input to {dim}");
        }

        private void EnsurePostProcessResources(IReadOnlyList<OrtValue> outputs)
        {
            Assert.AreEqual(2, outputs.Count);

            // Output 0
            var info0 = outputs[0].GetTensorTypeAndShape();
            Assert.AreEqual(3, info0.DimensionsCount);
            int3 shape0 = new((int)info0.Shape[0], (int)info0.Shape[1], (int)info0.Shape[2]);
            if (!shape0.Equals(this.output0Shape))
            {
                output0Shape = shape0;
                Debug.Log($"New Output 0 shape: {shape0}");

                output0Transposed.Dispose();
                output0Transposed = new NativeArray<float>((int)info0.ElementCount, Allocator.Persistent);

                proposalList.Dispose();
                proposalList = new NativeList<Detection>(shape0.z, Allocator.Persistent);

                detectionList.Dispose();
                detectionList = new NativeList<Detection>(options.maxDetectionCount, Allocator.Persistent);
            }

            // Output 1
            var info1 = outputs[1].GetTensorTypeAndShape();
            Assert.AreEqual(4, info1.DimensionsCount);
            int3 shape1 = new((int)info1.Shape[1], (int)info1.Shape[2], (int)info1.Shape[3]);

            if (segmentation == null || !segmentation.shape.Equals(shape1))
            {
                Debug.Log($"New Output 1 shape: {shape1}");
                segmentation?.Dispose();
                segmentation = new Yolo11SegVisualize(shape1, Colors, options);
            }
        }

        private JobHandle ScheduleGenerateProposalsJob(ReadOnlySpan<float> tensor, NativeList<Detection> proposals, float confidenceThreshold)
        {
            proposals.Clear();

            Assert.AreEqual(1, output0Shape.x, "Support only batch size 1");

            // shape: 116,8400 (XyWh+conf_cls(80)+nm(32), Num_anchors)
            var tensor2D = tensor.AsSpan2D(output0Shape.yz);

            // shape: 8400,116 (Num_anchors, XyWh+conf_cls(80)+nm(32))
            var tensorTransposed = new Span2D<float>(output0Transposed, output0Shape.zy);
            var transposeJobHandle = tensor2D.ScheduleTransposeJob(tensorTransposed);

            // Then generate proposals
            var proposalsWriter = proposals.AsParallelWriter();
            return new GenerateProposalsJob
            {
                output0Transposed = output0Transposed,
                classCount = classCount,
                confidenceThreshold = confidenceThreshold,
                // reciprocal width and height
                sizeScale = new float2(1f / Width, 1f / Height),
                anchorStride = output0Shape.y,
                proposals = proposalsWriter,
            }.Schedule(output0Shape.z, 64, transposeJobHandle);
        }

        [BurstCompile]
        private struct GenerateProposalsJob : IJobParallelFor
        {
            // shape: 8400,116 (Num_anchors, XyWh+conf_cls(80)+nm(32))
            [ReadOnly]
            public NativeArray<float> output0Transposed;
            public int classCount;
            public float confidenceThreshold;
            public float2 sizeScale;
            public int anchorStride;

            [WriteOnly]
            public NativeList<Detection>.ParallelWriter proposals;

            public void Execute(int anchorId)
            {
                var anchor = output0Transposed
                    .AsReadOnlySpan()
                    .Slice(anchorId * anchorStride, anchorStride);

                // Find max confidence
                var confidences = anchor.Slice(4, classCount);
                int classId = confidences.ArgMax();
                float maxConfidence = confidences[classId];

                // Filter out low confidence anchors
                if (maxConfidence < confidenceThreshold)
                {
                    return;
                }

                // Normalize Rect
                float cx = anchor[0] * sizeScale.x;
                float cy = anchor[1] * sizeScale.y;
                float w = anchor[2] * sizeScale.x;
                float h = anchor[3] * sizeScale.y;
                float x = cx - w * 0.5f;
                float y = cy - h * 0.5f;

                proposals.AddNoResize(new Detection(
                    new Rect(x, y, w, h),
                    classId,
                    maxConfidence,
                    anchorId)
                );
            }
        }
    }
}
