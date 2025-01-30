using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Runtime.CompilerServices;
using Microsoft.ML.OnnxRuntime.Unity;
using UnityEngine;
using UnityEngine.Assertions;
using Unity.Collections;
using Unity.Mathematics;
using System.IO;


namespace Microsoft.ML.OnnxRuntime.Examples
{
    /// <summary>
    /// Licensed under AGPL-3.0 license
    /// See LICENSE for full license information.
    /// https://github.com/ultralytics/ultralytics/blob/main/LICENSE
    /// 
    /// https://docs.ultralytics.com/tasks/segment/
    /// </summary>
    public class Yolo11Seg : ImageInference<float>
    {
        [Serializable]
        public class Options : ImageInferenceOptions
        {
            [Header("YOLO options")]
            public TextAsset labelFile;
            [Range(1, 100)]
            public int maxDetections = 100;
            [Range(0f, 1f)]
            public float confidenceThreshold = 0.25f;
            [Range(0f, 1f)]
            public float nmsThreshold = 0.45f;
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

        private readonly Options options;
        public readonly int classCount;
        public readonly ReadOnlyCollection<string> labelNames;

        // [0: predictions] shape: 1,116,8400 (Batch_size=1, xywh_conf_cls_nm, Num_anchors)
        private readonly int[] output0Shape;
        // [1: protos] shape: 1,32,160,160
        private readonly int[] output1Shape;
        private readonly float[] tOutput0; // transpose of output0
        private NativeArray<Detection> proposalsArray;
        private NativeArray<Detection> detectionsArray;
        private int detectionCount = 0;

        public Yolo11Seg(byte[] model, Options options)
            : base(model, options)
        {
            this.options = options;

            Assert.AreEqual(2, outputs.Count);
            output0Shape = Array.ConvertAll(outputs[0].GetTensorTypeAndShape().Shape, x => (int)x);
            output1Shape = Array.ConvertAll(outputs[1].GetTensorTypeAndShape().Shape, x => (int)x);

            int maxDetections = options.maxDetections;
            proposalsArray = new NativeArray<Detection>(maxDetections, Allocator.Persistent);
            detectionsArray = new NativeArray<Detection>(maxDetections, Allocator.Persistent);

            var labels = options.labelFile.text.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            labelNames = Array.AsReadOnly(labels);
            classCount = labelNames.Count;

            Assert.IsTrue(classCount <= output0Shape[1] - 4); // 4:xywh

            tOutput0 = new float[outputs[0].GetTensorDataAsSpan<float>().Length];
        }

        public override void Dispose()
        {
            base.Dispose();
            proposalsArray.Dispose();
            detectionsArray.Dispose();
        }

        protected override void PostProcess()
        {
            // https://github.com/ultralytics/ultralytics/blob/aecf1da32b50e144b226f54ae4e979dc3cc62145/examples/YOLOv8-Segmentation-ONNXRuntime-Python/main.py#L109

            // [1: protos] shape: 1,32,160,160

            var output0 = outputs[0].GetTensorDataAsSpan<float>();
            // Transpose(output0, tOutput0, output0Shape[1], output0Shape[2]);

            var proposals = GenerateProposals(output0, options.confidenceThreshold);
            if (proposals.Length == 0)
            {
                detectionCount = 0;
                return;
            }
            proposals.Sort();
            // Debug.Log($"Max conf: {proposals[0].probability} min conf: {proposals[^1].probability}");
        }

        public void SaveOutputToFile(string filePath)
        {
            var output0 = outputs[0].GetTensorDataAsSpan<float>();
            // var output0 = tOutput0;
            var sb = new System.Text.StringBuilder();
            for (int i = 0; i < output0.Length; i++)
            {
                sb.Append($"{output0[i]:F2}");
                sb.Append(", ");
                if (i % 116 == 115)
                // if (i % 8400 == 8399)
                {
                    sb.AppendLine();
                }
            }
            File.WriteAllText(filePath, sb.ToString());
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static int Idx(int x, int y, int stride)
        {
            return x * stride + y;
        }

        private NativeSlice<Detection> GenerateProposals(
            in ReadOnlySpan<float> predictions, float confidenceThreshold)
        {
            // [0: predictions] shape: 1,116,8400 (Batch_size=1, xywh+conf_cls(80)+nm(32), Num_anchors)

            int stride = output0Shape[1];
            int anchorCount = output0Shape[2];
            int classCount = this.classCount;
            int width = this.width;
            int height = this.height;

            int proposalsCount = 0;

            // Y (dim:2)
            for (int anchorId = 0; anchorId < anchorCount; anchorCount++)
            {

                int classId = int.MinValue;
                float maxConfidence = float.MinValue;

                // X (dim:1)
                for (int i = 0; i < classCount; i++)
                {
                    float confidence = predictions[Idx(i + 4, anchorId, stride)];
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
                float cx = predictions[Idx(0, anchorId, stride)] / width;
                float cy = predictions[Idx(1, anchorId, stride)] / height;
                float w = predictions[Idx(2, anchorId, stride)] / width;
                float h = predictions[Idx(3, anchorId, stride)] / height;
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

        static void Transpose(ReadOnlySpan<float> input, Span<float> output, int width, int height)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    output[x * height + y] = input[y * width + x];
                }
            }
        }
    }
}
