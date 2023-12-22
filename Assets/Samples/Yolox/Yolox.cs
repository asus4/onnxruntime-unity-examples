using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime.Unity;
using UnityEngine;
using UnityEngine.Assertions;
using Unity.Collections;
using Unity.Mathematics;


namespace Microsoft.ML.OnnxRuntime.Examples
{
    /// <summary>
    /// Licensed under Apache-2.0.
    /// See LICENSE for full license information.
    /// https://github.com/Megvii-BaseDetection/YOLOX
    /// 
    /// Converted Onnx model from PINTO_model_zoo
    /// Licensed under MIT.
    /// https://github.com/PINTO0309/PINTO_model_zoo/tree/main/132_YOLOX
    /// </summary>
    public class Yolox : ImageInference<float>
    {
        /// <summary>
        /// Options for Yolox
        /// </summary>
        [Serializable]
        public class Options : ImageInferenceOptions
        {
            public TextAsset labelFile;
            [Range(1, 100)]
            public int maxDetections = 100;
            [Range(0f, 1f)]
            public float probThreshold = 0.3f;
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
                return other.probability.CompareTo(probability);
            }
        }

        private readonly struct Anchor
        {
            public readonly int grid0;
            public readonly int grid1;
            public readonly int stride;

            public Anchor(int grid0, int grid1, int stride)
            {
                this.grid0 = grid0;
                this.grid1 = grid1;
                this.stride = stride;
            }

            public static Anchor[] GenerateAnchors(int width, int height)
            {
                ReadOnlySpan<int> strides = stackalloc int[] { 8, 16, 32 };
                List<Anchor> anchors = new();

                foreach (int stride in strides)
                {
                    int numGridY = height / stride;
                    int numGridX = width / stride;
                    for (int g1 = 0; g1 < numGridY; g1++)
                    {
                        for (int g0 = 0; g0 < numGridX; g0++)
                        {
                            anchors.Add(new Anchor(g0, g1, stride));
                        }
                    }
                }
                return anchors.ToArray();
            }
        }

        private const int NUM_CLASSES = 80;
        public readonly string[] labels;
        private readonly Anchor[] anchors;
        private readonly Options options;

        private NativeArray<Detection> proposalsArray;
        private NativeArray<Detection> pickedArray;
        private int detectionCount = 0;

        public ReadOnlySpan<string> LabelNames => labels;
        public ReadOnlySpan<Detection> Detections => pickedArray.AsReadOnlySpan()[..detectionCount];

        public Yolox(byte[] model, Options options)
            : base(model, options)
        {
            this.options = options;

            int maxDetections = options.maxDetections;
            proposalsArray = new NativeArray<Detection>(maxDetections, Allocator.Persistent);
            pickedArray = new NativeArray<Detection>(maxDetections, Allocator.Persistent);

            labels = options.labelFile.text.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            Assert.AreEqual(NUM_CLASSES, labels.Length);
            anchors = Anchor.GenerateAnchors(width, height);
        }

        public override void Dispose()
        {
            base.Dispose();
            proposalsArray.Dispose();
            pickedArray.Dispose();
        }

        protected override void PostProcess()
        {
            var output = outputs[0].GetTensorDataAsSpan<float>();
            var proposals = GenerateProposals(output, options.probThreshold);
            proposals.Sort();
            detectionCount = NMS(proposals, pickedArray, options.nmsThreshold);
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

        private NativeSlice<Detection> GenerateProposals(
            in ReadOnlySpan<float> feat_blob, float prob_threshold)
        {
            int num_anchors = anchors.Length;

            float widthScale = 1f / width;
            float heightScale = 1f / height;

            int proposalsCount = 0;

            for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
            {
                var anchor = anchors[anchor_idx];
                int grid0 = anchor.grid0;
                int grid1 = anchor.grid1;
                int stride = anchor.stride;

                int basic_pos = anchor_idx * (NUM_CLASSES + 5);

                // yolox/models/yolo_head.py decode logic
                float x_center = (feat_blob[basic_pos + 0] + grid0) * stride;
                float y_center = (feat_blob[basic_pos + 1] + grid1) * stride;
                float w = math.exp(feat_blob[basic_pos + 2]) * stride;
                float h = math.exp(feat_blob[basic_pos + 3]) * stride;
                // Normalize model space to 0..1
                x_center *= widthScale;
                y_center *= heightScale;
                w *= widthScale;
                h *= heightScale;

                // Skip if out of bounds
                if (x_center < 0 || x_center > 1 || y_center < 0 || y_center > 1)
                {
                    continue;
                }

                float x0 = x_center - w * 0.5f;
                float y0 = y_center - h * 0.5f;

                float box_objectness = feat_blob[basic_pos + 4];
                for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++)
                {
                    float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
                    float box_prob = box_objectness * box_cls_score;
                    if (box_prob > prob_threshold)
                    {
                        // Insert with sorted descent order
                        proposalsArray[proposalsCount] = new Detection(
                            new Rect(x0, y0, w, h),
                            class_idx,
                            box_prob
                        );
                        proposalsCount++;

                        if (proposalsCount >= proposalsArray.Length)
                        {
                            break;
                        }
                    }
                }

                if (proposalsCount >= proposalsArray.Length)
                {
                    break;
                }
            }

            if (proposalsCount == 0)
            {
                return proposalsArray.Slice(0, 0);
            }

            return proposalsArray.Slice(0, Math.Min(proposalsCount, proposalsArray.Length));
        }

        private static int NMS(
            in NativeSlice<Detection> proposals,
            NativeArray<Detection> picked,
            float iou_threshold)
        {
            int pickedCount = 0;

            foreach (Detection a in proposals)
            {
                bool keep = true;
                for (int i = 0; i < pickedCount; i++)
                {
                    Detection b = picked[i];

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
                    picked[pickedCount] = a;
                    pickedCount++;
                }
            }

            return pickedCount;
        }
    }
}
