using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
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
    /// Licensed under Apache-2.0.
    /// See LICENSE for full license information.
    /// https://github.com/Megvii-BaseDetection/YOLOX
    /// 
    /// The included model is downloaded from the following link:
    /// https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx
    /// 
    /// And converted using Runtime optimization: 
    /// python -m onnxruntime.tools.convert_onnx_models_to_ort yolox_nano.onnx --optimization_style Runtime
    /// </summary>
    public class Yolox : ImageInference<float>
    {
        /// <summary>
        /// Options for Yolox
        /// </summary>
        [Serializable]
        public class Options : ImageInferenceOptions
        {
            [Header("Yolox options")]
            public TextAsset labelFile;
            [Range(0f, 1f)]
            public float probThreshold = 0.3f;
            [Range(0f, 1f)]
            public float nmsThreshold = 0.45f;
        }

        public readonly struct Detection : IDetection<Detection>
        {
            public readonly int label;
            public readonly Rect rect;
            public readonly float probability;

            public readonly int Label => label;
            public readonly Rect Rect => rect;

            public Detection(Rect rect, int label, float probability)
            {
                this.rect = rect;
                this.label = label;
                this.probability = probability;
            }

            public int CompareTo(Detection other)
            {
                // Descending Order
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

            public static NativeArray<Anchor> GenerateAnchors(int width, int height, Allocator allocator)
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

                return new NativeArray<Anchor>(anchors.ToArray(), allocator);
            }
        }

        public readonly ReadOnlyCollection<string> labelNames;
        private const int NUM_CLASSES = 80;
        private readonly NativeArray<Anchor> anchors;
        private readonly Options options;

        private NativeArray<float> output0Native;
        private NativeList<Detection> proposalsList;
        private NativeList<Detection> detectionsList;

        public ReadOnlySpan<Detection> Detections => detectionsList.AsReadOnly();


        static readonly ProfilerMarker generateProposalsMarker = new($"{typeof(Yolox).Name}.GenerateProposals");


        public Yolox(byte[] model, Options options)
            : base(model, options)
        {
            this.options = options;

            const int maxDetections = 100;
            const Allocator allocator = Allocator.Persistent;

            proposalsList = new NativeList<Detection>(maxDetections, allocator);
            detectionsList = new NativeList<Detection>(maxDetections, allocator);

            var labels = options.labelFile.text.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            labelNames = Array.AsReadOnly(labels);
            Assert.AreEqual(NUM_CLASSES, labelNames.Count);
            anchors = Anchor.GenerateAnchors(Width, Height, allocator);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                output0Native.Dispose();
                proposalsList.Dispose();
                detectionsList.Dispose();
                anchors.Dispose();
            }
            base.Dispose(disposing);
        }

        protected override void PostProcess(IReadOnlyList<OrtValue> outputs)
        {
            var output0 = outputs[0].GetTensorDataAsSpan<float>();

            generateProposalsMarker.Begin();
            var handle = GenerateProposals(output0, proposalsList, options.probThreshold);
            handle.Complete();
            generateProposalsMarker.End();

            proposalsList.Sort();
            IDetection<Detection>.NMS(proposalsList, detectionsList, options.nmsThreshold);
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

        private JobHandle GenerateProposals(
            in ReadOnlySpan<float> feat_blob,
            NativeList<Detection> result,
            float prob_threshold)
        {
            result.Clear();

            // TODO: Consider using unsafe NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray
            // instead of copying
            if (!output0Native.IsCreated)
            {
                output0Native = new NativeArray<float>(feat_blob.Length, Allocator.Persistent);
            }
            feat_blob.CopyTo(output0Native.AsSpan());

            // Execute the job
            var job = new GenerateProposalsJob
            {
                anchors = anchors,
                featBlob = output0Native,
                widthScale = 1f / Width,
                heightScale = 1f / Height,
                probThreshold = prob_threshold,
                proposals = result.AsParallelWriter()
            };
            return job.Schedule(anchors.Length, 64);
        }

        [BurstCompile]
        private struct GenerateProposalsJob : IJobParallelFor
        {
            [ReadOnly]
            public NativeArray<Anchor> anchors;

            // shape: 1,3549,85
            [ReadOnly]
            public NativeArray<float> featBlob;

            public float widthScale;
            public float heightScale;
            public float probThreshold;

            [WriteOnly]
            public NativeList<Detection>.ParallelWriter proposals;

            public void Execute(int anchorId)
            {
                var anchor = anchors[anchorId];
                int grid0 = anchor.grid0;
                int grid1 = anchor.grid1;
                int stride = anchor.stride;

                int basic_pos = anchorId * (NUM_CLASSES + 5);

                // yolox/models/yolo_head.py decode logic
                float x_center = (featBlob[basic_pos + 0] + grid0) * stride;
                float y_center = (featBlob[basic_pos + 1] + grid1) * stride;
                float w = math.exp(featBlob[basic_pos + 2]) * stride;
                float h = math.exp(featBlob[basic_pos + 3]) * stride;
                // Normalize model space to 0..1
                x_center *= widthScale;
                y_center *= heightScale;
                w *= widthScale;
                h *= heightScale;

                // Skip if out of bounds
                if (x_center < 0 || x_center > 1 || y_center < 0 || y_center > 1)
                {
                    return;
                }

                float x0 = x_center - w * 0.5f;
                float y0 = y_center - h * 0.5f;

                float box_objectness = featBlob[basic_pos + 4];
                for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++)
                {
                    float box_cls_score = featBlob[basic_pos + 5 + class_idx];
                    float box_prob = box_objectness * box_cls_score;
                    if (box_prob > probThreshold)
                    {
                        // Insert with sorted descent order
                        proposals.AddNoResize(new Detection(
                            new Rect(x0, y0, w, h),
                            class_idx,
                            box_prob
                        ));
                    }
                }
            }
        }
    }
}
