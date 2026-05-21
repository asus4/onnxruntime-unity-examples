using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using Microsoft.ML.OnnxRuntime.Unity;
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
    /// PP-PicoDet anchor-free object detector with Distribution Focal Loss (DFL) box regression.
    ///
    /// Licensed under Apache-2.0.
    /// See LICENSE for full license information.
    /// https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.9/configs/picodet
    ///
    /// Reference implementations:
    /// - Python: deploy/python/picodet_postprocess.py
    /// - C++:    deploy/third_engine/demo_openvino/picodet_openvino.cpp
    /// </summary>
    public class PicoDet : ImageInference<float>
    {
        [Serializable]
        public class Options : ImageInferenceOptions
        {
            [Header("PicoDet options")]
            public TextAsset labelFile;
            [Range(0f, 1f)]
            public float probThreshold = 0.3f;
            [Range(0f, 1f)]
            public float nmsThreshold = 0.6f;
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
                // Descending order
                return other.probability.CompareTo(probability);
            }
        }

        private readonly struct Anchor
        {
            public readonly float centerX;
            public readonly float centerY;
            public readonly int stride;
            public readonly int strideIndex;
            public readonly int cellIndex;

            public Anchor(float centerX, float centerY, int stride, int strideIndex, int cellIndex)
            {
                this.centerX = centerX;
                this.centerY = centerY;
                this.stride = stride;
                this.strideIndex = strideIndex;
                this.cellIndex = cellIndex;
            }

            public static NativeArray<Anchor> GenerateAnchors(int width, int height, Allocator allocator)
            {
                List<Anchor> anchors = new();
                for (int s = 0; s < Strides.Length; s++)
                {
                    int stride = Strides[s];
                    // Ceil division mirrors the FPN downsampling for inputs not cleanly divisible by stride
                    // (e.g. 416 / 64 → 7×7 grid, not 6×6).
                    int gridH = (height + stride - 1) / stride;
                    int gridW = (width + stride - 1) / stride;
                    int cell = 0;
                    for (int gy = 0; gy < gridH; gy++)
                    {
                        for (int gx = 0; gx < gridW; gx++)
                        {
                            float cx = (gx + 0.5f) * stride;
                            float cy = (gy + 0.5f) * stride;
                            anchors.Add(new Anchor(cx, cy, stride, s, cell));
                            cell++;
                        }
                    }
                }
                return new NativeArray<Anchor>(anchors.ToArray(), allocator);
            }
        }

        public readonly ReadOnlyCollection<string> labelNames;

        private const int NUM_CLASSES = 80;
        private const int REG_MAX = 7;
        private const int REG_BINS = REG_MAX + 1; // 8 bins per box side
        private const int BOX_CHANNELS = REG_BINS * 4; // 32

        private static readonly int[] Strides = { 8, 16, 32, 64 };

        private readonly Options options;
        private readonly NativeArray<Anchor> anchors;

        // 4 score buffers (NUM_CLASSES per anchor) and 4 box buffers (BOX_CHANNELS per anchor)
        private NativeArray<float> scores0, scores1, scores2, scores3;
        private NativeArray<float> boxes0, boxes1, boxes2, boxes3;

        private NativeList<Detection> proposalsList;
        private NativeList<Detection> detectionsList;

        public ReadOnlySpan<Detection> Detections => detectionsList.AsReadOnly();

        static readonly ProfilerMarker generateProposalsMarker = new($"{typeof(PicoDet).Name}.GenerateProposals");

        public PicoDet(byte[] model, Options options)
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
                if (scores0.IsCreated) scores0.Dispose();
                if (scores1.IsCreated) scores1.Dispose();
                if (scores2.IsCreated) scores2.Dispose();
                if (scores3.IsCreated) scores3.Dispose();
                if (boxes0.IsCreated) boxes0.Dispose();
                if (boxes1.IsCreated) boxes1.Dispose();
                if (boxes2.IsCreated) boxes2.Dispose();
                if (boxes3.IsCreated) boxes3.Dispose();
                proposalsList.Dispose();
                detectionsList.Dispose();
                anchors.Dispose();
            }
            base.Dispose(disposing);
        }

        protected override void PostProcess(IReadOnlyList<OrtValue> outputs)
        {
            // session.OutputNames order (verified): 4 score tensors then 4 DFL box tensors,
            // each block ordered by stride 8/16/32/64.
            CopyOutput(outputs[0], ref scores0);
            CopyOutput(outputs[1], ref scores1);
            CopyOutput(outputs[2], ref scores2);
            CopyOutput(outputs[3], ref scores3);
            CopyOutput(outputs[4], ref boxes0);
            CopyOutput(outputs[5], ref boxes1);
            CopyOutput(outputs[6], ref boxes2);
            CopyOutput(outputs[7], ref boxes3);

            generateProposalsMarker.Begin();
            proposalsList.Clear();
            var job = new GenerateProposalsJob
            {
                anchors = anchors,
                scores0 = scores0,
                scores1 = scores1,
                scores2 = scores2,
                scores3 = scores3,
                boxes0 = boxes0,
                boxes1 = boxes1,
                boxes2 = boxes2,
                boxes3 = boxes3,
                widthScale = 1f / Width,
                heightScale = 1f / Height,
                probThreshold = options.probThreshold,
                proposals = proposalsList.AsParallelWriter(),
            };
            job.Schedule(anchors.Length, 64).Complete();
            generateProposalsMarker.End();

            proposalsList.Sort();
            DetectionUtil.NMS(proposalsList, detectionsList, options.nmsThreshold);
        }

        private static void CopyOutput(OrtValue value, ref NativeArray<float> buffer)
        {
            var src = value.GetTensorDataAsSpan<float>();
            if (!buffer.IsCreated || buffer.Length != src.Length)
            {
                if (buffer.IsCreated) buffer.Dispose();
                buffer = new NativeArray<float>(src.Length, Allocator.Persistent);
            }
            src.CopyTo(buffer.AsSpan());
        }

        /// <summary>
        /// Convert a normalized model-space rect (0..1) to viewport space.
        /// </summary>
        public Rect ConvertToViewport(in Rect rect)
        {
            Rect unityRect = rect.FlipY();
            var mtx = InputToViewportMatrix;
            Vector2 min = mtx.MultiplyPoint3x4(unityRect.min);
            Vector2 max = mtx.MultiplyPoint3x4(unityRect.max);
            return new Rect(min, max - min);
        }

        [BurstCompile]
        private struct GenerateProposalsJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<Anchor> anchors;

            [ReadOnly] public NativeArray<float> scores0;
            [ReadOnly] public NativeArray<float> scores1;
            [ReadOnly] public NativeArray<float> scores2;
            [ReadOnly] public NativeArray<float> scores3;

            [ReadOnly] public NativeArray<float> boxes0;
            [ReadOnly] public NativeArray<float> boxes1;
            [ReadOnly] public NativeArray<float> boxes2;
            [ReadOnly] public NativeArray<float> boxes3;

            public float widthScale;
            public float heightScale;
            public float probThreshold;

            [WriteOnly] public NativeList<Detection>.ParallelWriter proposals;

            public void Execute(int anchorId)
            {
                Anchor anchor = anchors[anchorId];
                int cell = anchor.cellIndex;
                int scoreBase = cell * NUM_CLASSES;
                int boxBase = cell * BOX_CHANNELS;

                // Resolve the score/box slice for this stride level.
                float maxScore = 0f;
                int maxLabel = 0;
                switch (anchor.strideIndex)
                {
                    case 0: ArgMax(scores0, scoreBase, out maxScore, out maxLabel); break;
                    case 1: ArgMax(scores1, scoreBase, out maxScore, out maxLabel); break;
                    case 2: ArgMax(scores2, scoreBase, out maxScore, out maxLabel); break;
                    default: ArgMax(scores3, scoreBase, out maxScore, out maxLabel); break;
                }

                if (maxScore < probThreshold)
                {
                    return;
                }

                // DFL: split 32-ch into 4 sides (left, top, right, bottom),
                // softmax over 8 bins, then sum(i * p_i) * stride.
                float distLeft, distTop, distRight, distBottom;
                switch (anchor.strideIndex)
                {
                    case 0:
                        distLeft = DflIntegral(boxes0, boxBase + 0 * REG_BINS);
                        distTop = DflIntegral(boxes0, boxBase + 1 * REG_BINS);
                        distRight = DflIntegral(boxes0, boxBase + 2 * REG_BINS);
                        distBottom = DflIntegral(boxes0, boxBase + 3 * REG_BINS);
                        break;
                    case 1:
                        distLeft = DflIntegral(boxes1, boxBase + 0 * REG_BINS);
                        distTop = DflIntegral(boxes1, boxBase + 1 * REG_BINS);
                        distRight = DflIntegral(boxes1, boxBase + 2 * REG_BINS);
                        distBottom = DflIntegral(boxes1, boxBase + 3 * REG_BINS);
                        break;
                    case 2:
                        distLeft = DflIntegral(boxes2, boxBase + 0 * REG_BINS);
                        distTop = DflIntegral(boxes2, boxBase + 1 * REG_BINS);
                        distRight = DflIntegral(boxes2, boxBase + 2 * REG_BINS);
                        distBottom = DflIntegral(boxes2, boxBase + 3 * REG_BINS);
                        break;
                    default:
                        distLeft = DflIntegral(boxes3, boxBase + 0 * REG_BINS);
                        distTop = DflIntegral(boxes3, boxBase + 1 * REG_BINS);
                        distRight = DflIntegral(boxes3, boxBase + 2 * REG_BINS);
                        distBottom = DflIntegral(boxes3, boxBase + 3 * REG_BINS);
                        break;
                }

                float stride = anchor.stride;
                float xMin = (anchor.centerX - distLeft * stride) * widthScale;
                float yMin = (anchor.centerY - distTop * stride) * heightScale;
                float xMax = (anchor.centerX + distRight * stride) * widthScale;
                float yMax = (anchor.centerY + distBottom * stride) * heightScale;

                proposals.AddNoResize(new Detection(
                    Rect.MinMaxRect(xMin, yMin, xMax, yMax),
                    maxLabel,
                    maxScore));
            }

            private static void ArgMax(in NativeArray<float> buffer, int baseIndex, out float maxScore, out int maxLabel)
            {
                maxScore = buffer[baseIndex];
                maxLabel = 0;
                for (int c = 1; c < NUM_CLASSES; c++)
                {
                    float s = buffer[baseIndex + c];
                    if (s > maxScore)
                    {
                        maxScore = s;
                        maxLabel = c;
                    }
                }
            }

            private static float DflIntegral(in NativeArray<float> buffer, int baseIndex)
            {
                // Numerically-stable softmax: subtract max before exp.
                float max = buffer[baseIndex];
                for (int i = 1; i < REG_BINS; i++)
                {
                    float v = buffer[baseIndex + i];
                    if (v > max) max = v;
                }

                float sumExp = 0f;
                float weighted = 0f;
                for (int i = 0; i < REG_BINS; i++)
                {
                    float e = math.exp(buffer[baseIndex + i] - max);
                    sumExp += e;
                    weighted += i * e;
                }
                return weighted / sumExp;
            }
        }
    }
}
