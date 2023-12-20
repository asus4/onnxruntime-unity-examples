using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime.Unity;
using UnityEngine;
using UnityEngine.Assertions;
using Unity.Profiling;
using Unity.Mathematics;


namespace Microsoft.ML.OnnxRuntime.Examples
{
    /// <summary>
    /// https://github.com/Megvii-BaseDetection/YOLOX
    /// 
    /// Converted Onnx model from PINTO_model_zoo
    /// https://github.com/PINTO0309/PINTO_model_zoo/tree/main/132_YOLOX
    /// </summary>
    public class Yolox : ImageInference<float>
    {
        [Serializable]
        public class Options : ImageInferenceOptions
        {
            public TextAsset labelFile;
            [Range(0f, 1f)]
            public float probThreshold = 0.3f;
        }

        public readonly struct Detection
        {
            public readonly float4 rect;
            public readonly int label;
            public readonly float probability;

            public Detection(float4 rect, int label, float probability)
            {
                this.rect = rect;
                this.label = label;
                this.probability = probability;
            }
        }

        private readonly struct GridAndStride
        {
            public readonly int grid0;
            public readonly int grid1;
            public readonly int stride;

            public GridAndStride(int grid0, int grid1, int stride)
            {
                this.grid0 = grid0;
                this.grid1 = grid1;
                this.stride = stride;
            }
        }

        private const int NUM_CLASSES = 80;
        private readonly string[] labels;
        private readonly GridAndStride[] gridStrides;
        private List<Detection> detections = new();
        private Options options;

        static readonly ProfilerMarker postProcessPerfMarker = new("Yolox.PostProcess");


        public Yolox(byte[] model, Options options)
            : base(model, options)
        {
            this.options = options;

            labels = options.labelFile.text.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            Assert.AreEqual(NUM_CLASSES, labels.Length);
            gridStrides = GenerateGridsAndStrides(width, height);
        }

        public override void Run(Texture texture)
        {
            base.Run(texture);

            postProcessPerfMarker.Begin();
            var output = outputs[0].GetTensorDataAsSpan<float>();
            PostProcess(output);
            postProcessPerfMarker.End();
        }

        private void PostProcess(ReadOnlySpan<float> output)
        {
            var detections = GenerateYoloxProposals(output, options.probThreshold);
            Debug.Log($"detections: {detections.Count}");
        }

        private static GridAndStride[] GenerateGridsAndStrides(int width, int height)
        {
            ReadOnlySpan<int> strides = stackalloc int[] { 8, 16, 32 };
            List<GridAndStride> gridStrides = new();

            foreach (int stride in strides)
            {
                int numGridY = height / stride;
                int numGridX = width / stride;
                for (int g1 = 0; g1 < numGridY; g1++)
                {
                    for (int g0 = 0; g0 < numGridX; g0++)
                    {
                        gridStrides.Add(new GridAndStride(g0, g1, stride));
                    }
                }
            }
            return gridStrides.ToArray();
        }

        private List<Detection> GenerateYoloxProposals(ReadOnlySpan<float> feat_blob, float prob_threshold)
        {
            detections.Clear();

            int num_anchors = gridStrides.Length;

            for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
            {
                var grid = gridStrides[anchor_idx];
                int grid0 = grid.grid0;
                int grid1 = grid.grid1;
                int stride = grid.stride;

                int basic_pos = anchor_idx * (NUM_CLASSES + 5);

                // yolox/models/yolo_head.py decode logic
                float x_center = (feat_blob[basic_pos + 0] + grid0) * stride;
                float y_center = (feat_blob[basic_pos + 1] + grid1) * stride;
                float w = math.exp(feat_blob[basic_pos + 2]) * stride;
                float h = math.exp(feat_blob[basic_pos + 3]) * stride;
                float x0 = x_center - w * 0.5f;
                float y0 = y_center - h * 0.5f;

                float box_objectness = feat_blob[basic_pos + 4];
                for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++)
                {
                    float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
                    float box_prob = box_objectness * box_cls_score;
                    if (box_prob > prob_threshold)
                    {
                        detections.Add(new Detection(
                            new float4(x0, y0, w, h),
                            class_idx,
                            box_prob
                        ));
                    }
                } // class loop
            } // point anchor loop

            return detections;
        }
    }
}
