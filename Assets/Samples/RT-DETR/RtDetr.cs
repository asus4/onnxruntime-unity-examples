using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using Microsoft.ML.OnnxRuntime.Unity;
using Microsoft.ML.OnnxRuntime.UnityEx;
using UnityEngine;

namespace Microsoft.ML.OnnxRuntime.Examples
{
    /// <summary>
    /// RT-DETRv2: RT-DETRv2 Beat YOLOs on Real-time Object Detection
    /// 
    /// Licensed under Apache License 2.0
    /// See the original source code at:
    /// https://github.com/lyuwenyu/RT-DETR
    /// </summary>
    public class RtDetr : ImageInference<float>
    {
        [Serializable]
        public class Options : ImageInferenceOptions
        {
            [Header("RT-DETR options")]
            public TextAsset labelFile;
            [Range(0f, 1f)]
            public float probThreshold = 0.3f;
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

        private readonly Options options;
        private Detection[] detections;
        private int detectedCount;
        private Vector2Int originalTargetSize;
        public readonly ReadOnlyCollection<string> labelNames;

        public ReadOnlySpan<Detection> Detections
            => new(detections, 0, detectedCount);

        public RtDetr(byte[] model, Options options)
            : base(model, options)
        {
            this.options = options;

            var labels = options.labelFile.text.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            labelNames = Array.AsReadOnly(labels);

            detections = new Detection[300];
        }

        protected override void PreProcess(Texture texture)
        {
            base.PreProcess(texture);

            // Set [orig_target_sizes]
            originalTargetSize = new Vector2Int(texture.width, texture.height);
            var sizes = inputs[1].GetTensorMutableDataAsSpan<long>();
            sizes[0] = originalTargetSize.x;
            sizes[1] = originalTargetSize.y;
        }

        protected override void PostProcess(IReadOnlyList<OrtValue> outputs)
        {
            var labels = outputs[0].GetTensorDataAsSpan<long>();
            var boxes = outputs[1].GetTensorDataAsSpan<float>();
            var scores = outputs[2].GetTensorDataAsSpan<float>();

            if (detections.Length != scores.Length)
            {
                Debug.Log($"Resizing detection buffer from {detections.Length} to {scores.Length}");
                detections = new Detection[scores.Length];
            }

            float widthScale = 1f / originalTargetSize.x;
            float heightScale = 1f / originalTargetSize.y;
            int added = 0;

            for (int i = 0; i < detections.Length; i++)
            {
                if (scores[i] <= options.probThreshold)
                {
                    continue;
                }

                float x_min = boxes[i * 4 + 0] * widthScale;
                float y_min = boxes[i * 4 + 1] * heightScale;
                float x_max = boxes[i * 4 + 2] * widthScale;
                float y_max = boxes[i * 4 + 3] * heightScale;

                detections[added] = new Detection(
                    Rect.MinMaxRect(x_min, y_min, x_max, y_max),
                    (int)labels[i],
                    scores[i]
                );
                added++;
            }

            detectedCount = added;
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
    }
}
