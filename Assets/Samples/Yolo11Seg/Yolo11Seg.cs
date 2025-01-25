using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
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
    /// </summary>
    public class Yolo11Seg : ImageInference<float>
    {
        [Serializable]
        public class Options : ImageInferenceOptions
        {
            [Header("Yolo11Seg options")]
            public TextAsset labelFile;
            [Range(0f, 1f)]
            public float probThreshold = 0.3f;
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
            public readonly float x;
            public readonly float y;

            public Anchor(float x, float y)
            {
                this.x = x;
                this.y = y;
            }
        }

        private readonly Options options;
        private readonly string[] labels;
        private readonly int maxDetections;
        private readonly float probThreshold;
        private readonly float nmsThreshold;

        public Yolo11Seg(byte[] model, Options options)
            : base(model, options)
        {
            this.options = options;
        }
        public override void Dispose()
        {
            base.Dispose();
        }
    }
}
