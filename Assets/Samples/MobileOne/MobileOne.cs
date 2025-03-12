using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Unity;
using UnityEngine;
using UnityEngine.Assertions;

namespace Microsoft.ML.OnnxRuntime.Examples
{
    /// <summary>
    /// Apple's MobileOne
    /// See LICENSE for full license information.
    /// https://github.com/apple/ml-mobileone
    /// 
    /// Converted Onnx model from PINTO_model_zoo
    /// Licensed under MIT.
    /// https://github.com/PINTO0309/PINTO_model_zoo/tree/main/317_MobileOne
    /// </summary>
    public sealed class MobileOne : ImageInference<float>
    {
        [Serializable]
        public class Options : ImageInferenceOptions
        {
            [Header("MobileOne options")]
            public TextAsset labelFile;
            [Min(1)]
            public int topK = 10;
        }

        public struct Label
        {
            public readonly int index;
            public float score;

            public Label(int index, float score)
            {
                this.index = index;
                this.score = score;
            }
        }

        readonly Options options;
        public readonly Label[] labels;
        public readonly ReadOnlyCollection<string> labelNames;

        public IEnumerable<Label> TopKLabels { get; private set; }

        public MobileOne(byte[] model, Options options)
            : base(model, options)
        {
            this.options = options;
            var info = outputs[0].GetTensorTypeAndShape();
            int length = (int)info.Shape[1];
            labels = new Label[length];

            var labelTexts = options.labelFile.text.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            labelNames = Array.AsReadOnly(labelTexts);

            Assert.AreEqual(length, labelNames.Count,
                $"The labels count does not match to MobileOne output count: {length} != {labelNames.Count}");

            for (int i = 0; i < length; i++)
            {
                labels[i] = new Label(i, 0);
            }
        }

        protected override void PostProcess(IReadOnlyList<OrtValue> outputs)
        {
            // Copy scores to labels
            var output = outputs[0].GetTensorDataAsSpan<float>();
            for (int i = 0; i < output.Length; i++)
            {
                labels[i].score = output[i];
            }
            // sort by score
            TopKLabels = labels
                .OrderByDescending(x => x.score)
                .Take(options.topK);
        }
    }
}
