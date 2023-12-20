using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Unity;
using UnityEngine;
using UnityEngine.Assertions;

namespace Microsoft.ML.OnnxRuntime.Examples
{
    /// <summary>
    /// Apple's MobileOne
    /// https://github.com/apple/ml-mobileone
    /// 
    /// Converted Onnx model from PINTO_model_zoo
    /// https://github.com/PINTO0309/PINTO_model_zoo/tree/main/317_MobileOne
    /// </summary>
    public sealed class MobileOne : ImageInference<float>
    {
        [System.Serializable]
        public class Options : ImageInferenceOptions
        {
            public TextAsset labelFile;
            [Min(1)]
            public int topK = 10;
        }

        public class Label
        {
            public readonly int index;
            public readonly string name;
            public float score;

            public Label(int index, string name, float score)
            {
                this.index = index;
                this.name = name;
                this.score = score;
            }
        }

        private readonly int topK;
        public readonly Label[] labels;
        public IEnumerable<Label> TopKLabels;

        public MobileOne(byte[] model, Options options)
            : base(model, options)
        {
            this.topK = options.topK;
            var info = outputs[0].GetTensorTypeAndShape();
            int length = (int)info.Shape[1];
            labels = new Label[length];

            string[] labelNames = options.labelFile.text.Split('\n');
            Assert.AreEqual(length, labelNames.Length,
                $"The labels count does not match to MobileOne output count: {length} != {labelNames.Length}");

            for (int i = 0; i < length; i++)
            {
                labels[i] = new Label(i, labelNames[i], 0);
            }
        }

        public override void Run(Texture texture)
        {
            base.Run(texture);

            // Copy scores to labels
            var output = outputs[0].GetTensorDataAsSpan<float>();
            for (int i = 0; i < output.Length; i++)
            {
                labels[i].score = output[i];
            }
            // sort by score
            TopKLabels = labels.OrderByDescending(x => x.score).Take(topK);
        }
    }
}
