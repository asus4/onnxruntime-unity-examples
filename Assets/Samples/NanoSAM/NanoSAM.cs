using System;
using System.Collections;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime.Unity;
using UnityEngine;

namespace Microsoft.ML.OnnxRuntime.Examples
{
    /// <summary>
    /// Nvidia's NanoSAM
    /// https://github.com/NVIDIA-AI-IOT/nanosam/
    /// 
    /// See LICENSE for full license information.
    public sealed class NanoSAM : IDisposable
    {
        [Serializable]
        public class Options
        {
            public ImageInferenceOptions encoderOptions;
            public ExecutionProviderOptions decoderOptions;
        }

        private readonly NanoSAMEncoder encoder;
        private readonly NanoSAMDecoder decoder;

        public NanoSAM(byte[] encoderModel, byte[] decoderModel, Options options)
        {
            encoder = new NanoSAMEncoder(encoderModel, options.encoderOptions);
            decoder = new NanoSAMDecoder(decoderModel, options.decoderOptions);
        }

        public void Dispose()
        {
            encoder?.Dispose();
        }

        public void Run(Texture texture, Vector2 point)
        {
            encoder.Run(texture);
            Debug.Log($"Output: {encoder.Output[0]}");
        }
    }

    internal sealed class NanoSAMEncoder : ImageInference<float>
    {
        public Span<float> Output => outputs[0].GetTensorMutableDataAsSpan<float>();

        public NanoSAMEncoder(byte[] model, ImageInferenceOptions options) : base(model, options)
        {
        }
    }

    internal sealed class NanoSAMDecoder : BasicInference
    {
        public NanoSAMDecoder(byte[] model, ExecutionProviderOptions options) : base(model, options)
        {
        }
    }
}
