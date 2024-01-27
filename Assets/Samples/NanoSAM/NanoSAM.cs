using System;
using Microsoft.ML.OnnxRuntime.Unity;
using Microsoft.ML.OnnxRuntime.Tensors;
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
            decoder.Run(encoder.ImageEmbeddings, point);
        }
    }

    internal sealed class NanoSAMEncoder : ImageInference<float>
    {
        public OrtValue ImageEmbeddings => outputs[0];

        public NanoSAMEncoder(byte[] model, ImageInferenceOptions options) : base(model, options)
        {
        }
    }

    internal sealed class NanoSAMDecoder : IDisposable
    {
        private readonly SessionOptions sessionOptions;
        private readonly RunOptions runOptions;
        private readonly InferenceSession session;

        private readonly OrtValue[] inputs;

        public NanoSAMDecoder(byte[] model, ExecutionProviderOptions options)
        {
            try
            {
                sessionOptions = new SessionOptions();
                options.AppendExecutionProviders(sessionOptions);
                session = new InferenceSession(model, sessionOptions);
                runOptions = new RunOptions();
            }
            catch (Exception e)
            {
                session?.Dispose();
                sessionOptions?.Dispose();
                throw e;
            }
            session.LogIOInfo();

            /*
            Version: 1.16.3
            Input:
            [image_embeddings] shape: 1,256,64,64, type: System.Single
            [point_coords] shape: 1,-1,2, type: System.Single
            [point_labels] shape: 1,-1, type: System.Single
            [mask_input] shape: 1,1,256,256, type: System.Single
            [has_mask_input] shape: 1, type: System.Single

            Output:
            [iou_predictions] shape: -1,4, type: System.Single
            [low_res_masks] shape: -1,-1,-1,-1, type: System.Single
            */

            // Allocate inputs/outputs
            var inputMetadata = session.InputMetadata;

            var allocator = OrtAllocator.DefaultInstance;
            inputs = new OrtValue[]
            {
                // image_embeddings
                null, // shared from encoder
                // point_coords
                OrtValue.CreateAllocatedTensorValue(allocator, TensorElementType.Float, new long[] { 1, 1, 2 }),
                // point_labels
                OrtValue.CreateAllocatedTensorValue(allocator, TensorElementType.Float, new long[] { 1, 1 }),
                // mask_input
                inputMetadata["mask_input"].CreateTensorOrtValue(),
                // has_mask_input
                OrtValue.CreateTensorValueFromMemory(new float[]{ 0 }, new long[] { 1 }),
            };
        }

        public void Dispose()
        {
            runOptions?.Dispose();
            session?.Dispose();
            sessionOptions?.Dispose();
            foreach (var input in inputs)
            {
                input.Dispose();
            }
        }

        public void Run(OrtValue imageEmbeddings, Vector2 point)
        {
            // image_embeddings
            inputs[0] = imageEmbeddings;

            // point_coords
            var pointCoords = inputs[1].GetTensorMutableDataAsSpan<float>();
            pointCoords[0] = point.x;
            pointCoords[1] = point.y;

            // point_labels
            var pointLabels = inputs[2].GetTensorMutableDataAsSpan<float>();
            pointLabels[0] = 0;

            // mask_input and has_mask_input not used

            // Run
            using var outputs = session.Run(runOptions, session.InputNames, inputs, session.OutputNames);
            foreach (var output in outputs)
            {
                var info = output.GetTensorTypeAndShape();
                Debug.Log($"Output type:{info.ElementDataType}, shape:[{string.Join(",", info.Shape)}]");
            }
        }
    }
}
