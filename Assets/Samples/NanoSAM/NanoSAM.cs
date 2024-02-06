using System;
using Microsoft.ML.OnnxRuntime.Unity;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine;
using Unity.Profiling;
using System.Collections.ObjectModel;

namespace Microsoft.ML.OnnxRuntime.Examples
{
    /// <summary>
    /// Nvidia's NanoSAM
    /// https://github.com/NVIDIA-AI-IOT/nanosam/
    /// 
    /// See LICENSE for full license information.
    public sealed class NanoSAM : IDisposable
    {
        public readonly struct Point
        {
            public readonly Vector2 point;
            public readonly float label;

            public Point(Vector2 point, float label)
            {
                this.point = point;
                this.label = label;
            }
        }

        [Serializable]
        public class Options
        {
            public ImageInferenceOptions encoderOptions;
            public ExecutionProviderOptions decoderOptions;
        }

        private readonly NanoSAMEncoder encoder;
        private readonly NanoSAMDecoder decoder;
        private bool disposed;

        static readonly ProfilerMarker runPerfMarker = new($"{typeof(NanoSAM).Name}.Run");

        public ReadOnlySpan<float> OutputMask => decoder.OutputMask;

        public NanoSAM(byte[] encoderModel, byte[] decoderModel, Options options)
        {
            encoder = new NanoSAMEncoder(encoderModel, options.encoderOptions);
            decoder = new NanoSAMDecoder(decoderModel, options.decoderOptions);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private void Dispose(bool disposing)
        {
            if (disposed) { return; }
            if (disposing)
            {
                encoder?.Dispose();
                decoder?.Dispose();
                disposed = true;
            }
        }

        public void Run(Texture texture, Vector2 normalizedPoint)
        {
            var point = new Point(normalizedPoint, 1);
            Run(texture, Array.AsReadOnly(new Point[] { point }));
        }

        public void Run(Texture texture, ReadOnlyCollection<Point> normalizedPoints)
        {
            runPerfMarker.Begin();
            encoder.Run(texture);
            decoder.Run(encoder.ImageEmbeddings, normalizedPoints);
            runPerfMarker.End();
        }

        public void ResetOutput()
        {
            decoder.ResetOutput();
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
        private static readonly Vector2Int ENCODER_SIZE = new(1024, 1024);
        private static readonly Vector2Int MASK_SIZE = new(256, 256);
        private static readonly Vector2 POINT_SCALE = new(ENCODER_SIZE.x, ENCODER_SIZE.y);

        private readonly SessionOptions sessionOptions;
        private readonly RunOptions runOptions;
        private readonly InferenceSession session;


        private readonly OrtValue[] inputs;
        private readonly OrtValue[] outputs;

        public ReadOnlySpan<float> OutputMask => outputs[1].GetTensorDataAsSpan<float>();

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
            // Fill mask
            var mask = inputs[3].GetTensorMutableDataAsSpan<float>();
            mask.Fill(0);

            outputs = new OrtValue[]
            {
                // iou_predictions
                OrtValue.CreateAllocatedTensorValue(allocator, TensorElementType.Float, new long[] { 1, 4 }),
                // low_res_masks
                OrtValue.CreateAllocatedTensorValue(allocator, TensorElementType.Float, new long[] { 1, 4, MASK_SIZE.x, MASK_SIZE.y }),
            };
        }

        public void Dispose()
        {
            runOptions?.Dispose();
            session?.Dispose();
            sessionOptions?.Dispose();
            foreach (var input in inputs)
            {
                input?.Dispose();
            }
            foreach (var output in outputs)
            {
                output?.Dispose();
            }
        }

        public void Run(OrtValue imageEmbeddings, ReadOnlyCollection<NanoSAM.Point> points)
        {
            // image_embeddings
            inputs[0] = imageEmbeddings;

            // point_coords and point_labels
            int length = points.Count;
            if (length == inputs[2].GetTensorTypeAndShape().ElementCount)
            {
                var coords = inputs[1].GetTensorMutableDataAsSpan<float>();
                var labels = inputs[2].GetTensorMutableDataAsSpan<float>();
                SetCoordAndLabels(coords, labels, points);
            }
            else
            {
                var coords = new float[length * 2];
                var labels = new float[length];
                SetCoordAndLabels(coords, labels, points);
                inputs[1].Dispose();
                inputs[2].Dispose();
                inputs[1] = OrtValue.CreateTensorValueFromMemory(coords, new long[] { 1, length, 2 });
                inputs[2] = OrtValue.CreateTensorValueFromMemory(labels, new long[] { 1, length });
            }

            // TODO: mask_input and has_mask_input not used
            // Make example of mask_input

            // Run
            session.Run(runOptions, session.InputNames, inputs, session.OutputNames, outputs);
        }

        public void ResetOutput()
        {
            var output = outputs[1].GetTensorMutableDataAsSpan<float>();
            output.Fill(0);
        }

        private void SetCoordAndLabels(Span<float> coords, Span<float> labels, ReadOnlyCollection<NanoSAM.Point> points)
        {
            int length = points.Count;
            for (int i = 0; i < length; i++)
            {
                var p = points[i];
                coords[i * 2] = p.point.x * POINT_SCALE.x;
                coords[i * 2 + 1] = p.point.y * POINT_SCALE.y;
                labels[i] = p.label;
            }
        }
    }
}
