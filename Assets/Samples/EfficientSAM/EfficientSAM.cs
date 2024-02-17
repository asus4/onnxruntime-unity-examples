using System;
using System.Collections.ObjectModel;
using Microsoft.ML.OnnxRuntime.Unity;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine;
using Unity.Profiling;

namespace Microsoft.ML.OnnxRuntime.Examples
{
    /// <summary>
    /// EfficientSAM
    /// https://github.com/yformer/EfficientSAM
    /// 
    /// See LICENSE for full license information.
    public sealed class EfficientSAM : IDisposable
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
        public class Options : ImageInferenceOptions
        {
            public int maxResolution = 512;
        }

        private readonly Options options;
        private readonly InferenceSession session;
        private readonly SessionOptions sessionOptions;
        private readonly RunOptions runOptions;
        private readonly OrtValue[] inputs;
        private readonly string[] outputNames;
        private readonly OrtValue[] outputs;

        private TextureToTensor<float> textureToTensor;
        private bool disposed;

        static readonly ProfilerMarker preprocessPerfMarker = new($"{typeof(EfficientSAM).Name}.Preprocess");
        static readonly ProfilerMarker runProfMarker = new($"{typeof(EfficientSAM).Name}.Run");

        public Vector2Int InputSize => new(textureToTensor.width, textureToTensor.height);
        public ReadOnlySpan<float> OutputMask => outputs[0].GetTensorDataAsSpan<float>();

        public EfficientSAM(byte[] encoderModel, Options options)
        {
            this.options = options;
            if (Application.platform == RuntimePlatform.Android)
            {
                Debug.LogWarning("Fallback to CPU on Android");
                // Use the CPU backend for Android
                options.executionProvider.executionProviderPriorities[0] = ExecutionProviderPriority.None;
            }

            try
            {
                sessionOptions = new SessionOptions();
                options.executionProvider.AppendExecutionProviders(sessionOptions);
                session = new InferenceSession(encoderModel, sessionOptions);
                runOptions = new RunOptions();
            }
            catch (Exception e)
            {
                session?.Dispose();
                sessionOptions?.Dispose();
                runOptions?.Dispose();
                throw e;
            }
            session.LogIOInfo();

            /*
            Input:
            [batched_images] shape: -1,3,-1,-1, type: System.Single isTensor: True
            [batched_point_coords] shape: 1,1,-1,2, type: System.Single isTensor: True
            [batched_point_labels] shape: 1,1,-1, type: System.Single isTensor: True

            Output:
            [output_masks] shape: 1,1,-1,-1,-1, type: System.Single isTensor: True
            [iou_predictions] shape: 1,1,-1, type: System.Single isTensor: True
            [onnx::Shape_2776] shape: -1,-1,-1,-1, type: System.Single isTensor: True
            */

            inputs = new OrtValue[3];
            // requires only masks
            outputNames = new string[] { "output_masks" };
            outputs = new OrtValue[1];
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
                textureToTensor?.Dispose();
                session?.Dispose();
                sessionOptions?.Dispose();
                runOptions?.Dispose();
                foreach (var input in inputs)
                {
                    input?.Dispose();
                }
                foreach (var output in outputs)
                {
                    output?.Dispose();
                }
                disposed = true;
            }
        }

        public void Run(Texture texture, ReadOnlyCollection<Point> normalizedPoints)
        {
            // Preprocess
            preprocessPerfMarker.Begin();
            (int width, int height) = LimitInputSize(texture, options.maxResolution);
            EnsureInputsOutputs(width, height, normalizedPoints.Count);
            textureToTensor.Transform(texture, options.aspectMode);

            var imageSpan = inputs[0].GetTensorMutableDataAsSpan<float>();
            textureToTensor.TensorData.CopyTo(imageSpan);

            var coords = inputs[1].GetTensorMutableDataAsSpan<float>();
            var labels = inputs[2].GetTensorMutableDataAsSpan<float>();
            SetCoordAndLabels(coords, labels, normalizedPoints, width, height);
            preprocessPerfMarker.End();

            // Run session
            runProfMarker.Begin();
            session.Run(runOptions, session.InputNames, inputs, outputNames, outputs);
            runProfMarker.End();

            // Postprocess
            // long[] shape = outputs[0].GetTypeInfo().TensorTypeAndShapeInfo.Shape;
            // Debug.Log($"Output shape: {string.Join(",", shape)}");
        }

        public void ResetOutput()
        {
            // Fill mask
            var mask = outputs[0].GetTensorMutableDataAsSpan<float>();
            mask.Fill(0f);
        }

        private static (int width, int height) LimitInputSize(Texture texture, int maxSize)
        {
            if (texture.width <= maxSize && texture.height <= maxSize)
            {
                return (texture.width, texture.height);
            }
            float aspect = texture.width / (float)texture.height;
            return aspect > 1
                ? (maxSize, Mathf.RoundToInt(texture.height * (maxSize / (float)texture.width)))
                : (Mathf.RoundToInt(texture.width * (maxSize / (float)texture.height)), maxSize);
        }

        private void EnsureInputsOutputs(int width, int height, int pointCount)
        {
            if (textureToTensor == null || textureToTensor.width != width || textureToTensor.height != height)
            {
                textureToTensor?.Dispose();
                textureToTensor = new TextureToTensor<float>(width, height);

                inputs[0]?.Dispose();
                inputs[0] = OrtValue.CreateAllocatedTensorValue(
                    OrtAllocator.DefaultInstance, TensorElementType.Float,
                    new long[] { 1, 3, height, width });

                outputs[0]?.Dispose();
                outputs[0] = OrtValue.CreateAllocatedTensorValue(
                    OrtAllocator.DefaultInstance, TensorElementType.Float,
                    new long[] { 1, 1, 3, height, width });
            }

            if (inputs[2] == null || inputs[2].GetTensorTypeAndShape().ElementCount != pointCount)
            {
                inputs[1]?.Dispose();
                inputs[2]?.Dispose();

                inputs[1] = OrtValue.CreateAllocatedTensorValue(
                    OrtAllocator.DefaultInstance, TensorElementType.Float,
                    new long[] { 1, 1, pointCount, 2 });
                inputs[2] = OrtValue.CreateAllocatedTensorValue(
                    OrtAllocator.DefaultInstance, TensorElementType.Float,
                    new long[] { 1, 1, pointCount });
            }
        }

        private static void SetCoordAndLabels(
            Span<float> coords, Span<float> labels,
            ReadOnlyCollection<Point> points,
            int width, int height)
        {
            int length = points.Count;
            for (int i = 0; i < length; i++)
            {
                var p = points[i];
                coords[i * 2] = p.point.x * width;
                coords[i * 2 + 1] = p.point.y * height;
                labels[i] = p.label;
            }
        }
    }
}
