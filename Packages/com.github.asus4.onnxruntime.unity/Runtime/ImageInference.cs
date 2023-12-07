#nullable enable

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine;

namespace Microsoft.ML.OnnxRuntime.Unity
{
    public class ImageInference<TInput> : IDisposable
    {
        protected readonly InferenceSession session;
        protected readonly Dictionary<string, OrtValue> inputs = new();

        protected readonly string inputImageKey;
        protected readonly int channels;
        protected readonly int height;
        protected readonly int width;

        /// <summary>
        /// Create an inference that has Image input
        /// </summary>
        /// <param name="model">byte array of the Ort model</param>
        public ImageInference(byte[] model)
        {
            try
            {
                session = new InferenceSession(model);
            }
            catch (Exception e)
            {
                session?.Dispose();
                throw e;
            }

            DisplayIOInfo(session);

            // Create input NamedOnnxValue
            foreach (var kv in session.InputMetadata)
            {
                string key = kv.Key;
                NodeMetadata meta = kv.Value;
                if (meta.IsTensor)
                {
                    long[] shape = meta.Dimensions.Select(x => (long)x).ToArray();
                    if (IsSupportedImage(shape))
                    {
                        inputImageKey = key;
                        channels = (int)shape[1];
                        height = (int)shape[2];
                        width = (int)shape[3];
                    }
                    var ortValue = OrtValue.CreateAllocatedTensorValue(
                        OrtAllocator.DefaultInstance, meta.ElementDataType, shape);
                    inputs.Add(key, ortValue);
                }
                else
                {
                    throw new ArgumentException("Only tensor input is supported");
                }
            }

            if (inputImageKey == null)
            {
                throw new ArgumentException("No supported image input found");
            }
        }

        public void Dispose()
        {
            session?.Dispose();
        }

        public virtual void Run(Texture texture)
        {
            // Debug.Log("TODO: Run");
        }

        private static void DisplayIOInfo(InferenceSession session)
        {
            foreach (var kv in session.InputMetadata)
            {
                string key = kv.Key;
                NodeMetadata meta = kv.Value;
                Debug.Log($"Input name: {key} shape: {string.Join(",", meta.Dimensions)}, type: {meta.ElementType} isTensor: {meta.IsTensor}");
            }
            foreach (var meta in session.OutputMetadata)
            {
                string key = meta.Key;
                NodeMetadata metaValue = meta.Value;
                Debug.Log($"Output name: {key} shape: {string.Join(",", metaValue.Dimensions)}, type: {metaValue.ElementType} isTensor: {metaValue.IsTensor}");
            }
        }

        private static bool IsSupportedImage(long[] shape)
        {
            long channels = shape.Length switch
            {
                4 => shape[0] == 1 ? shape[1] : 0,
                3 => shape[0],
                _ => 0
            };
            // return channels == 1 || channels == 3 || channels == 4;
            // RGB is supported for now
            return channels == 3;
        }
    }
}
