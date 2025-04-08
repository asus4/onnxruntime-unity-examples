/// Copyright (c) Microsoft Corporation. All rights reserved.
/// Licensed under the MIT License.
/// 
/// Modified by @asus4

using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntimeGenAI;
using Microsoft.ML.OnnxRuntime.Unity;
using UnityEngine;


namespace Microsoft.ML.OnnxRuntime.Examples
{
    /// <summary>
    /// PHI4 Multi Model Inference
    /// 
    /// Ported from C# example in GenAI repo
    /// https://github.com/microsoft/onnxruntime-genai/blob/6f70febdde54485726eabebc9b9b17c8806820f0/examples/csharp/HelloPhi3V/Program.cs
    /// </summary>
    public class Phi4 : IDisposable
    {
        readonly Config config;
        readonly Model model;
        readonly Tokenizer tokenizer;

        // TODO: Test with all providers
        // https://github.com/microsoft/onnxruntime-genai/blob/6f70febdde54485726eabebc9b9b17c8806820f0/build.py#L472-L474
        static readonly string[] supportedProviders = { "cuda", "rocm", "dml" };

        bool disposed = false;

        public Phi4(string modelPath, string provider)
        {
            // Set ORT_LIB_PATH environment variable to use GenAI
            OrtUnityEnv.InitializeOrtLibPath();

            provider = provider.ToLowerInvariant();

            if (string.IsNullOrWhiteSpace(provider))
            {
                model = new Model(modelPath);
            }
            // Check if provider is valid
            else if (supportedProviders.Contains(provider))
            {
                Debug.Log($"Configuring {provider} provider");
                // TODO: Test on Windows / Linux
                config = new Config(modelPath);
                config.ClearProviders();
                config.AppendProvider(provider);
                if (provider.Equals("cuda"))
                {
                    config.SetProviderOption(provider, "enable_cuda_graph", "0");
                }
            }
            else
            {
                string msg = $"Provider: `{provider}` is not supported. Use one of them: {string.Join(", ", supportedProviders)}. Falling back to CPU.";
                Debug.LogWarning(msg);
                model = new Model(modelPath);
            }
            tokenizer = new Tokenizer(model);
        }

        ~Phi4()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
        }

        void Dispose(bool disposing)
        {
            if (disposed)
            {
                return;
            }
            if (disposing)
            {
                tokenizer?.Dispose();
                model?.Dispose();
                config?.Dispose();
            }
            disposed = true;
        }

        public static async Awaitable<Phi4> InitAsync(string modelPath, string providerName, CancellationToken cancellationToken)
        {
            if (Debug.isDebugBuild)
            {
                // Verbose GenAI Log
                Environment.SetEnvironmentVariable("ORTGENAI_LOG_ORT_LIB", "1");
            }

            if (!Directory.Exists(modelPath))
            {
                string msg = $"Model not found at {modelPath}, download it from HuggingFace https://huggingface.co/microsoft";
                Debug.LogError(msg);
                throw new DirectoryNotFoundException(msg);
            }

            // Run in BG thread to avoid blocking the Unity thread
            await Awaitable.BackgroundThreadAsync();
            cancellationToken.ThrowIfCancellationRequested();

            Phi4 instance = null;
            try
            {
                instance = new Phi4(modelPath, providerName);
            }
            catch (Exception ex)
            {
                Debug.LogError($"Failed to create Phi4MultiModal: {ex.Message}");
                instance?.Dispose();
                throw ex;
            }

            await Awaitable.MainThreadAsync();
            cancellationToken.ThrowIfCancellationRequested();

            return instance;
        }

        public async IAsyncEnumerable<string> GenerateStream(
            string prompt,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            await Awaitable.BackgroundThreadAsync();
            cancellationToken.ThrowIfCancellationRequested();

            using var sequences = tokenizer.Encode($"<|user|>{prompt}<|end|><|assistant|>");
            using var tokenizerStream = tokenizer.CreateStream();

            const int minLength = 50;
            const int maxLength = 500;

            using var generatorParams = new GeneratorParams(model);
            generatorParams.SetSearchOption("min_length", minLength);
            generatorParams.SetSearchOption("max_length", maxLength);
            using var generator = new Generator(model, generatorParams);

            generator.AppendTokenSequences(sequences);

            // Return results in the Unity main thread
            await Awaitable.MainThreadAsync();

            var outputQueue = new ConcurrentQueue<string>();
            var generateTask = Task.Run(() =>
            {
                while (!generator.IsDone())
                {
                    if (cancellationToken.IsCancellationRequested) { return; }
                    generator.GenerateNextToken();
                    if (cancellationToken.IsCancellationRequested) { return; }
                    outputQueue.Enqueue(tokenizerStream.Decode(generator.GetSequence(0)[^1]));
                }
            }, cancellationToken);

            while (!cancellationToken.IsCancellationRequested && !generateTask.IsCompleted)
            {
                await Awaitable.NextFrameAsync(cancellationToken);
                if (outputQueue.TryDequeue(out var response))
                {
                    yield return response;
                }
            }
        }
    }
}
