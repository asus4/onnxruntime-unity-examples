/// Unity wrapper around the Supertonic TTS pipeline.
/// See Helper.cs for the upstream MIT license (Copyright (c) 2025 Supertone Inc.).

using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using Microsoft.ML.OnnxRuntime.Examples.Supertonic;
using UnityEngine;

namespace Microsoft.ML.OnnxRuntime.Examples
{
    /// <summary>
    /// Supertonic Text-To-Speech inference for Unity.
    ///
    /// Expects a directory layout matching https://huggingface.co/Supertone/supertonic-3 :
    ///   <modelDir>/onnx/{duration_predictor,text_encoder,vector_estimator,vocoder}.onnx
    ///   <modelDir>/onnx/{tts.json,unicode_indexer.json}
    ///   <modelDir>/voice_styles/{M1..M5,F1..F5}.json
    /// </summary>
    public sealed class SupertonicTTS : IDisposable
    {

        public enum PathType
        {
            Absolute,
            Data,
            Persistent,
            TemporaryCache,
            StreamingAssets,
        }

        [Serializable]
        public class Options
        {
            public RuntimePlatform[] platforms = { RuntimePlatform.OSXEditor, RuntimePlatform.OSXPlayer };
            public PathType pathType = PathType.Absolute;
            public string modelPath = string.Empty;

            public virtual bool TryGetModelPath(out string path)
            {
                path = modelPath;
                if (string.IsNullOrWhiteSpace(path))
                {
                    return false;
                }

                if (!Path.IsPathRooted(path))
                {
                    path = pathType switch
                    {
                        PathType.Absolute => path,
                        PathType.Data => Path.Combine(Application.dataPath, path),
                        PathType.Persistent => Path.Combine(Application.persistentDataPath, path),
                        PathType.TemporaryCache => Path.Combine(Application.temporaryCachePath, path),
                        PathType.StreamingAssets => Path.Combine(Application.streamingAssetsPath, path),
                        _ => throw new NotImplementedException($"PathType {pathType} is not implemented"),
                    };
                }

                return Directory.Exists(path);
            }
        }

        public static readonly string[] VoiceIds = { "M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5" };

        readonly TextToSpeech tts;
        readonly Dictionary<string, Style> styles;
        bool disposed = false;

        public int SampleRate => tts.SampleRate;
        public int TotalStep { get; set; } = 8;
        public float Speed { get; set; } = 1.05f;

        SupertonicTTS(TextToSpeech tts, Dictionary<string, Style> styles)
        {
            this.tts = tts;
            this.styles = styles;
        }

        ~SupertonicTTS() => Dispose(false);

        public void Dispose() => Dispose(true);

        void Dispose(bool _)
        {
            if (disposed) return;
            tts?.Dispose();
            disposed = true;
        }

        public static async Awaitable<SupertonicTTS> InitAsync(string modelDir, CancellationToken cancellationToken)
        {
            if (!Directory.Exists(modelDir))
            {
                string msg = $"Model directory not found at {modelDir}, download it from HuggingFace https://huggingface.co/Supertone/supertonic-3";
                Debug.LogError(msg);
                throw new DirectoryNotFoundException(msg);
            }

            string onnxDir = Path.Combine(modelDir, "onnx");
            string voiceDir = Path.Combine(modelDir, "voice_styles");
            if (!Directory.Exists(onnxDir) || !Directory.Exists(voiceDir))
            {
                string msg = $"Expected '{onnxDir}' and '{voiceDir}' under model directory.";
                Debug.LogError(msg);
                throw new DirectoryNotFoundException(msg);
            }

            await Awaitable.BackgroundThreadAsync();
            cancellationToken.ThrowIfCancellationRequested();

            TextToSpeech tts = null;
            var styles = new Dictionary<string, Style>(VoiceIds.Length);
            try
            {
                tts = Helper.LoadTextToSpeech(onnxDir);
                foreach (var id in VoiceIds)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    styles[id] = Helper.LoadVoiceStyle(Path.Combine(voiceDir, $"{id}.json"));
                }
            }
            catch (Exception)
            {
                tts?.Dispose();
                throw;
            }

            await Awaitable.MainThreadAsync();
            cancellationToken.ThrowIfCancellationRequested();

            return new SupertonicTTS(tts, styles);
        }

        /// <summary>
        /// Generates 1-channel float PCM audio in [-1, 1]. Runs on a background thread.
        /// </summary>
        public async Awaitable<float[]> GenerateAsync(string text, string voiceId, string lang, CancellationToken cancellationToken)
        {
            if (!styles.TryGetValue(voiceId, out var style))
            {
                throw new ArgumentException($"Unknown voiceId: {voiceId}");
            }

            await Awaitable.BackgroundThreadAsync();
            cancellationToken.ThrowIfCancellationRequested();

            var pcm = tts.Call(text, lang, style, TotalStep, Speed);

            await Awaitable.MainThreadAsync();
            cancellationToken.ThrowIfCancellationRequested();

            return pcm;
        }
    }
}
