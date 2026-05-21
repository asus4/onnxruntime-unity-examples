using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime.Examples.Supertonic;
using Microsoft.ML.OnnxRuntime.Unity;
using UnityEngine;

namespace Microsoft.ML.OnnxRuntime.Examples
{
    /// <summary>
    /// Supertonic Text-To-Speech
    /// </summary>
    public sealed class SupertonicTTS : IDisposable
    {
        // Downloads the model assets from HuggingFace
        const string ModelBaseUrl = "https://huggingface.co/Supertone/supertonic-3/resolve/main/";

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

        /// <summary>
        /// Initializes the TTS pipeline.
        /// On first call this downloads ~400MB assets in parallel.
        /// </summary>
        public static async Awaitable<SupertonicTTS> InitAsync(
            IProgress<float> progress,
            CancellationToken cancellationToken)
        {
            // Build manifest: 4 ONNX + 2 JSON config + 10 voice styles = 16 files.
            // Order matters: it determines how the resulting path array maps
            // back into TtsAssets / the voice style dictionary.
            var manifest = new List<string>
            {
                "onnx/duration_predictor.onnx",
                "onnx/text_encoder.onnx",
                "onnx/vector_estimator.onnx",
                "onnx/vocoder.onnx",
                "onnx/tts.json",
                "onnx/unicode_indexer.json",
            };
            manifest.AddRange(VoiceIds.Select(id => $"voice_styles/{id}.json"));

            var files = manifest.Select(p => new RemoteFile(ModelBaseUrl + p)).ToArray();

            // Get file sizes for progress.
            long[] sizes = await Task.WhenAll(files.Select(f => f.GetSize(cancellationToken).AsTask()));
            long totalBytes = sizes.Sum();

            // Pre-fill progress for cached files (which won't emit OnDownloadProgress).
            var perFileProgress = new float[files.Length];
            for (int i = 0; i < files.Length; i++)
            {
                if (files[i].HasCache) perFileProgress[i] = 1.0f;
            }
            ReportWeighted(progress, perFileProgress, sizes, totalBytes);

            // Subscribe to per-file progress and aggregate into a single 0..1 value.
            for (int i = 0; i < files.Length; i++)
            {
                int idx = i;
                files[i].OnDownloadProgress += p =>
                {
                    perFileProgress[idx] = p;
                    ReportWeighted(progress, perFileProgress, sizes, totalBytes);
                };
            }

            // Download or resolve from cache all 16 files in parallel.
            string[] paths = await Task.WhenAll(files.Select(f => f.EnsureLocal(cancellationToken).AsTask()));

            await Awaitable.BackgroundThreadAsync();
            cancellationToken.ThrowIfCancellationRequested();

            var assets = new TtsAssets
            {
                DurationPredictorOnnxPath = paths[0],
                TextEncoderOnnxPath = paths[1],
                VectorEstimatorOnnxPath = paths[2],
                VocoderOnnxPath = paths[3],
                TtsConfigJsonPath = paths[4],
                UnicodeIndexerJsonPath = paths[5],
            };

            TextToSpeech tts = null;
            var styles = new Dictionary<string, Style>(VoiceIds.Length);
            try
            {
                tts = Helper.LoadTextToSpeech(assets);
                for (int i = 0; i < VoiceIds.Length; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    styles[VoiceIds[i]] = Helper.LoadVoiceStyle(paths[6 + i]);
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

        static void ReportWeighted(IProgress<float> progress, float[] perFile, long[] sizes, long totalBytes)
        {
            if (progress == null) return;
            float weighted = 0;
            for (int j = 0; j < perFile.Length; j++)
            {
                weighted += perFile[j] * sizes[j];
            }
            progress.Report(weighted / totalBytes);
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
