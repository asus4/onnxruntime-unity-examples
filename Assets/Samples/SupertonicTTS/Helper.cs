/// MIT License
/// Copyright (c) 2025 Supertone Inc.
///
/// Derived from https://github.com/supertone-inc/supertonic/blob/main/csharp/Helper.cs
/// Adapted for Unity and simplified to the single-text / single-voice path
/// (batched inference paths from the upstream reference were removed).

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Microsoft.ML.OnnxRuntime.Examples.Supertonic
{
    // Available languages for multilingual TTS
    public static class Languages
    {
        public static readonly string[] Available = { "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi", "na" };
    }

    // ============================================================================
    // Configuration
    // ============================================================================

    public class Config
    {
        public int SampleRate;
        public int BaseChunkSize;
        public int ChunkCompressFactor;
        public int LatentDim;
    }

    // ============================================================================
    // Style class
    // ============================================================================

    public class Style
    {
        public readonly float[] Ttl;
        public readonly int[] TtlShape;
        public readonly float[] Dp;
        public readonly int[] DpShape;

        public Style(float[] ttl, int[] ttlShape, float[] dp, int[] dpShape)
        {
            Ttl = ttl;
            TtlShape = ttlShape;
            Dp = dp;
            DpShape = dpShape;
        }
    }

    // ============================================================================
    // Unicode text processor
    // ============================================================================

    public class UnicodeProcessor
    {
        private readonly Dictionary<int, long> _indexer;

        public UnicodeProcessor(string unicodeIndexerPath)
        {
            var json = File.ReadAllText(unicodeIndexerPath);
            var indexerArray = JsonConvert.DeserializeObject<long[]>(json)
                ?? throw new Exception("Failed to load indexer");
            _indexer = new Dictionary<int, long>(indexerArray.Length);
            for (int i = 0; i < indexerArray.Length; i++)
            {
                _indexer[i] = indexerArray[i];
            }
        }

        private static string RemoveEmojis(string text)
        {
            var result = new StringBuilder(text.Length);
            for (int i = 0; i < text.Length; i++)
            {
                int codePoint;
                if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
                {
                    // Get the full code point from surrogate pair
                    codePoint = char.ConvertToUtf32(text[i], text[i + 1]);
                    i++; // Skip the low surrogate
                }
                else
                {
                    codePoint = text[i];
                }

                // Check if code point is in emoji ranges
                bool isEmoji = (codePoint >= 0x1F600 && codePoint <= 0x1F64F) ||
                               (codePoint >= 0x1F300 && codePoint <= 0x1F5FF) ||
                               (codePoint >= 0x1F680 && codePoint <= 0x1F6FF) ||
                               (codePoint >= 0x1F700 && codePoint <= 0x1F77F) ||
                               (codePoint >= 0x1F780 && codePoint <= 0x1F7FF) ||
                               (codePoint >= 0x1F800 && codePoint <= 0x1F8FF) ||
                               (codePoint >= 0x1F900 && codePoint <= 0x1F9FF) ||
                               (codePoint >= 0x1FA00 && codePoint <= 0x1FA6F) ||
                               (codePoint >= 0x1FA70 && codePoint <= 0x1FAFF) ||
                               (codePoint >= 0x2600 && codePoint <= 0x26FF) ||
                               (codePoint >= 0x2700 && codePoint <= 0x27BF) ||
                               (codePoint >= 0x1F1E6 && codePoint <= 0x1F1FF);

                if (!isEmoji)
                {
                    if (codePoint > 0xFFFF)
                    {
                        // Add back as surrogate pair
                        result.Append(char.ConvertFromUtf32(codePoint));
                    }
                    else
                    {
                        result.Append((char)codePoint);
                    }
                }
            }
            return result.ToString();
        }

        // Replace various dashes / quotes / brackets / arrows with ASCII equivalents,
        // and expand known abbreviations.
        private static readonly Dictionary<string, string> SymbolReplacements = new()
        {
            {"–", "-"},      // en dash
            {"‑", "-"},      // non-breaking hyphen
            {"—", "-"},      // em dash
            {"_", " "},      // underscore
            {"“", "\""},     // left double quote
            {"”", "\""},     // right double quote
            {"‘", "'"},      // left single quote
            {"’", "'"},      // right single quote
            {"´", "'"},      // acute accent
            {"`", "'"},      // grave accent
            {"[", " "},      // left bracket
            {"]", " "},      // right bracket
            {"|", " "},      // vertical bar
            {"/", " "},      // slash
            {"#", " "},      // hash
            {"→", " "},      // right arrow
            {"←", " "},      // left arrow
            {"@", " at "},
            {"e.g.,", "for example, "},
            {"i.e.,", "that is, "},
        };

        private static string PreprocessText(string text, string lang)
        {
            // Validate language
            if (!Languages.Available.Contains(lang))
            {
                throw new ArgumentException($"Invalid language: {lang}. Available: {string.Join(", ", Languages.Available)}");
            }

            // TODO: Need advanced normalizer for better performance
            text = text.Normalize(NormalizationForm.FormKD);

            // Remove emojis (wide Unicode range)
            // C# doesn't support \u{...} syntax in regex, so we use character filtering instead
            text = RemoveEmojis(text);

            // Replace dashes, quotes, brackets, arrows and known expressions
            foreach (var kvp in SymbolReplacements)
            {
                text = text.Replace(kvp.Key, kvp.Value);
            }

            // Remove special symbols
            text = Regex.Replace(text, @"[♥☆♡©\\]", "");

            // Fix spacing around punctuation
            text = Regex.Replace(text, @" ,", ",");
            text = Regex.Replace(text, @" \.", ".");
            text = Regex.Replace(text, @" !", "!");
            text = Regex.Replace(text, @" \?", "?");
            text = Regex.Replace(text, @" ;", ";");
            text = Regex.Replace(text, @" :", ":");
            text = Regex.Replace(text, @" '", "'");

            // Remove duplicate quotes
            while (text.Contains("\"\"")) text = text.Replace("\"\"", "\"");
            while (text.Contains("''")) text = text.Replace("''", "'");
            while (text.Contains("``")) text = text.Replace("``", "`");

            // Remove extra spaces
            text = Regex.Replace(text, @"\s+", " ").Trim();

            // If text doesn't end with punctuation, quotes, or closing brackets, add a period
            if (!Regex.IsMatch(text, @"[.!?;:,'""“”‘’)\]}…。」』】〉》›»]$"))
            {
                text += ".";
            }

            // Wrap text with language tags
            return $"<{lang}>{text}</{lang}>";
        }

        public (long[] textIds, int length) Call(string text, string lang)
        {
            string processed = PreprocessText(text, lang);
            var textIds = new long[processed.Length];
            for (int i = 0; i < processed.Length; i++)
            {
                if (_indexer.TryGetValue(processed[i], out long val))
                {
                    textIds[i] = val;
                }
            }
            return (textIds, processed.Length);
        }
    }

    // ============================================================================
    // TextToSpeech class
    // ============================================================================

    public class TextToSpeech : IDisposable
    {
        private readonly Config _cfg;
        private readonly UnicodeProcessor _textProcessor;
        private readonly InferenceSession _dpOrt;
        private readonly InferenceSession _textEncOrt;
        private readonly InferenceSession _vectorEstOrt;
        private readonly InferenceSession _vocoderOrt;
        private bool _disposed;

        public int SampleRate => _cfg.SampleRate;

        public TextToSpeech(
            Config cfg,
            UnicodeProcessor textProcessor,
            InferenceSession dpOrt,
            InferenceSession textEncOrt,
            InferenceSession vectorEstOrt,
            InferenceSession vocoderOrt)
        {
            _cfg = cfg;
            _textProcessor = textProcessor;
            _dpOrt = dpOrt;
            _textEncOrt = textEncOrt;
            _vectorEstOrt = vectorEstOrt;
            _vocoderOrt = vocoderOrt;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _dpOrt?.Dispose();
            _textEncOrt?.Dispose();
            _vectorEstOrt?.Dispose();
            _vocoderOrt?.Dispose();
            _disposed = true;
        }

        // Top-level entry: splits long text into chunks the model can handle,
        // joins the per-chunk PCM with a short silence between each.
        public float[] Call(string text, string lang, Style style, int totalStep, float speed = 1.05f, float silenceDuration = 0.3f)
        {
            if (style.TtlShape[0] != 1)
            {
                throw new ArgumentException("Single speaker text to speech only supports single style");
            }

            int maxLen = (lang == "ko" || lang == "ja") ? 120 : 300;
            var chunks = Helper.ChunkText(text, maxLen);

            var wavCat = new List<float>();
            int silenceLen = (int)(silenceDuration * SampleRate);
            for (int i = 0; i < chunks.Count; i++)
            {
                if (i > 0) wavCat.AddRange(new float[silenceLen]);
                wavCat.AddRange(Infer(chunks[i], lang, style, totalStep, speed));
            }
            return wavCat.ToArray();
        }

        // Runs the 4-stage pipeline (duration predictor -> text encoder ->
        // iterative vector estimator -> vocoder) on a single chunk.
        private float[] Infer(string text, string lang, Style style, int totalStep, float speed)
        {
            // Process text
            var (textIds, textLen) = _textProcessor.Call(text, lang);

            var textMask = new float[textLen];
            Array.Fill(textMask, 1.0f);

            var textIdsTensor = new DenseTensor<long>(textIds, new[] { 1, textLen });
            var textMaskTensor = new DenseTensor<float>(textMask, new[] { 1, 1, textLen });
            var styleTtlTensor = new DenseTensor<float>(style.Ttl, style.TtlShape);
            var styleDpTensor = new DenseTensor<float>(style.Dp, style.DpShape);

            // Run duration predictor (apply speed factor to the output)
            float duration;
            using (var dpOutputs = _dpOrt.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor("text_ids", textIdsTensor),
                NamedOnnxValue.CreateFromTensor("style_dp", styleDpTensor),
                NamedOnnxValue.CreateFromTensor("text_mask", textMaskTensor),
            }))
            {
                duration = dpOutputs.First(o => o.Name == "duration").AsTensor<float>().ToArray()[0] / speed;
            }

            // Run text encoder
            using var textEncOutputs = _textEncOrt.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor("text_ids", textIdsTensor),
                NamedOnnxValue.CreateFromTensor("style_ttl", styleTtlTensor),
                NamedOnnxValue.CreateFromTensor("text_mask", textMaskTensor),
            });
            var textEmbTensor = textEncOutputs.First(o => o.Name == "text_emb").AsTensor<float>();

            // Sample noisy latent (and build the matching latent mask)
            int chunkSize = _cfg.BaseChunkSize * _cfg.ChunkCompressFactor;
            int wavLen = (int)(duration * SampleRate);
            int latentLen = (wavLen + chunkSize - 1) / chunkSize;
            int latentDim = _cfg.LatentDim * _cfg.ChunkCompressFactor;

            // Single-batch latent mask is all ones (no padding to compete with),
            // so we can skip the per-sample noise masking the upstream code does.
            var latentMask = new float[latentLen];
            Array.Fill(latentMask, 1.0f);
            var latentMaskTensor = new DenseTensor<float>(latentMask, new[] { 1, 1, latentLen });

            var xt = SampleStandardNormal(latentDim * latentLen);
            int[] latentShape = { 1, latentDim, latentLen };

            // Iterative denoising
            float[] totalStepArr = { totalStep };
            for (int step = 0; step < totalStep; step++)
            {
                float[] currentStepArr = { step };
                using var vectorEstOutputs = _vectorEstOrt.Run(new[]
                {
                    NamedOnnxValue.CreateFromTensor("noisy_latent", new DenseTensor<float>(xt, latentShape)),
                    NamedOnnxValue.CreateFromTensor("text_emb", textEmbTensor),
                    NamedOnnxValue.CreateFromTensor("style_ttl", styleTtlTensor),
                    NamedOnnxValue.CreateFromTensor("text_mask", textMaskTensor),
                    NamedOnnxValue.CreateFromTensor("latent_mask", latentMaskTensor),
                    NamedOnnxValue.CreateFromTensor("total_step", new DenseTensor<float>(totalStepArr, new[] { 1 })),
                    NamedOnnxValue.CreateFromTensor("current_step", new DenseTensor<float>(currentStepArr, new[] { 1 })),
                });
                // Update xt with the denoised latent for the next step
                var denoised = vectorEstOutputs.First(o => o.Name == "denoised_latent").AsTensor<float>();
                for (int i = 0; i < xt.Length; i++)
                {
                    xt[i] = denoised.GetValue(i);
                }
            }

            // Run vocoder
            using var vocoderOutputs = _vocoderOrt.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor("latent", new DenseTensor<float>(xt, latentShape)),
            });
            return vocoderOutputs.First(o => o.Name == "wav_tts").AsTensor<float>().ToArray();
        }

        // Box-Muller transform for standard normal distribution N(0, 1).
        private static float[] SampleStandardNormal(int count)
        {
            var random = new System.Random();
            var buf = new float[count];
            for (int i = 0; i < count; i++)
            {
                double u1 = 1.0 - random.NextDouble();
                double u2 = 1.0 - random.NextDouble();
                buf[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
            }
            return buf;
        }
    }

    // ============================================================================
    // Helper class with utility functions
    // ============================================================================

    public static class Helper
    {
        // ============================================================================
        // TextToSpeech loading (config + 4 ONNX models + unicode indexer)
        // ============================================================================

        public static TextToSpeech LoadTextToSpeech(TtsAssets assets)
        {
            var cfg = LoadConfig(assets.TtsConfigJsonPath);
            var textProcessor = new UnicodeProcessor(assets.UnicodeIndexerJsonPath);
            var opts = new SessionOptions();

            InferenceSession dp = null, textEnc = null, vectorEst = null, vocoder = null;
            try
            {
                // ORT loads these via the native path, allowing mmap so the model
                // bytes (256MB for vector_estimator) never enter the managed heap.
                dp = new InferenceSession(assets.DurationPredictorOnnxPath, opts);
                textEnc = new InferenceSession(assets.TextEncoderOnnxPath, opts);
                vectorEst = new InferenceSession(assets.VectorEstimatorOnnxPath, opts);
                vocoder = new InferenceSession(assets.VocoderOnnxPath, opts);
            }
            catch
            {
                // Make sure we don't leak any sessions that already loaded
                dp?.Dispose();
                textEnc?.Dispose();
                vectorEst?.Dispose();
                vocoder?.Dispose();
                throw;
            }

            return new TextToSpeech(cfg, textProcessor, dp, textEnc, vectorEst, vocoder);
        }

        private static Config LoadConfig(string cfgPath)
        {
            var root = JObject.Parse(File.ReadAllText(cfgPath));
            return new Config
            {
                SampleRate = (int)root["ae"]["sample_rate"],
                BaseChunkSize = (int)root["ae"]["base_chunk_size"],
                ChunkCompressFactor = (int)root["ttl"]["chunk_compress_factor"],
                LatentDim = (int)root["ttl"]["latent_dim"],
            };
        }

        // ============================================================================
        // Voice style loading
        // ============================================================================

        public static Style LoadVoiceStyle(string voiceStylePath)
        {
            var root = JObject.Parse(File.ReadAllText(voiceStylePath));
            var ttlDims = root["style_ttl"]["dims"].ToObject<int[]>();
            var dpDims = root["style_dp"]["dims"].ToObject<int[]>();

            var ttl = new float[ttlDims[0] * ttlDims[1] * ttlDims[2]];
            FlattenInto(root["style_ttl"]["data"], ttl, 0);

            var dp = new float[dpDims[0] * dpDims[1] * dpDims[2]];
            FlattenInto(root["style_dp"]["data"], dp, 0);

            return new Style(ttl, ttlDims, dp, dpDims);
        }

        // Flattens an arbitrarily nested JSON array of numbers into `dest` starting at `offset`.
        private static int FlattenInto(JToken token, float[] dest, int offset)
        {
            if (token is JArray arr)
            {
                foreach (var child in arr)
                {
                    offset = FlattenInto(child, dest, offset);
                }
                return offset;
            }
            dest[offset] = (float)token;
            return offset + 1;
        }

        // ============================================================================
        // Chunk text
        // ============================================================================

        // Split by paragraph (two or more newlines)
        private static readonly Regex ParagraphRegex = new(@"\n\s*\n+");

        // Split by sentence boundaries, excluding common abbreviations
        private static readonly Regex SentenceRegex = new(@"(?<!Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.|Sr\.|Jr\.|Ph\.D\.|etc\.|e\.g\.|i\.e\.|vs\.|Inc\.|Ltd\.|Co\.|Corp\.|St\.|Ave\.|Blvd\.)(?<!\b[A-Z]\.)(?<=[.!?])\s+");

        public static List<string> ChunkText(string text, int maxLen = 300)
        {
            var chunks = new List<string>();

            var paragraphs = ParagraphRegex.Split(text.Trim())
                .Select(p => p.Trim())
                .Where(p => !string.IsNullOrEmpty(p));

            foreach (var paragraph in paragraphs)
            {
                string currentChunk = "";
                foreach (var sentence in SentenceRegex.Split(paragraph))
                {
                    if (string.IsNullOrEmpty(sentence)) continue;

                    if (currentChunk.Length + sentence.Length + 1 <= maxLen)
                    {
                        if (currentChunk.Length > 0) currentChunk += " ";
                        currentChunk += sentence;
                    }
                    else
                    {
                        if (currentChunk.Length > 0) chunks.Add(currentChunk.Trim());
                        currentChunk = sentence;
                    }
                }
                if (currentChunk.Length > 0) chunks.Add(currentChunk.Trim());
            }

            // If no chunks were created, return the original text
            if (chunks.Count == 0) chunks.Add(text.Trim());
            return chunks;
        }
    }
}
