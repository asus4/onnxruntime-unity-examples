using System;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Examples;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class SupertonicTTSSample : MonoBehaviour
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

    const string Lang = "en";
    const string DefaultText = "This morning, I took a walk in the park, and the sound of the birds and the breeze was so pleasant that I stopped for a long time just to listen.";

    [SerializeField] Options[] platformOptions = { };

    [Header("UI References")]
    [SerializeField] TMP_InputField input;
    [SerializeField] TMP_Dropdown voiceDropdown;
    [SerializeField] Button generateButton;
    [SerializeField] TMP_Text statusLabel;
    [SerializeField] AudioSource audioSource;

    SupertonicTTS tts;

    async Awaitable Start()
    {
        SetStatus("Loading model...");
        SetButtonEnabled(false);
        if (string.IsNullOrEmpty(input.text))
        {
            input.text = DefaultText;
        }

        voiceDropdown.ClearOptions();
        voiceDropdown.AddOptions(SupertonicTTS.VoiceIds.ToList());

        var platform = Application.platform;
        var options = platformOptions.FirstOrDefault(o => o.platforms.Contains(platform));
        if (options == null || !options.TryGetModelPath(out var modelPath))
        {
            SetStatus("Model not found. Download it via:\n  hf download Supertone/supertonic-3 --local-dir <modelDir>");
            return;
        }

        try
        {
            tts = await SupertonicTTS.InitAsync(modelPath, destroyCancellationToken);
        }
        catch (OperationCanceledException)
        {
            return;
        }
        catch (Exception ex)
        {
            SetStatus($"Failed to load: {ex.Message}");
            Debug.LogException(ex);
            return;
        }

        SetStatus("Ready.");
        generateButton.onClick.AddListener(OnGenerateClick);
        SetButtonEnabled(true);
    }

    void OnDestroy()
    {
        if (generateButton != null)
        {
            generateButton.onClick.RemoveListener(OnGenerateClick);
        }
        tts?.Dispose();
    }

    async void OnGenerateClick()
    {
        string text = input.text;
        if (string.IsNullOrWhiteSpace(text)) return;

        string voiceId = SupertonicTTS.VoiceIds[voiceDropdown.value];

        SetButtonEnabled(false);
        SetStatus($"Generating ({voiceId})...");

        float[] pcm;
        try
        {
            pcm = await tts.GenerateAsync(text, voiceId, Lang, destroyCancellationToken);
        }
        catch (OperationCanceledException)
        {
            return;
        }
        catch (Exception ex)
        {
            SetStatus($"Generation failed: {ex.Message}");
            Debug.LogException(ex);
            SetButtonEnabled(true);
            return;
        }

        var clip = AudioClip.Create($"supertonic_{voiceId}", pcm.Length, 1, tts.SampleRate, false);
        clip.SetData(pcm, 0);
        audioSource.clip = clip;
        audioSource.Play();

        float seconds = (float)pcm.Length / tts.SampleRate;
        SetStatus($"Playing {seconds:F1}s ({voiceId}).");
        SetButtonEnabled(true);
    }

    void SetStatus(string msg)
    {
        if (statusLabel != null) statusLabel.SetText(msg);
    }

    void SetButtonEnabled(bool enabled)
    {
        if (generateButton != null) generateButton.interactable = enabled;
    }
}
