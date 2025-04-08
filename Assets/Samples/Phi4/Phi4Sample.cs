using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using Microsoft.ML.OnnxRuntime.Examples;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class Phi4Sample : MonoBehaviour
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
        public string providerName = string.Empty;

        public virtual bool TryGetModelPath(out string path)
        {
            // Check Path is valid
            path = modelPath;
            if (string.IsNullOrWhiteSpace(path))
            {
                return false;
            }

            if (!Path.IsPathRooted(path))
            {
                // Assuming the directory
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

            if (!Directory.Exists(path))
            {
                return false;
            }
            return true;
        }
    }

    [SerializeField]
    Options[] platformOptions = { };

    [SerializeField]
    TMP_Text label;

    [SerializeField]
    Button button;

    [SerializeField]
    string prompt = "What is the capital of France?";

    readonly StringBuilder sb = new();
    Phi4 inference;

    async Awaitable Start()
    {
        ToggleButton(false);

        // Find options for this platform
        RuntimePlatform platform = Application.platform;
        var options = platformOptions.FirstOrDefault(o => o.platforms.Contains(platform));
        if (options == null)
        {
            Debug.LogError($"Platform {platform} not supported.");
            return;
        }
        if (!options.TryGetModelPath(out var modelPath))
        {
            string msg = $"Model not found at {modelPath}, download it from HuggingFace https://huggingface.co/microsoft";
            Debug.LogError(msg);
            return;
        }

        // Initialize the model
        UpdateLabel("Now loading LLM model. wait a bit...");
        try
        {
            string providerName = options.providerName;
            inference = await Phi4.InitAsync(modelPath, providerName, destroyCancellationToken);
        }
        catch (Exception ex)
        {
            UpdateLabel($"Failed to initialize: {ex.Message}");
            Debug.LogException(ex);
            return;
        }

        UpdateLabel("Model initialized!");
        button.onClick.AddListener(OnAskButtonClick);
        ToggleButton(true);
    }

    void OnDestroy()
    {
        button.onClick.RemoveListener(OnAskButtonClick);
        inference?.Dispose();
        Debug.Log($"Phi4MultiModal disposed");
    }

    async void OnAskButtonClick()
    {
        await Generate(prompt, destroyCancellationToken);
    }

    async Awaitable Generate(string prompt, CancellationToken cancellationToken)
    {
        ToggleButton(false);
        UpdateLabel("Generating...", true, true);

        var stream = inference.GenerateStream(prompt, cancellationToken);
        await foreach (var text in stream)
        {
            UpdateLabel(text, false);
            cancellationToken.ThrowIfCancellationRequested();
        }
        UpdateLabel("");
        UpdateLabel("Done!");
        ToggleButton(true);
    }

    void ToggleButton(bool isOn)
    {
        button.gameObject.SetActive(isOn);
    }

    void UpdateLabel(string text, bool isNewLine = true, bool clear = false)
    {
        if (clear)
        {
            sb.Clear();
        }
        if (isNewLine)
        {
            sb.AppendLine(text);
        }
        else
        {
            sb.Append(text);
        }
        // Using String Builder to reduce GC
        label.SetText(sb);
    }
}
