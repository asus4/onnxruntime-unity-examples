using System;
using Microsoft.ML.OnnxRuntime.Examples;
using UnityEngine;
using TMPro;
using UnityEngine.UI;
using System.Text;
using System.Threading;

public class Phi4MultiModalSample : MonoBehaviour
{
    [SerializeField]
    Phi4MultiModal.Options options;

    [SerializeField]
    TMP_Text label;

    [SerializeField]
    Button button;

    [SerializeField]
    string prompt = "What is the capital of France?";

    readonly StringBuilder sb = new();
    Phi4MultiModal inference;

    async Awaitable Start()
    {
        if (Application.isMobilePlatform)
        {
            Debug.LogError("Phi4MultiModal is not supported on mobile yet.");
            UpdateLabel("Phi4MultiModal is not supported on mobile yet.", true, true);
            ToggleButton(false);
            return;
        }

        ToggleButton(false);
        UpdateLabel("Now loading LLM model. wait a bit...");

        try
        {
            inference = await Phi4MultiModal.InitAsync(options, destroyCancellationToken);
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
