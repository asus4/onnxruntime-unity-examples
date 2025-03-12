using System.Text;
using Microsoft.ML.OnnxRuntime.Unity;
using Microsoft.ML.OnnxRuntime.Examples;
using TextureSource;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;

[RequireComponent(typeof(VirtualTextureSource))]
public sealed class MobileOneSample : MonoBehaviour
{
    [System.Serializable]
    public class TextUnityEvent : UnityEvent<string> { }

    [SerializeField]
    OrtAsset model;

    [SerializeField]
    MobileOne.Options options;

    [SerializeField]
    RawImage debugImage;

    [SerializeField]
    bool runBackground = false;

    MobileOne inference;

    public TextUnityEvent onDebugTopK;

    readonly StringBuilder sb = new();

    void Start()
    {
        inference = new MobileOne(model.bytes, options);

        // Listen to OnTexture event from VirtualTextureSource
        // Also able to bind in the inspector
        if (TryGetComponent(out VirtualTextureSource source))
        {
            source.OnTexture.AddListener(OnTexture);
        }
    }

    void OnDestroy()
    {
        if (TryGetComponent(out VirtualTextureSource source))
        {
            source.OnTexture.RemoveListener(OnTexture);
        }

        inference?.Dispose();
    }

    public void OnTexture(Texture texture)
    {
        if (!runBackground)
        {
            Run(texture);
        }
    }

    void Run(Texture texture)
    {
        inference?.Run(texture);
        ShowLabels();
    }

    void ShowLabels()
    {
        var texture = inference.InputTexture;
        debugImage.texture = texture;

        var labelNames = inference.labelNames;

        sb.Clear();
        sb.AppendLine($"Input: {texture.width}x{texture.height}");
        sb.AppendLine($"Top K:");
        foreach (var label in inference.TopKLabels)
        {
            sb.AppendLine($"{labelNames[label.index]} ({label.score})");
        }
        onDebugTopK.Invoke(sb.ToString());
    }
}
