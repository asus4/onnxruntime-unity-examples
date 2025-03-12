using System.Text;
using System.Threading;
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
    Awaitable currentAwaitable = null;

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
        if (runBackground)
        {
            bool isNextAvailable = currentAwaitable == null || currentAwaitable.IsCompleted;
            if (isNextAvailable)
            {
                currentAwaitable = RunAsync(texture, destroyCancellationToken);
            }
        }
        else
        {
            Run(texture);
        }
    }

    void Run(Texture texture)
    {
        inference?.Run(texture);
        ShowLabels();
    }

    async Awaitable RunAsync(Texture texture, CancellationToken cancellationToken)
    {
        await inference.RunAsync(texture, cancellationToken);
        await Awaitable.MainThreadAsync();
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
