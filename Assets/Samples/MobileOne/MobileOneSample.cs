using System.Text;
using Microsoft.ML.OnnxRuntime.Unity;
using Microsoft.ML.OnnxRuntime.Examples;
using TextureSource;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;

[RequireComponent(typeof(VirtualTextureSource))]
public class MobileOneSample : MonoBehaviour
{
    [System.Serializable]
    public class TextUnityEvent : UnityEvent<string> { }

    [SerializeField]
    private OrtAsset model;

    [SerializeField]
    private MobileOne.Options options;

    [SerializeField]
    private RawImage debugImage;


    private MobileOne inference;

    public TextUnityEvent onDebugTopK;

    private readonly StringBuilder sb = new();

    private void Start()
    {
        inference = new MobileOne(model.bytes, options);

        // Listen to OnTexture event from VirtualTextureSource
        // Also able to bind in the inspector
        if (TryGetComponent(out VirtualTextureSource source))
        {
            source.OnTexture.AddListener(OnTexture);
        }
    }

    private void OnDestroy()
    {
        if (TryGetComponent(out VirtualTextureSource source))
        {
            source.OnTexture.RemoveListener(OnTexture);
        }

        inference?.Dispose();
    }

    public void OnTexture(Texture texture)
    {
        inference?.Run(texture);

        DebugLabels();
    }

    private void DebugLabels()
    {
        var texture = inference.InputTexture;
        debugImage.texture = texture;

        sb.Clear();
        sb.AppendLine($"Input: {texture.width}x{texture.height}");
        sb.AppendLine($"Top K:");
        foreach (var label in inference.TopKLabels)
        {
            sb.AppendLine($"{label.name} ({label.score})");
        }
        onDebugTopK.Invoke(sb.ToString());
    }
}
