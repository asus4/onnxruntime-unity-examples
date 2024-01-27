using Microsoft.ML.OnnxRuntime.Examples;
using Microsoft.ML.OnnxRuntime.Unity;
using TextureSource;
using UnityEngine;
using UnityEngine.EventSystems;

/// <summary>
/// Nvidia's NanoSAM Sample
/// https://github.com/NVIDIA-AI-IOT/nanosam/
/// 
/// See LICENSE for full license information.
[RequireComponent(typeof(VirtualTextureSource))]
public sealed class NanoSAMSample : MonoBehaviour
{
    [Header("NanoSAM Options")]
    [SerializeField]
    private OrtAsset encoderModel;

    [SerializeField]
    private OrtAsset decoderModel;

    [SerializeField]
    private NanoSAM.Options options;

    [Header("UI")]
    [SerializeField]
    private RectTransform preview;

    private NanoSAM inference;
    private Texture inputTexture;

    private void Start()
    {
        inference = new NanoSAM(encoderModel.bytes, decoderModel.bytes, options);

        if (TryGetComponent(out VirtualTextureSource source))
        {
            source.OnTexture.AddListener(OnTexture);
        }

        // Register pointer down event
        EventTrigger.TriggerEvent callback = new();
        callback.AddListener((data) => OnPointerDown((PointerEventData)data));
        var trigger = preview.gameObject.AddComponent<EventTrigger>();
        trigger.triggers.Add(new()
        {
            eventID = EventTriggerType.PointerDown,
            callback = callback,
        });
    }

    private void OnDestroy()
    {
        if (preview != null && preview.TryGetComponent(out EventTrigger trigger))
        {
            Destroy(trigger);
        }
        if (TryGetComponent(out VirtualTextureSource source))
        {
            source.OnTexture.RemoveListener(OnTexture);
        }
        inference?.Dispose();
    }

    public void OnTexture(Texture texture)
    {
        inputTexture = texture;
    }

    private void OnPointerDown(PointerEventData data)
    {
        if (!RectTransformUtility.ScreenPointToLocalPointInRectangle(preview, data.position, data.pressEventCamera, out Vector2 position))
        {
            return;
        }

        position = Rect.PointToNormalized(preview.rect, position);
        // Flip Y axis to align with the model coordinate
        position.y = 1.0f - position.y;
        Debug.Log($"norm:{position}");

        Run(inputTexture, position);
    }

    private void Run(Texture texture, Vector2 point)
    {
        inference.Run(texture, point);
    }
}
