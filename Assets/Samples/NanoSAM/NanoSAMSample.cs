using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime.Examples;
using TextureSource;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

/// <summary>
/// Nvidia's NanoSAM Sample
/// https://github.com/NVIDIA-AI-IOT/nanosam/
/// 
/// See LICENSE for full license information.
[RequireComponent(typeof(VirtualTextureSource), typeof(NanoSAMVisualizer))]
public sealed class NanoSAMSample : MonoBehaviour
{
    [Header("NanoSAM Options")]
    [SerializeField]
    private RemoteFile encoderModelFile = new("https://huggingface.co/asus4/nanosam-ort/resolve/main/resnet18_image_encoder.with_runtime_opt.ort?download=true");

    [SerializeField]
    private RemoteFile decoderModelFile = new("https://huggingface.co/asus4/nanosam-ort/resolve/main/mobile_sam_mask_decoder.with_runtime_opt.ort?download=true");

    [SerializeField]
    private NanoSAM.Options options;

    [Header("UI")]
    [SerializeField]
    private RectTransform preview;

    [SerializeField]
    private GameObject loadingIndicator;

    [SerializeField]
    private Button resetButton;

    [SerializeField]
    private TMPro.TMP_Dropdown maskDropdown;

    [SerializeField]
    private Image positivePointPrefab;
    [SerializeField]
    private Image negativePointPrefab;

    private readonly List<NanoSAM.Point> points = new();
    private readonly List<Image> pointImages = new();
    private NanoSAM inference;
    private Texture inputTexture;
    private NanoSAMVisualizer visualizer;

    private async void Start()
    {
        // Show loading indicator
        loadingIndicator.SetActive(true);

        // Load model files, this will take some time at first run
        byte[] encoderModel = await encoderModelFile.Load();
        byte[] decoderModel = await decoderModelFile.Load();

        inference = new NanoSAM(encoderModel, decoderModel, options);
        visualizer = GetComponent<NanoSAMVisualizer>();

        // Register pointer down event to preview rect
        var callback = new EventTrigger.TriggerEvent();
        callback.AddListener((data) => OnPointerDown((PointerEventData)data));
        var trigger = preview.gameObject.AddComponent<EventTrigger>();
        trigger.triggers.Add(new()
        {
            eventID = EventTriggerType.PointerDown,
            callback = callback,
        });

        // Listen to texture update event
        if (TryGetComponent(out VirtualTextureSource source))
        {
            source.OnTexture.AddListener(OnTexture);
        }

        // Create mask dropdown options
        maskDropdown.ClearOptions();
        maskDropdown.AddOptions(new List<string> {
            "Negative", "Positive",
        });
        maskDropdown.value = 1;
        resetButton.onClick.AddListener(ResetMask);

        // Hide loading indicator
        loadingIndicator.SetActive(false);
    }

    private void OnDestroy()
    {
        if (resetButton != null)
        {
            resetButton.onClick.RemoveListener(ResetMask);
        }
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

    private void OnTexture(Texture texture)
    {
        inputTexture = texture;
    }

    private void OnPointerDown(PointerEventData data)
    {
        // Get click position in the rectTransform
        if (!RectTransformUtility.ScreenPointToLocalPointInRectangle(preview, data.position, data.pressEventCamera, out Vector2 rectPosition))
        {
            return;
        }

        // Normalize range to 0.0 - 1.0
        Vector2 point = Rect.PointToNormalized(preview.rect, rectPosition);
        // Flip Y axis (top 0.0 to bottom 1.0)
        point.y = 1.0f - point.y;

        // 0: negative, 1: positive
        int label = maskDropdown.value;

        // Create point object
        points.Add(new NanoSAM.Point(point, label));
        // Add image
        var prefab = label == 0 ? negativePointPrefab : positivePointPrefab;
        var image = Instantiate(prefab, preview);
        image.rectTransform.anchoredPosition = rectPosition;
        pointImages.Add(image);

        Debug.Log($"Add {(label == 0 ? "Negative" : "Positive")} point: {point}");
        Run(points);
    }

    private void ResetMask()
    {
        points.Clear();
        foreach (var image in pointImages)
        {
            Destroy(image.gameObject);
        }
        pointImages.Clear();

        inference.ResetOutput();
        visualizer.UpdateMask(inference.OutputMask, inputTexture);
    }

    private void Run(List<NanoSAM.Point> points)
    {
        if (inputTexture == null)
        {
            return;
        }

        inference.Run(inputTexture, points.AsReadOnly());
        visualizer.UpdateMask(inference.OutputMask, inputTexture);
    }
}
