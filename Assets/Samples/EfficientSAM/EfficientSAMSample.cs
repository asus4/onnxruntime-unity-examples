using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime.Examples;
using TextureSource;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

/// <summary>
/// EfficientSAM Sample
/// https://github.com/yformer/EfficientSAM
/// 
/// See LICENSE for full license information.
[RequireComponent(typeof(VirtualTextureSource), typeof(EfficientSAMVisualizer))]
public sealed class EfficientSAMSample : MonoBehaviour
{
    [Header("EfficientSAM Options")]
    [SerializeField]
    private RemoteFile modelFile = new("https://github.com/yformer/EfficientSAM/raw/main/weights/efficient_sam_vitt.onnx");

    [SerializeField]
    private EfficientSAM.Options options;

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

    private readonly List<EfficientSAM.Point> points = new();
    private readonly List<Image> pointImages = new();
    private EfficientSAM inference;
    private Texture inputTexture;
    private EfficientSAMVisualizer visualizer;

    private async void Start()
    {
        // Show loading indicator
        loadingIndicator.SetActive(true);

        var token = destroyCancellationToken;
        byte[] modelBytes = await modelFile.Load(token);

        inference = new EfficientSAM(modelBytes, options);
        visualizer = GetComponent<EfficientSAMVisualizer>();

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
        points.Add(new EfficientSAM.Point(point, label));
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
        Vector2Int inputSize = inference.InputSize;
        visualizer.UpdateMask(inference.OutputMask, inputSize, inputTexture);
    }

    private void Run(List<EfficientSAM.Point> points)
    {
        if (inputTexture == null)
        {
            return;
        }

        // TODO: Implement async run
        inference.Run(inputTexture, points.AsReadOnly());
        Vector2Int inputSize = inference.InputSize;
        visualizer.UpdateMask(inference.OutputMask, inputSize, inputTexture);
    }
}
