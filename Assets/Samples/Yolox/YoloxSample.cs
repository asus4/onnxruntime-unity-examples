using System;
using System.Text;
using Microsoft.ML.OnnxRuntime.Unity;
using Microsoft.ML.OnnxRuntime.Examples;
using TextureSource;
using UnityEngine;

[RequireComponent(typeof(VirtualTextureSource))]
public class YoloxSample : MonoBehaviour
{
    [SerializeField]
    private OrtAsset model;

    [SerializeField]
    private Yolox.Options options;

    [Header("Visualization Options")]
    [SerializeField]
    private TMPro.TMP_Text detectionBoxPrefab;

    [SerializeField]
    private RectTransform detectionContainer;

    [SerializeField]
    private int maxDetections = 20;

    private Yolox inference;
    private TMPro.TMP_Text[] detectionBoxes;
    private readonly StringBuilder sb = new();

    private void Start()
    {
        inference = new Yolox(model.bytes, options);

        detectionBoxes = new TMPro.TMP_Text[maxDetections];
        for (int i = 0; i < maxDetections; i++)
        {
            var box = Instantiate(detectionBoxPrefab, detectionContainer);
            box.name = $"Detection {i}";
            box.gameObject.SetActive(false);
            detectionBoxes[i] = box;
        }

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
        if (inference == null)
        {
            return;
        }

        inference.Run(texture);

        UpdateDetectionBox();
    }

    private void UpdateDetectionBox()
    {
        var labels = inference.Labels;
        var detections = inference.Detections;
        Vector2 size = detectionContainer.rect.size;

        int i;
        int length = Math.Min(detections.Count, maxDetections);
        for (i = 0; i < length; i++)
        {
            var detection = detections[i];
            string label = labels[detection.label];

            var box = detectionBoxes[i];
            box.gameObject.SetActive(true);

            // Using StringBuilder to reduce GC
            sb.Clear();
            sb.Append(label);
            sb.Append(": ");
            sb.Append((int)(detection.probability * 100));
            sb.Append('%');
            box.SetText(sb);

            RectTransform rt = box.rectTransform;
            Rect rect = inference.ConvertToViewport(detection.rect);
            rt.anchoredPosition = rect.min * size;
            rt.sizeDelta = rect.size * size;
        }
        for (; i < maxDetections; i++)
        {
            detectionBoxes[i].gameObject.SetActive(false);
        }
    }
}
