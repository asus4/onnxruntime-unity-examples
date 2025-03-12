using System;
using System.Text;
using Microsoft.ML.OnnxRuntime.Unity;
using Microsoft.ML.OnnxRuntime.Examples;
using TextureSource;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;
using System.Threading;

[RequireComponent(typeof(VirtualTextureSource))]
public class Yolo11SegSample : MonoBehaviour
{
    [Serializable]
    public class TextureEvent : UnityEvent<Texture> { }
    [Serializable]
    public class AspectChangeEvent : UnityEvent<float> { }

    [SerializeField]
    private OrtAsset model;

    [SerializeField]
    private RemoteFile modelFile = new("https://github.com/asus4/onnxruntime-unity-examples/releases/download/v0.2.7/yolo11n-seg-dynamic.onnx");

    [SerializeField]
    private Yolo11Seg.Options options;

    [SerializeField]
    private bool runBackground = false;

    [Header("Visualization Options")]
    [SerializeField]
    private TMPro.TMP_Text detectionBoxPrefab;

    [SerializeField]
    private RectTransform detectionContainer;

    [SerializeField]
    private int maxDetections = 20;

    public TextureEvent OnSegmentationTexture = new();
    public AspectChangeEvent OnSegmentationAspectChange = new();

    private Yolo11Seg inference;
    private TMPro.TMP_Text[] detectionBoxes;
    private Image[] detectionBoxOutline;
    private Texture prevSegmentationTexture;
    private readonly StringBuilder sb = new();
    private Awaitable currentTask = null;

    private async void Start()
    {
        byte[] onnxFile = model != null
            ? model.bytes
            : await modelFile.Load(destroyCancellationToken);
        inference = new Yolo11Seg(onnxFile, options);

        detectionBoxes = new TMPro.TMP_Text[maxDetections];
        detectionBoxOutline = new Image[maxDetections];
        for (int i = 0; i < maxDetections; i++)
        {
            var box = Instantiate(detectionBoxPrefab, detectionContainer);
            box.name = $"Detection {i}";
            box.gameObject.SetActive(false);
            detectionBoxes[i] = box;
            detectionBoxOutline[i] = box.transform.GetChild(0).GetComponent<Image>();
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
        prevSegmentationTexture = null;
    }

    public void OnTexture(Texture texture)
    {
        if (inference == null)
        {
            return;
        }

        if (runBackground)
        {
            // Async version
            bool isNextAvailable = currentTask == null || currentTask.IsCompleted;
            if (isNextAvailable)
            {
                currentTask = RunAsync(texture, destroyCancellationToken);
            }
        }
        else
        {
            // Sync version
            Run(texture);
        }
    }

    private void Run(Texture texture)
    {
        inference.Run(texture);

        UpdateDetectionBox(inference.Detections);

        // Invoke events when the segmentation texture is updated
        var segTex = inference.SegmentationTexture;
        if (prevSegmentationTexture != segTex)
        {
            OnSegmentationTexture.Invoke(segTex);
            OnSegmentationAspectChange.Invoke((float)segTex.width / segTex.height);
            prevSegmentationTexture = segTex;
        }
    }

    private async Awaitable RunAsync(Texture texture, CancellationToken cancellationToken)
    {
        try
        {
            await inference.RunAsync(texture, cancellationToken);
        }
        catch (OperationCanceledException e)
        {
            Debug.LogWarning(e);
            return;
        }
        await Awaitable.MainThreadAsync();

        UpdateDetectionBox(inference.Detections);

        // Invoke events when the segmentation texture is updated
        var segTex = inference.SegmentationTexture;
        if (prevSegmentationTexture != segTex)
        {
            OnSegmentationTexture.Invoke(segTex);
            OnSegmentationAspectChange.Invoke((float)segTex.width / segTex.height);
            prevSegmentationTexture = segTex;
        }
    }

    private void UpdateDetectionBox(ReadOnlySpan<Yolo11Seg.Detection> detections)
    {
        var labels = inference.labelNames;
        Vector2 viewportSize = detectionContainer.rect.size;

        int i;
        int length = Math.Min(detections.Length, maxDetections);
        for (i = 0; i < length; i++)
        {
            var detection = detections[i];

            var color = detection.GetColor();

            var box = detectionBoxes[i];
            box.gameObject.SetActive(true);

            // Using StringBuilder to reduce GC
            sb.Clear();
            sb.Append(labels[detection.label]);
            sb.Append(": ");
            sb.Append((int)(detection.probability * 100));
            sb.Append('%');
            box.SetText(sb);
            box.color = color;

            // The detection rect is model space
            // Needs to be converted to viewport space
            RectTransform rt = box.rectTransform;
            Rect rect = inference.ConvertToViewport(detection.rect);
            rt.anchoredPosition = rect.min * viewportSize;
            rt.sizeDelta = rect.size * viewportSize;

            detectionBoxOutline[i].color = color;
        }

        // Hide unused boxes
        for (; i < maxDetections; i++)
        {
            detectionBoxes[i].gameObject.SetActive(false);
        }
    }
}
