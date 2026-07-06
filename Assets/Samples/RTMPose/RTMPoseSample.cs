using System.Text;
using Microsoft.ML.OnnxRuntime.Unity;
using Microsoft.ML.OnnxRuntime.Examples;
using TextureSource;
using UnityEngine;

/// <summary>
/// Multi-person 2D pose estimation example.
/// Two-stage top-down pipeline:
/// 1. YOLOX-Nano detects persons
/// 2. RTMPose-t estimates keypoints for each person
/// </summary>
[RequireComponent(typeof(VirtualTextureSource))]
public class RTMPoseSample : MonoBehaviour
{
    [Header("Person Detection (YOLOX)")]
    [SerializeField]
    private OrtAsset detectionModel;

    [SerializeField]
    private Yolox.Options detectionOptions;

    [Header("Pose Estimation (RTMPose)")]
    [SerializeField]
    private OrtAsset poseModel;

    [SerializeField]
    private RemoteFile poseModelFile = new("https://github.com/asus4/onnxruntime-unity-examples/releases/download/v0.4.8/rtmpose-t_body7_256x192.with_runtime_opt.ort");

    [SerializeField]
    private RTMPose.Options poseOptions;

    [Header("Visualization Options")]
    [SerializeField]
    private TMPro.TMP_Text detectionBoxPrefab;

    [SerializeField]
    private RectTransform detectionContainer;

    [SerializeField]
    private PoseVisualizer poseVisualizer;

    // "person" label in COCO dataset
    private const int PERSON_LABEL = 0;

    private Yolox detector;
    private RTMPose pose;
    private RTMPose.Pose[] results;
    private TMPro.TMP_Text[] detectionBoxes;
    private readonly StringBuilder sb = new();

    private async void Start()
    {
        detector = new Yolox(detectionModel.bytes, detectionOptions);

        byte[] poseModelBytes = poseModel != null
            ? poseModel.bytes
            : await poseModelFile.Load(destroyCancellationToken);
        pose = new RTMPose(poseModelBytes, poseOptions);

        int maxPoses = poseOptions.maxPoses;
        results = new RTMPose.Pose[maxPoses];
        detectionBoxes = new TMPro.TMP_Text[maxPoses];
        for (int i = 0; i < maxPoses; i++)
        {
            results[i] = new RTMPose.Pose();

            var box = Instantiate(detectionBoxPrefab, detectionContainer);
            box.name = $"Person {i}";
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

        detector?.Dispose();
        pose?.Dispose();
    }

    public void OnTexture(Texture texture)
    {
        if (detector == null || pose == null)
        {
            return;
        }

        detector.Run(texture);

        // Run pose estimation for each person.
        // Detections are already sorted by probability and NMS-ed.
        int count = 0;
        foreach (var detection in detector.Detections)
        {
            if (detection.label != PERSON_LABEL)
            {
                continue;
            }

            Rect viewportRect = detector.ConvertToViewport(detection.rect);
            Rect cropRect = pose.GetCropRect(viewportRect, texture.width, texture.height);
            var keypoints = pose.Run(texture, cropRect);

            // Copy results as the keypoint buffer is reused for the next person
            var result = results[count];
            result.rect = viewportRect;
            result.probability = detection.probability;
            keypoints.CopyTo(result.keypoints);

            count++;
            if (count >= results.Length)
            {
                break;
            }
        }

        UpdateVisualization(count);
    }

    private void UpdateVisualization(int count)
    {
        Vector2 viewportSize = detectionContainer.rect.size;

        int i;
        for (i = 0; i < count; i++)
        {
            var result = results[i];

            var box = detectionBoxes[i];
            box.gameObject.SetActive(true);

            // Using StringBuilder to reduce GC
            sb.Clear();
            sb.Append("person: ");
            sb.Append((int)(result.probability * 100));
            sb.Append('%');
            box.SetText(sb);

            RectTransform rt = box.rectTransform;
            rt.anchoredPosition = result.rect.min * viewportSize;
            rt.sizeDelta = result.rect.size * viewportSize;
        }
        // Hide unused boxes
        for (; i < detectionBoxes.Length; i++)
        {
            detectionBoxes[i].gameObject.SetActive(false);
        }

        poseVisualizer.SetPoses(results, count, poseOptions.keypointThreshold);
    }
}
