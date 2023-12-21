using System.Text;
using Microsoft.ML.OnnxRuntime.Unity;
using Microsoft.ML.OnnxRuntime.Examples;
using TextureSource;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;
using UnityEngine.Pool;

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
    private RectTransform detectionBoxParent;

    [SerializeField]
    private int maxDetections = 20;

    private Yolox inference;
    private TMPro.TMP_Text[] detectionBoxes;

    private void Start()
    {
        inference = new Yolox(model.bytes, options);

        detectionBoxes = new TMPro.TMP_Text[maxDetections];
        for (int i = 0; i < maxDetections; i++)
        {
            detectionBoxes[i] = Instantiate(detectionBoxPrefab, detectionBoxParent);
            detectionBoxes[i].gameObject.SetActive(false);
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
        inference?.Run(texture);
    }
}
