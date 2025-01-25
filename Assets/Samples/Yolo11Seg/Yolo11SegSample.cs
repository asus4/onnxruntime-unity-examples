using System;
using System.Text;
using Microsoft.ML.OnnxRuntime.Unity;
using Microsoft.ML.OnnxRuntime.Examples;
using TextureSource;
using UnityEngine;

[RequireComponent(typeof(VirtualTextureSource))]
public class Yolo11SegSample : MonoBehaviour
{
    [SerializeField]
    private OrtAsset model;

    private void Start()
    {
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
    }

    public void OnTexture(Texture texture)
    {

    }
}
