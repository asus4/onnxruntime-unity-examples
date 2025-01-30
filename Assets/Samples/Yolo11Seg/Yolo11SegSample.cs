using System;
using System.Text;
using Microsoft.ML.OnnxRuntime.Unity;
using Microsoft.ML.OnnxRuntime.Examples;
using TextureSource;
using UnityEngine;
using UnityEngine.InputSystem;

[RequireComponent(typeof(VirtualTextureSource))]
public class Yolo11SegSample : MonoBehaviour
{
    [SerializeField]
    private OrtAsset model;

    [SerializeField]
    private Yolo11Seg.Options options;

    private Yolo11Seg inference;

    private void Start()
    {
        inference = new Yolo11Seg(model.bytes, options);

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
        inference.Run(texture);

        if (Keyboard.current[Key.Space].wasPressedThisFrame)
        {
            string desktop = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
            string path = System.IO.Path.Combine(desktop, "output0.csv");
            inference.SaveOutputToFile(path);
        }
    }
}
