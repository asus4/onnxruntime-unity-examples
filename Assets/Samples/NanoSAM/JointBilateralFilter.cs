
using System;
using System.Runtime.CompilerServices;
using UnityEngine;

public sealed class JointBilateralFilter : IDisposable
{
    /// <summary>
    /// Pixel radius: use (sigma_space*2+1)x(sigma_space*2+1) window.
    /// This should be set based on output image pixel space.
    /// </summary>
    public float SigmaSpace { get; set; } = 2f;

    /// <summary>
    /// Color variance: normalized [0-1] color difference allowed.
    /// </summary>
    public float SigmaColor { get; set; } = 1f;

    private readonly ComputeShader shader;

    private RenderTexture resultTexture;

    private static readonly int _InputMaskTex = Shader.PropertyToID("_InputMaskTex");
    private static readonly int _InputGuideTex = Shader.PropertyToID("_InputGuideTex");
    private static readonly int _OutputTex = Shader.PropertyToID("_OutputTex");
    private static readonly int _InputGuideTex_TexelSize = Shader.PropertyToID("_InputGuideTex_TexelSize");
    private static readonly int _OutputSize = Shader.PropertyToID("_OutputSize");

    private readonly int kernel;

    public JointBilateralFilter(ComputeShader shader)
    {
        this.shader = shader;

        kernel = shader.FindKernel("JointBilateralFilter");
    }

    public void Dispose()
    {
        if (resultTexture != null)
        {
            resultTexture.Release();
            UnityEngine.Object.Destroy(resultTexture);
        }
    }

    public Texture Run(Texture mask, Texture guide)
    {
        EnsureRenderTexture(ref resultTexture, guide.width, guide.height);

        UpdateSigmas();

        shader.SetTexture(kernel, _InputMaskTex, mask);
        shader.SetTexture(kernel, _InputGuideTex, guide);
        shader.SetTexture(kernel, _OutputTex, resultTexture);
        shader.SetVector(_InputGuideTex_TexelSize, new Vector4(1.0f / guide.width, 1.0f / guide.height, guide.width, guide.height));
        shader.SetInts(_OutputSize, resultTexture.width, resultTexture.height);

        shader.Dispatch(kernel, Mathf.CeilToInt(guide.width / 8.0f), Mathf.CeilToInt(guide.height / 8.0f), 1);

        return resultTexture;
    }

    private void UpdateSigmas()
    {
        const float kSparsityFactor = 0.66f;
        float sparsity = Mathf.Max(1.0f, Mathf.Sqrt(SigmaSpace) * kSparsityFactor);
        float step = sparsity;
        float radius = SigmaSpace;
        float offset = step > 1.0f ? (step * 0.5f) : 0.0f;

        shader.SetFloat("_SigmaSpace", SigmaSpace);
        shader.SetFloat("_SigmaColor", SigmaColor);
        shader.SetFloat("_Step", step);
        shader.SetFloat("_Radius", radius);
        shader.SetFloat("_Offset", offset);
    }

    private static void EnsureRenderTexture(ref RenderTexture tex, int width, int height)
    {
        if (tex == null || tex.width != width || tex.height != height)
        {
            if (tex != null)
            {
                tex.Release();
                UnityEngine.Object.Destroy(tex);
            }

            tex = new RenderTexture(width, height, 0, RenderTextureFormat.ARGBHalf)
            {
                enableRandomWrite = true
            };
            tex.Create();
        }
    }
}
