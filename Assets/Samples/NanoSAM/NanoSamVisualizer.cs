using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using Microsoft.ML.OnnxRuntime.Unity;
using Unity.Mathematics;

namespace Microsoft.ML.OnnxRuntime.Examples
{
    public class NanoSAMVisualizer : MonoBehaviour
    {
        [Serializable]
        public class TextureEvent : UnityEvent<Texture> { }

        [SerializeField]
        private ComputeShader shader;

        public TextureEvent onTexture = new();

        private static readonly int _InputTensor = Shader.PropertyToID("_InputTensor");
        private static readonly int _OutputTex = Shader.PropertyToID("_OutputTex");

        private RenderTexture renderTexture;
        private ComputeBuffer maskBuffer;
        private int kernel;

        private void Start()
        {
            renderTexture = new RenderTexture(256, 256, 0, RenderTextureFormat.ARGBHalf);
            renderTexture.enableRandomWrite = true;
            renderTexture.Create();

            maskBuffer = new ComputeBuffer(4 * 256 * 256, sizeof(float));

            kernel = shader.FindKernel("VisualizeMask");
            shader.SetInts("_OutputSize", renderTexture.width, renderTexture.height);
        }

        private void OnDestroy()
        {
            if (renderTexture != null)
            {
                renderTexture.Release();
                Destroy(renderTexture);
            }
            maskBuffer?.Release();
        }

        public void UpdateMask(ReadOnlySpan<float> outputMask)
        {
            maskBuffer.SetData(outputMask);
            shader.SetBuffer(kernel, _InputTensor, maskBuffer);
            shader.SetTexture(kernel, _OutputTex, renderTexture);
            shader.Dispatch(kernel, renderTexture.width / 8, renderTexture.height / 8, 1);

            onTexture.Invoke(renderTexture);
        }
    }
}
