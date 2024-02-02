using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
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

        [SerializeField]
        private Color[] maskColors = new Color[]
        {
            new Color(0f, 0.725f, 0.46f, 1f),
        };

        [SerializeField]
        [Min(0)]
        private float threshold = 0.1f;

        public TextureEvent onTexture = new();

        private static readonly int _InputTensor = Shader.PropertyToID("_InputTensor");
        private static readonly int _OutputTex = Shader.PropertyToID("_OutputTex");
        private static readonly int _Threshold = Shader.PropertyToID("_Threshold");
        private static readonly int _MaskColors = Shader.PropertyToID("_MaskColors");

        private RenderTexture renderTexture;
        private ComputeBuffer maskBuffer;
        private int kernel;

        public Color[] MaskColors
        {
            get => maskColors;
            set
            {
                maskColors = value;
                var vectors = value.Select(c => (Vector4)c).ToArray();
                shader.SetVectorArray(_MaskColors, vectors);
            }
        }

        public float Threshold
        {
            get => threshold;
            set
            {
                threshold = Math.Max(value, 0);
                shader.SetFloat(_Threshold, value);
            }
        }

        private void Start()
        {
            renderTexture = new RenderTexture(256, 256, 0, RenderTextureFormat.ARGBHalf);
            renderTexture.enableRandomWrite = true;
            renderTexture.Create();

            maskBuffer = new ComputeBuffer(4 * 256 * 256, sizeof(float));

            kernel = shader.FindKernel("VisualizeMask");
            shader.SetInts("_OutputSize", renderTexture.width, renderTexture.height);
            Threshold = threshold;
            MaskColors = maskColors;
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

#if UNITY_EDITOR
        private void OnValidate()
        {
            Threshold = threshold;
            MaskColors = maskColors;
        }
#endif // UNITY_EDITOR

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
