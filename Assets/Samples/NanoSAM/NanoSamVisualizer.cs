using System;
using UnityEngine;
using UnityEngine.Events;
using Microsoft.ML.OnnxRuntime.Unity;

namespace Microsoft.ML.OnnxRuntime.Examples
{
    public class NanoSAMVisualizer : MonoBehaviour
    {
        [Serializable]
        public class TextureEvent : UnityEvent<Texture> { }

        [SerializeField]
        private ComputeShader shader;

        [SerializeField]
        private Color maskColor = new(0f, 0.725f, 0.46f, 1f);

        [SerializeField]
        [Min(0)]
        private float threshold = 0.1f;

        public TextureEvent onTexture = new();

        private static readonly int _InputTensor = Shader.PropertyToID("_InputTensor");
        private static readonly int _OutputTex = Shader.PropertyToID("_OutputTex");
        private static readonly int _Threshold = Shader.PropertyToID("_Threshold");
        private static readonly int _MaskColor = Shader.PropertyToID("_MaskColor");

        private RenderTexture renderTexture;
        private ComputeBuffer maskBuffer;
        private int kernel;

        public Color MaskColor
        {
            get => maskColor;
            set
            {
                maskColor = value;
                shader.SetVector(_MaskColor, maskColor);
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
            renderTexture = new RenderTexture(256, 256, 0, RenderTextureFormat.ARGBHalf)
            {
                enableRandomWrite = true
            };
            renderTexture.Create();

            maskBuffer = new ComputeBuffer(4 * 256 * 256, sizeof(float));

            kernel = shader.FindKernel("VisualizeMask");
            shader.SetInts("_OutputSize", renderTexture.width, renderTexture.height);
            MaskColor = maskColor;
            Threshold = threshold;
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
            if (!Application.isPlaying || maskBuffer == null)
            {
                return;
            }
            MaskColor = maskColor;
            Threshold = threshold;
            UpdateMask(null);
        }
#endif // UNITY_EDITOR

        public void UpdateMask(ReadOnlySpan<float> outputMask)
        {
            if (outputMask != null)
            {
                maskBuffer.SetData(outputMask);
            }
            shader.SetBuffer(kernel, _InputTensor, maskBuffer);
            shader.SetTexture(kernel, _OutputTex, renderTexture);
            shader.Dispatch(kernel, renderTexture.width / 8, renderTexture.height / 8, 1);

            onTexture.Invoke(renderTexture);
        }
    }
}
