using System;
using Microsoft.ML.OnnxRuntime.Unity;
using UnityEngine;
using UnityEngine.Events;
using Unity.Profiling;

namespace Microsoft.ML.OnnxRuntime.Examples
{
    public class EfficientSAMVisualizer : MonoBehaviour
    {
        [Serializable]
        public class TextureEvent : UnityEvent<Texture> { }

        [SerializeField]
        private ComputeShader tensorToTexShader;

        [SerializeField]
        [Min(0)]
        private float threshold = 0.1f;

        [Header("Bilateral Filter")]
        [SerializeField]
        private bool useFilter = true;

        [SerializeField]
        private ComputeShader filterShader;

        [SerializeField]
        [Min(0)]
        private float sigmaSpace;

        [SerializeField]
        [Range(0, 1)]
        private float sigmaColor;

        public TextureEvent onTexture = new();

        private static readonly int _InputTensor = Shader.PropertyToID("_InputTensor");
        private static readonly int _OutputTex = Shader.PropertyToID("_OutputTex");
        private static readonly int _Threshold = Shader.PropertyToID("_Threshold");

        private RenderTexture renderTexture;
        private ComputeBuffer maskBuffer;
        private int kernel;

        private JointBilateralFilter filter;

        static readonly ProfilerMarker tensorToTexProfMarker = new($"{typeof(EfficientSAMVisualizer).Name}.Run");
        static readonly ProfilerMarker filterProfMarker = new($"{typeof(JointBilateralFilter).Name}.Run");

        public float Threshold
        {
            get => threshold;
            set
            {
                threshold = Math.Max(value, 0);
                tensorToTexShader.SetFloat(_Threshold, value);
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

            kernel = tensorToTexShader.FindKernel("VisualizeMask");
            tensorToTexShader.SetInts("_OutputSize", renderTexture.width, renderTexture.height);
            Threshold = threshold;

            filter = new JointBilateralFilter(filterShader);
        }

        private void OnDestroy()
        {
            if (renderTexture != null)
            {
                renderTexture.Release();
                Destroy(renderTexture);
            }
            maskBuffer?.Release();
            filter?.Dispose();
        }

#if UNITY_EDITOR
        private void OnValidate()
        {
            if (!Application.isPlaying || maskBuffer == null)
            {
                return;
            }
            Threshold = threshold;
            filter.SigmaSpace = sigmaSpace;
            filter.SigmaColor = sigmaColor;
        }
#endif // UNITY_EDITOR

        public void UpdateMask(ReadOnlySpan<float> outputMask, Vector2Int maskSize, Texture guide)
        {
            EnsureBuffer(maskSize.x, maskSize.y);

            tensorToTexProfMarker.Begin();
            if (outputMask != null)
            {
                maskBuffer.SetData(outputMask);
            }
            tensorToTexShader.SetBuffer(kernel, _InputTensor, maskBuffer);
            tensorToTexShader.SetTexture(kernel, _OutputTex, renderTexture);
            tensorToTexShader.Dispatch(kernel, renderTexture.width / 8, renderTexture.height / 8, 1);
            tensorToTexProfMarker.End();

            filterProfMarker.Begin();
            var tex = useFilter
                ? filter.Run(renderTexture, guide)
                : renderTexture;
            filterProfMarker.End();

            onTexture.Invoke(tex);
        }

        private void EnsureBuffer(int width, int height)
        {
            if (renderTexture != null && renderTexture.width == width && renderTexture.height == height)
            {
                return;
            }
            if (renderTexture != null)
            {
                renderTexture.Release();
                Destroy(renderTexture);
            }

            renderTexture = new RenderTexture(width, height, 0, RenderTextureFormat.ARGBHalf)
            {
                enableRandomWrite = true
            };
            renderTexture.Create();

            maskBuffer?.Release();
            maskBuffer = new ComputeBuffer(3 * width * height, sizeof(float));

            tensorToTexShader.SetInts("_OutputSize", width, height);
        }
    }
}
