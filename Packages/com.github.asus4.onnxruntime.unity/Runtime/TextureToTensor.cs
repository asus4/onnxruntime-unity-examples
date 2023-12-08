namespace Microsoft.ML.OnnxRuntime.Unity
{
    using System;
    using System.Runtime.InteropServices;
    using UnityEngine;

    /// <summary>
    /// Port from TextureSource: MIT License
    /// https://github.com/asus4/TextureSource
    /// </summary>
    public class TextureToTensor<T> : IDisposable
        where T : unmanaged
    {
        private static ComputeShader compute;
        private static readonly int _InputTex = Shader.PropertyToID("_InputTex");
        private static readonly int _OutputTex = Shader.PropertyToID("_OutputTex");
        private static readonly int _OutputTensor = Shader.PropertyToID("_OutputTensor");
        private static readonly int _OutputSize = Shader.PropertyToID("_OutputSize");
        private static readonly int _TransformMatrix = Shader.PropertyToID("_TransformMatrix");

        private static readonly Matrix4x4 PopMatrix = Matrix4x4.Translate(new Vector3(0.5f, 0.5f, 0));
        private static readonly Matrix4x4 PushMatrix = Matrix4x4.Translate(new Vector3(-0.5f, -0.5f, 0));


        private readonly int kernel;
        private readonly RenderTexture texture;
        private readonly GraphicsBuffer tensor;
        private const int CHANNELS = 3; // RGB for now
        public readonly int width;
        public readonly int height;

        public RenderTexture Texture => texture;
        private readonly T[] tensorData;
        public ReadOnlySpan<T> TensorData => tensorData;

        public TextureToTensor(int width, int height)
        {
            this.width = width;
            this.height = height;

            var desc = new RenderTextureDescriptor(width, height, RenderTextureFormat.ARGB32)
            {
                enableRandomWrite = true,
                useMipMap = false,
                depthBufferBits = 0,
            };
            texture = new RenderTexture(desc);
            texture.Create();

            int stride = Marshal.SizeOf(default(T));
            tensor = new GraphicsBuffer(GraphicsBuffer.Target.Structured, CHANNELS * width * height, stride);
            tensorData = new T[CHANNELS * width * height];

            if (compute == null)
            {
                const string SHADER_PATH = "com.github.asus4.onnxruntime.unity/TextureToTensor";
                compute = Resources.Load<ComputeShader>(SHADER_PATH);
            }
            kernel = compute.FindKernel("TextureToTensor");
        }

        public void Dispose()
        {
            texture.Release();
            UnityEngine.Object.Destroy(texture);
            tensor.Release();
        }

        public RenderTexture Transform(Texture input, Matrix4x4 t)
        {
            compute.SetTexture(kernel, _InputTex, input, 0);
            compute.SetTexture(kernel, _OutputTex, texture, 0);
            compute.SetBuffer(kernel, _OutputTensor, tensor);
            compute.SetInts(_OutputSize, texture.width, texture.height);
            compute.SetMatrix(_TransformMatrix, t);
            compute.Dispatch(kernel, Mathf.CeilToInt(texture.width / 8f), Mathf.CeilToInt(texture.height / 8f), 1);

            tensor.GetData(tensorData);
            return texture;
        }

        public RenderTexture Transform(Texture input, Vector2 translate, float eulerRotation, Vector2 scale)
        {
            Matrix4x4 trs = Matrix4x4.TRS(
                new Vector3(-translate.x, -translate.y, 0),
                Quaternion.Euler(0, 0, -eulerRotation),
                new Vector3(1f / scale.x, 1f / scale.y, 1));
            return Transform(input, PopMatrix * trs * PushMatrix);
        }

        public RenderTexture Transform(Texture input, AspectMode aspectMode)
        {
            float srcAspect = (float)input.width / input.height;
            float dstAspect = (float)width / height;
            GetTextureST(srcAspect, dstAspect, aspectMode,
                out Vector2 scale, out Vector2 translate);
            return Transform(input, translate, 0, scale);
        }

        public static void GetTextureST(float srcAspect, float dstAspect, AspectMode mode,
            out Vector2 scale, out Vector2 translate)
        {
            switch (mode)
            {
                case AspectMode.None:
                    scale = new Vector2(1, 1);
                    translate = new Vector2(0, 0);
                    return;
                case AspectMode.Fit:
                    if (srcAspect > dstAspect)
                    {
                        float s = srcAspect / dstAspect;
                        scale = new Vector2(1, s);
                        translate = new Vector2(0, (1 - s) / 2);
                        return;
                    }
                    else
                    {
                        float s = dstAspect / srcAspect;
                        scale = new Vector2(s, 1);
                        translate = new Vector2((1 - s) / 2, 0);
                        return;
                    }
                case AspectMode.Fill:
                    if (srcAspect > dstAspect)
                    {
                        float s = dstAspect / srcAspect;
                        scale = new Vector2(s, 1);
                        translate = new Vector2((1 - s) / 2, 0);
                        return;
                    }
                    else
                    {
                        float s = srcAspect / dstAspect;
                        scale = new Vector2(1, s);
                        translate = new Vector2(0, (1 - s) / 2);
                        return;
                    }
                default:
                    throw new Exception("Unknown aspect mode");
            }
        }

    }
}
