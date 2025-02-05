using System;
using UnityEngine;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Microsoft.ML.OnnxRuntime.Examples
{
    /// <summary>
    /// Licensed under AGPL-3.0 license
    /// See LICENSE for full license information.
    /// https://github.com/ultralytics/ultralytics/blob/main/LICENSE
    /// 
    /// https://docs.ultralytics.com/tasks/segment/
    /// 
    /// Ported from python code: 
    /// https://github.com/ultralytics/ultralytics/blob/aecf1da32b50e144b226f54ae4e979dc3cc62145/examples/YOLOv8-Segmentation-ONNXRuntime-Python/main.py
    /// </summary>
    internal sealed class Yolo11SegVisualize : IDisposable
    {
        private readonly int kernel;
        private readonly ComputeShader compute;
        private readonly NativeArray<float> tensorNativeArray;
        private readonly GraphicsBuffer tensorBuffer;
        private readonly GraphicsBuffer colorTableBuffer;
        private readonly RenderTexture texture;

        public Texture Texture => texture;

        public Yolo11SegVisualize(int3 shape, ComputeShader compute, Color[] colors)
        {
            this.compute = compute;

            int count = shape.x * shape.y * shape.z;
            tensorNativeArray = new NativeArray<float>(count, Allocator.Persistent);
            tensorBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, count, sizeof(float));
            int masks = shape.x;
            int width = shape.y;
            int height = shape.z;

            colorTableBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, masks, UnsafeUtility.SizeOf<Color>());
            var colorTable = new Color[masks];
            for (int i = 0; i < colorTable.Length; i++)
            {
                colorTable[i] = colors[i % colors.Length];
            }
            colorTableBuffer.SetData(colorTable);
            texture = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32)
            {
                enableRandomWrite = true
            };
            texture.Create();

            kernel = compute.FindKernel("SegmentationToTexture");
            compute.SetBuffer(kernel, "_InputBuffer", tensorBuffer);
            compute.SetBuffer(kernel, "_ColorTable", colorTableBuffer);
            compute.SetTexture(kernel, "_OutputTexture", texture);
            compute.SetInts("_OutputSize", new int[] { width, height });
        }

        public void Dispose()
        {
            tensorNativeArray.Dispose();
            tensorBuffer.Release();
            colorTableBuffer.Release();
            texture.Release();
            UnityEngine.Object.Destroy(texture);
        }

        public void Process(ReadOnlySpan<float> tensor)
        {
            // Copy tensor to buffer
            tensor.CopyTo(tensorNativeArray.AsSpan());
            tensorBuffer.SetData(tensorNativeArray);

            // Run compute shader
            compute.Dispatch(kernel, texture.width / 8, texture.height / 8, 1);
        }
    }
}
