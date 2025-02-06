using System;
using Microsoft.ML.OnnxRuntime.UnityEx;
using UnityEngine;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine.Assertions;

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
        private readonly GraphicsBuffer segmentationBuffer;
        private readonly NativeArray<float> maskData;
        private readonly GraphicsBuffer maskBuffer;
        private readonly GraphicsBuffer colorTableBuffer;
        private readonly RenderTexture texture;

        public Texture Texture => texture;

        private static readonly int _MaskCount = Shader.PropertyToID("_MaskCount");

        public Yolo11SegVisualize(int3 shape, ComputeShader compute, Color[] colors, int maxCount)
        {
            this.compute = compute;
            // Segmentation Buffer
            {
                int count = shape.x * shape.y * shape.z;
                segmentationBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, count, sizeof(float));
            }

            // Mask Buffer
            {
                maskData = new NativeArray<float>(maxCount * 32, Allocator.Persistent);
                maskBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, maxCount, sizeof(float) * 32);
            }

            // Fill Color Table
            {
                colorTableBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, maxCount, UnsafeUtility.SizeOf<Color>());
                var colorTable = new Color[maxCount];
                for (int i = 0; i < colorTable.Length; i++)
                {
                    colorTable[i] = colors[i % colors.Length];
                }
                colorTableBuffer.SetData(colorTable);
            }

            int width = shape.y;
            int height = shape.z;
            texture = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32)
            {
                enableRandomWrite = true
            };
            texture.Create();

            kernel = compute.FindKernel("SegmentationToTexture");
            compute.SetBuffer(kernel, "_SegmentationBuffer", segmentationBuffer);
            compute.SetBuffer(kernel, "_MaskBuffer", maskBuffer);
            compute.SetBuffer(kernel, "_ColorTable", colorTableBuffer);
            compute.SetTexture(kernel, "_OutputTexture", texture);
            compute.SetInts("_OutputSize", new int[] { width, height });
        }

        public void Dispose()
        {
            segmentationBuffer.Release();
            maskData.Dispose();
            maskBuffer.Release();
            colorTableBuffer.Release();
            texture.Release();
            UnityEngine.Object.Destroy(texture);
        }

        public void Process(
            // 1(batch), 8400(anchor), 116(data)
            NativeArray<float> output0Transposed,
            Span<float> output1,
            NativeArray<Yolo11Seg.Detection>.ReadOnly detections)
        {
            segmentationBuffer.SetData(output1);

            // Prepare mask buffer
            {
                var output0Span = output0Transposed.AsReadOnlySpan();
                var output0Tensor = output0Span.AsSpan2D(new int2(8400, 116));
                int count = Math.Min(maskBuffer.count, detections.Length);
                var detectionSpan = detections.AsReadOnlySpan()[..count];
                var destination = maskData.AsSpan();
                const int MASK_SIZE = 32;

                // Copy each detection's mask to buffer
                for (int i = 0; i < detectionSpan.Length; i++)
                {
                    var detection = detectionSpan[i];
                    var mask = output0Tensor[detection.anchorId][^MASK_SIZE..];

                    mask.CopyTo(destination.Slice(i * MASK_SIZE, MASK_SIZE));
                }
                maskBuffer.SetData(maskData, 0, 0, count * MASK_SIZE);
                compute.SetInt(_MaskCount, count);
            }

            // Run compute shader
            compute.Dispatch(kernel, texture.width / 8, texture.height / 8, 1);
        }
    }
}
