using System;
using Microsoft.ML.OnnxRuntime.Unity;
using Microsoft.ML.OnnxRuntime.UnityEx;
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
        private readonly GraphicsBuffer segmentationBuffer;
        private readonly GraphicsBuffer detectionBuffer;
        private readonly NativeArray<float> maskData;
        private readonly GraphicsBuffer maskBuffer;
        private readonly GraphicsBuffer colorTableBuffer;
        private readonly RenderTexture texture;

        public readonly int3 shape;
        public Texture Texture => texture;

        private static readonly int _DetectionCount = Shader.PropertyToID("_DetectionCount");
        private static readonly int _MaskThreshold = Shader.PropertyToID("_MaskThreshold");

        private readonly Yolo11Seg.Options options;

        public Yolo11SegVisualize(int3 shape, Color[] colors, Yolo11Seg.Options options)
        {
            this.shape = shape;
            this.options = options;
            this.compute = options.visualizeSegmentationShader;
            int maxCount = options.maxDetectionCount;

            // Segmentation Buffer
            {
                int count = shape.x * shape.y * shape.z;
                segmentationBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, count, sizeof(float));
            }

            // Mask Buffer
            {
                detectionBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, maxCount, UnsafeUtility.SizeOf<Yolo11Seg.Detection>());

                maskData = new NativeArray<float>(maxCount * 32, Allocator.Persistent);
                maskBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, maxCount, sizeof(float) * 32);
            }

            // Fill Color Table
            {
                const int LABEL_COUNT = 80;
                colorTableBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, LABEL_COUNT, UnsafeUtility.SizeOf<Color>());
                var colorTable = new Color[LABEL_COUNT];
                for (int i = 0; i < colorTable.Length; i++)
                {
                    colorTable[i] = colors[i % colors.Length];
                }
                colorTableBuffer.SetData(colorTable);
            }

            int width = shape.z;
            int height = shape.y;
            texture = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32)
            {
                enableRandomWrite = true
            };
            texture.Create();

            kernel = compute.FindKernel("SegmentationToTexture");
            compute.SetBuffer(kernel, "_SegmentationBuffer", segmentationBuffer);
            compute.SetBuffer(kernel, "_DetectionBuffer", detectionBuffer);
            compute.SetBuffer(kernel, "_MaskBuffer", maskBuffer);
            compute.SetBuffer(kernel, "_ColorTable", colorTableBuffer);
            compute.SetTexture(kernel, "_OutputTexture", texture);
            compute.SetInts("_OutputSize", new int[] { width, height });
            compute.SetFloat(_MaskThreshold, options.maskThreshold);
        }

        public void Dispose()
        {
            segmentationBuffer.Release();
            maskData.Dispose();
            maskBuffer.Release();
            detectionBuffer.Release();
            colorTableBuffer.Release();
            texture.Release();
            UnityEngine.Object.Destroy(texture);
        }

        public void Process(
            ReadOnlySpan2D<float> output0Transposed,
            ReadOnlySpan<float> output1,
            NativeArray<Yolo11Seg.Detection>.ReadOnly detections)
        {
            const int MASK_SIZE = 32;
            int count = Math.Min(maskBuffer.count, detections.Length);
            var detectionSpan = detections.AsReadOnlySpan()[..count];

            // Prepare mask data
            {
                var maskSpan = maskData.AsSpan();

                // Copy each detection mask
                for (int i = 0; i < detectionSpan.Length; i++)
                {
                    var detection = detectionSpan[i];
                    // Mask: 32 from the end
                    var mask = output0Transposed[detection.anchorId][^MASK_SIZE..];
                    mask.CopyTo(maskSpan.Slice(i * MASK_SIZE, MASK_SIZE));
                }
            }

            // Set data to buffer
            segmentationBuffer.SetData(output1);
            maskBuffer.SetData(maskData, 0, 0, count * MASK_SIZE);
            detectionBuffer.SetData(detectionSpan);
            compute.SetInt(_DetectionCount, count);
            compute.SetFloat(_MaskThreshold, options.maskThreshold);

            // Run compute shader
            compute.Dispatch(kernel, texture.width / 8, texture.height / 8, 1);
        }
    }
}
