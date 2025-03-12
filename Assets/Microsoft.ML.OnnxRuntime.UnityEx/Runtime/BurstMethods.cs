using System;
using Unity.Burst;
using Unity.Burst.CompilerServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using UnityEngine.Assertions;

namespace Microsoft.ML.OnnxRuntime.UnityEx
{
    [BurstCompile]
    public static class BurstMethods
    {
        public static unsafe JobHandle ScheduleTransposeJob<T>(this ReadOnlySpan<T> input, Span<T> output, int outW, int outH)
            where T : unmanaged
        {
            Assert.AreEqual(outW * outH, input.Length);
            Assert.AreEqual(outW * outH, output.Length);

            fixed (T* inputPtr = input)
            fixed (T* outputPtr = output)
            {
                // Transpose<T>(inputPtr, outputPtr, outW, outH);
                var job = new TransposeJobCore<T>
                {
                    input = inputPtr,
                    output = outputPtr,
                    outW = outW,
                    outH = outH
                };
                return job.Schedule(input.Length, 64);
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        static unsafe void Transpose<T>([NoAlias] T* input, [NoAlias] T* output, int outW, int outH)
            where T : unmanaged
        {
            for (int y = 0; y < outW; y++)
            {
                for (int x = 0; x < outH; x++)
                {
                    output[y * outH + x] = input[x * outW + y];
                }
            }
        }

        [BurstCompile]
        unsafe struct TransposeJobCore<T> : IJobParallelFor
            where T : unmanaged
        {
            [ReadOnly]
            [NativeDisableUnsafePtrRestriction]
            public T* input;
            [WriteOnly]
            [NativeDisableUnsafePtrRestriction]
            public T* output;
            public int outW;
            public int outH;

            public readonly void Execute(int index)
            {
                int y = index / outH;
                int x = index % outH;
                output[y * outH + x] = input[x * outW + y];
            }
        }

        public unsafe static int ArgMax(this ReadOnlySpan<float> values)
        {
            fixed (float* valuesPtr = values)
            {
                ArgMax(valuesPtr, values.Length, out int maxIndex);
                return maxIndex;
            }
        }

        // [BurstCompile(CompileSynchronously = true)]
        static unsafe void ArgMax(float* values, [AssumeRange(1, int.MaxValue)] int length, out int maxIndex)
        {
            int maxIdx = -1;
            float maxValue = float.MinValue;

            for (int i = 0; i < length; i++)
            {
                if (values[i] > maxValue)
                {
                    maxValue = values[i];
                    maxIdx = i;
                }
            }
            maxIndex = maxIdx;
        }
    }
}
