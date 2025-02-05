using System;
using Unity.Burst;
using UnityEngine.Assertions;

namespace Microsoft.ML.OnnxRuntime.UnityEx
{
    public static class BurstMethods
    {
        public static unsafe void Transpose<T>(ReadOnlySpan<T> input, Span<T> output, int outW, int outH)
            where T : unmanaged
        {
            Assert.AreEqual(outW * outH, input.Length);
            Assert.AreEqual(outW * outH, output.Length);

            fixed (void* inputPtr = input)
            fixed (void* outputPtr = output)
            {
                Transpose<T>(inputPtr, outputPtr, outW, outH);
            }
        }

        [BurstCompile]
        static unsafe void Transpose<T>([NoAlias] in void* input, [NoAlias] in void* output, int outW, int outH)
            where T : unmanaged
        {
            for (int y = 0; y < outW; y++)
            {
                for (int x = 0; x < outH; x++)
                {
                    ((T*)output)[y * outH + x] = ((T*)input)[x * outW + y];
                }
            }
        }
    }
}
