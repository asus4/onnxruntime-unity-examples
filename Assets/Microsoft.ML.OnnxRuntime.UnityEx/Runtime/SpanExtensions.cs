using System;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Burst.CompilerServices;
using Unity.Mathematics;
using UnityEngine.Assertions;

namespace Microsoft.ML.OnnxRuntime.UnityEx
{
    /// <summary>
    /// Experimental:
    /// Accesses Multi-Dimensional Tensor Value in 1D-Span
    /// </summary>
    [BurstCompile]
    public static class SpanExtensions
    {
        public static Span3D<T> AsSpan3D<T>(this ref Span<T> span, int3 shape) where T : unmanaged
        {
            return new Span3D<T>(span, shape);
        }

        public static ReadOnlySpan3D<T> AsSpan3D<T>(this ref ReadOnlySpan<T> span, int3 shape) where T : unmanaged
        {
            return new ReadOnlySpan3D<T>(span, shape);
        }

        public unsafe static void Transpose<T>(ReadOnlySpan<T> input, Span<T> output, int2 outShape) where T : unmanaged
        {
            int length = outShape.x * outShape.y;
            Assert.AreEqual(length, input.Length);
            Assert.AreEqual(length, output.Length);

            fixed (void* inputPtr = input)
            fixed (void* outputPtr = output)
            {
                Transpose<T>(inputPtr, outputPtr, outShape.x, outShape.y);
            }
        }

        [BurstCompile]
        static unsafe void Transpose<T>(
            [NoAlias] void* input,
            [NoAlias] void* output,
            int outW, int outH) where T : unmanaged
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

    /// <summary>
    /// Experimental:
    /// Allows Span<T> to access like 3D multi-dimensional array
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public readonly ref struct Span3D<T> where T : unmanaged
    {
        private readonly Span<T> span;
        private readonly int3 shape;

        public Span3D(Span<T> span, int3 shape)
        {
            this.span = span;
            this.shape = shape;
        }

        public T this[int x, int y, int z]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => span[(x * shape.y + y) * shape.z + z];
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => span[(x * shape.y + y) * shape.z + z] = value;
        }

        public static implicit operator ReadOnlySpan3D<T>(Span3D<T> span) => new(span.span, span.shape);
        public static implicit operator Span<T>(Span3D<T> span) => span.span;
    }

    /// <summary>
    /// Experimental:
    /// Allows ReadOnlySpan<T> to access like 3D multi-dimensional array
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public readonly ref struct ReadOnlySpan3D<T> where T : unmanaged
    {
        private readonly ReadOnlySpan<T> span;
        private readonly int3 shape;

        public ReadOnlySpan3D(ReadOnlySpan<T> span, int3 shape)
        {
            this.span = span;
            this.shape = shape;
        }

        public T this[int x, int y, int z]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => span[(x * shape.y + y) * shape.z + z];
        }

        public static implicit operator ReadOnlySpan<T>(ReadOnlySpan3D<T> span) => span.span;
    }
}
