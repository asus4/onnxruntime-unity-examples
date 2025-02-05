using System;
using System.Runtime.CompilerServices;
using Unity.Mathematics;
using UnityEngine.Assertions;

namespace Microsoft.ML.OnnxRuntime.UnityEx
{
    /// <summary>
    /// Experimental:
    /// Allows Span<T> to access like 3D multi-dimensional array
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public readonly ref struct Span3D<T> where T : unmanaged
    {
        public readonly Span<T> span;
        public readonly int3 shape;

        public Span3D(Span<T> span, int3 shape)
        {
            Assert.AreEqual(span.Length, shape.x * shape.y * shape.z);
            this.span = span;
            this.shape = shape;
        }

        public T this[int x, int y, int z]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => span[(x * shape.y + y) * shape.z + z];
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => span[(x * shape.y + y) * shape.z + z] = value;
        }

        public Span2D<T> this[int x]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => new(span.Slice(x * shape.y * shape.z, shape.y * shape.z), shape.yz);
        }

        public static implicit operator ReadOnlySpan3D<T>(Span3D<T> span) => new(span.span, span.shape);
    }


    /// <summary>
    /// Experimental:
    /// Allows ReadOnlySpan<T> to access like 3D multi-dimensional array
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public readonly ref struct ReadOnlySpan3D<T> where T : unmanaged
    {
        public readonly ReadOnlySpan<T> span;
        public readonly int3 shape;

        public ReadOnlySpan3D(ReadOnlySpan<T> span, int3 shape)
        {
            Assert.AreEqual(span.Length, shape.x * shape.y * shape.z);
            this.span = span;
            this.shape = shape;
        }

        public T this[int x, int y, int z]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => span[(x * shape.y + y) * shape.z + z];
        }

        public ReadOnlySpan2D<T> this[int x]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => new(span.Slice(x * shape.y * shape.z, shape.y * shape.z), shape.yz);
        }
    }
}
