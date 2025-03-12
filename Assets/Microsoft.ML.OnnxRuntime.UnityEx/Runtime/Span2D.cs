using System;
using System.Runtime.CompilerServices;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine.Assertions;

namespace Microsoft.ML.OnnxRuntime.UnityEx
{
    /// <summary> 
    /// Allows Span<T> to access like 2D multi-dimensional array
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public readonly ref struct Span2D<T> where T : unmanaged
    {
        public readonly Span<T> span;
        public readonly int2 shape;

        public Span2D(Span<T> span, int2 shape)
        {
            Assert.AreEqual(span.Length, shape.x * shape.y);
            this.span = span;
            this.shape = shape;
        }

        public T this[int x, int y]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => span[x * shape.y + y];
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => span[x * shape.y + y] = value;
        }

        public Span<T> this[int x]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => span.Slice(x * shape.y, shape.y);
        }

        public static implicit operator ReadOnlySpan2D<T>(Span2D<T> span) => new(span.span, span.shape);

        public JobHandle ScheduleTransposeJob(Span2D<T> output)
        {
            ReadOnlySpan<T> readOnlySpan = span;
            return readOnlySpan.ScheduleTransposeJob(output.span, shape.y, shape.x);
        }
    }

    /// <summary>
    /// Allows ReadOnlySpan<T> to access like 2D multi-dimensional array
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public readonly ref struct ReadOnlySpan2D<T> where T : unmanaged
    {
        public readonly ReadOnlySpan<T> span;
        public readonly int2 shape;

        public ReadOnlySpan2D(ReadOnlySpan<T> span, int2 shape)
        {
            Assert.AreEqual(span.Length, shape.x * shape.y);
            this.span = span;
            this.shape = shape;
        }

        public T this[int x, int y]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => span[x * shape.y + y];
        }

        public ReadOnlySpan<T> this[int x]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => span.Slice(x * shape.y, shape.y);
        }

        public unsafe JobHandle ScheduleTransposeJob(Span2D<T> output)
        {
            return span.ScheduleTransposeJob(output.span, shape.y, shape.x);
        }
    }
}
