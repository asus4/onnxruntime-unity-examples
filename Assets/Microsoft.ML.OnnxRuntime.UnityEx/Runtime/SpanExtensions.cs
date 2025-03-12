using System;
using Unity.Mathematics;

namespace Microsoft.ML.OnnxRuntime.UnityEx
{
    /// <summary>
    /// Experimental Span Extensions
    /// </summary>
    public static class SpanExtensions
    {
        public static Span2D<T> AsSpan2D<T>(this ref Span<T> span, int2 shape) where T : unmanaged
            => new(span, shape);
        public static ReadOnlySpan2D<T> AsSpan2D<T>(this ref ReadOnlySpan<T> span, int2 shape) where T : unmanaged
            => new(span, shape);
        public static Span3D<T> AsSpan3D<T>(this ref Span<T> span, int3 shape) where T : unmanaged
            => new(span, shape);
        public static ReadOnlySpan3D<T> AsSpan3D<T>(this ref ReadOnlySpan<T> span, int3 shape) where T : unmanaged
            => new(span, shape);
    }
}
