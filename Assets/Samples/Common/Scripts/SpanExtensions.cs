using System;
using System.Runtime.CompilerServices;
using Unity.Mathematics;

namespace Microsoft.ML.OnnxRuntime.Unity
{
    /// <summary>
    /// Experimental:
    /// Accesses Multi-Dimensional Tensor Value in 1D-Span
    /// </summary>
    public static class SpanExtensions
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T GetValue<T>(this ref ReadOnlySpan<T> span, int x, int y, int2 shape) where T : struct
        {
            return span[y * shape.x + x];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T GetValue<T>(this ref ReadOnlySpan<T> span, int x, int y, int z, int3 shape) where T : struct
        {
            return span[(z * shape.y + y) * shape.x + x];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SetValue<T>(this ref Span<T> span, int x, int y, int2 shape, T value) where T : struct
        {
            span[y * shape.x + x] = value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SetValue<T>(this ref Span<T> span, int x, int y, int z, int3 shape, T value) where T : struct
        {
            span[(z * shape.y + y) * shape.x + x] = value;
        }
    }
}
