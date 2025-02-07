using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

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

        /// <summary>
        /// Set Span to GraphicsBuffer without copying
        /// </summary>
        /// <param name="buffer">A GraphicsBuffer</param>
        /// <param name="span">The span data to be set</param>
        /// <typeparam name="T">The type of data</typeparam>
        public unsafe static void SetData<T>(this GraphicsBuffer buffer, ReadOnlySpan<T> span) where T : unmanaged
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            var handle = AtomicSafetyHandle.Create();
#endif // ENABLE_UNITY_COLLECTIONS_CHECKS
            fixed (void* ptr = span)
            {
                var arr = NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<T>(ptr, span.Length, Allocator.None);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref arr, handle);
#endif // ENABLE_UNITY_COLLECTIONS_CHECKS
                buffer.SetData(arr);
            }
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            AtomicSafetyHandle.Release(handle);
#endif // ENABLE_UNITY_COLLECTIONS_CHECKS
        }
    }
}
