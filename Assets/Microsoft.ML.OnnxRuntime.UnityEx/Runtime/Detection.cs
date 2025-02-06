using System;
using UnityEngine;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Microsoft.ML.OnnxRuntime.Unity;

namespace Microsoft.ML.OnnxRuntime.UnityEx
{
    /// <summary>
    /// Interface for Detection Task
    /// </summary>
    /// <typeparam name="T">Detection struct</typeparam>
    public interface IDetection<T> : IComparable<T>
        where T : unmanaged
    {
        int Label { get; }
        Rect Rect { get; }

        /// <summary>
        /// Non-Maximum Suppression (Multi-Class)
        /// </summary>
        /// <param name="proposals">A list of proposals which should be sorted in descending order</param>
        /// <param name="result">A result of NMS</param>
        /// <param name="iouThreshold">A threshold of IoU (Intersection over Union)</param>
        /// <typeparam name="U">Detection</typeparam>
        public unsafe static void NMS<U>(
            NativeSlice<U> proposals,
            NativeList<U> result,
            float iouThreshold)
            where U : unmanaged, IDetection<U>
        {
            result.Clear();

            int proposalsLength = proposals.Length;
            U* proposalsPtr = (U*)proposals.GetUnsafeReadOnlyPtr();

            for (int i = 0; i < proposalsLength; i++)
            {
                U* a = proposalsPtr + i;
                bool keep = true;

                for (int j = 0; j < result.Length; j++)
                {
                    U* b = (U*)result.GetUnsafeReadOnlyPtr() + j;

                    // Ignore different classes
                    if (b->Label != a->Label)
                    {
                        continue;
                    }

                    float iou = a->Rect.IntersectionOverUnion(b->Rect);
                    if (iou > iouThreshold)
                    {
                        keep = false;
                        break;
                    }
                }

                if (keep)
                {
                    result.Add(*a);
                }
            }
        }
    }
}
