using System;
using UnityEngine;
using Unity.Collections;
using Microsoft.ML.OnnxRuntime.Unity;

namespace Microsoft.ML.OnnxRuntime.UnityEx
{
    public interface IDetection<T> : IComparable<T>
        where T : unmanaged
    {
        int Label { get; }
        Rect Rect { get; }

        public static void NMS<U>(
            NativeSlice<U> proposals,
            NativeList<U> result,
            float iouThreshold)
            where U : unmanaged, IDetection<U>
        {
            result.Clear();

            foreach (U a in proposals)
            {
                bool keep = true;
                foreach (U b in result)
                {
                    // Ignore different classes
                    if (b.Label != a.Label)
                    {
                        continue;
                    }
                    float iou = a.Rect.IntersectionOverUnion(b.Rect);
                    if (iou > iouThreshold)
                    {
                        keep = false;
                    }
                }

                if (keep)
                {
                    result.Add(a);
                }
            }
        }
    }
}
