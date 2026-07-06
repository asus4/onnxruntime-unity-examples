using Microsoft.ML.OnnxRuntime.Examples;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Draws COCO 17 keypoint skeletons as a single UI mesh.
/// </summary>
public class PoseVisualizer : MaskableGraphic
{
    [SerializeField]
    private float keypointRadius = 8f;

    [SerializeField]
    private float boneWidth = 4f;

    [SerializeField]
    private Color leftColor = new(0.2f, 0.6f, 1f, 1f);

    [SerializeField]
    private Color rightColor = new(1f, 0.5f, 0.2f, 1f);

    [SerializeField]
    private Color centerColor = new(0.2f, 1f, 0.4f, 1f);

    // Pairs of COCO 17 keypoint indices
    private static readonly int[] Bones =
    {
        15, 13, 13, 11, 16, 14, 14, 12, 11, 12,
        5, 11, 6, 12, 5, 6, 5, 7, 6, 8,
        7, 9, 8, 10, 1, 2, 0, 1, 0, 2,
        1, 3, 2, 4, 3, 5, 4, 6,
    };

    private RTMPose.Pose[] poses;
    private int poseCount;
    private float scoreThreshold = 0.3f;

    public void SetPoses(RTMPose.Pose[] poses, int count, float threshold)
    {
        this.poses = poses;
        poseCount = count;
        scoreThreshold = threshold;
        SetVerticesDirty();
    }

    protected override void OnPopulateMesh(VertexHelper vh)
    {
        vh.Clear();
        if (poses == null)
        {
            return;
        }

        Rect area = GetPixelAdjustedRect();

        for (int i = 0; i < poseCount; i++)
        {
            var keypoints = poses[i].keypoints;

            for (int b = 0; b < Bones.Length; b += 2)
            {
                var kpA = keypoints[Bones[b]];
                var kpB = keypoints[Bones[b + 1]];
                if (kpA.score < scoreThreshold || kpB.score < scoreThreshold)
                {
                    continue;
                }
                AddLine(vh,
                    ToLocalPosition(kpA.position, area),
                    ToLocalPosition(kpB.position, area),
                    GetColor(Bones[b], Bones[b + 1]));
            }

            for (int k = 0; k < keypoints.Length; k++)
            {
                var kp = keypoints[k];
                if (kp.score < scoreThreshold)
                {
                    continue;
                }
                AddDot(vh, ToLocalPosition(kp.position, area), GetColor(k, k));
            }
        }
    }

    private static Vector2 ToLocalPosition(in Vector2 viewportPosition, in Rect area)
    {
        return area.min + viewportPosition * area.size;
    }

    private Color GetColor(int a, int b)
    {
        // Left: odd indices, Right: even indices, except 0:nose
        bool isLeftA = a % 2 == 1;
        bool isLeftB = b % 2 == 1;
        if (a == 0 || b == 0 || isLeftA != isLeftB)
        {
            return centerColor;
        }
        return isLeftA ? leftColor : rightColor;
    }

    private void AddLine(VertexHelper vh, in Vector2 a, in Vector2 b, in Color color)
    {
        Vector2 dir = (b - a).normalized;
        Vector2 perp = new Vector2(-dir.y, dir.x) * (boneWidth * 0.5f);

        int index = vh.currentVertCount;
        vh.AddVert(a - perp, color, Vector4.zero);
        vh.AddVert(a + perp, color, Vector4.zero);
        vh.AddVert(b + perp, color, Vector4.zero);
        vh.AddVert(b - perp, color, Vector4.zero);
        vh.AddTriangle(index, index + 1, index + 2);
        vh.AddTriangle(index, index + 2, index + 3);
    }

    private void AddDot(VertexHelper vh, in Vector2 p, in Color color)
    {
        float r = keypointRadius * 0.5f;
        int index = vh.currentVertCount;
        vh.AddVert(p + new Vector2(-r, -r), color, Vector4.zero);
        vh.AddVert(p + new Vector2(-r, r), color, Vector4.zero);
        vh.AddVert(p + new Vector2(r, r), color, Vector4.zero);
        vh.AddVert(p + new Vector2(r, -r), color, Vector4.zero);
        vh.AddTriangle(index, index + 1, index + 2);
        vh.AddTriangle(index, index + 2, index + 3);
    }
}
