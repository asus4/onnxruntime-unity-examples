using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime.Unity;
using UnityEngine;
using UnityEngine.Assertions;

namespace Microsoft.ML.OnnxRuntime.Examples
{
    /// <summary>
    /// RTMPose: Real-Time Multi-Person Pose Estimation
    /// Licensed under Apache-2.0.
    /// See LICENSE for full license information.
    /// https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
    ///
    /// The included model is downloaded from the following link:
    /// https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip
    ///
    /// Then fixed the dynamic batch size to 1 and converted using Runtime optimization:
    /// python -m onnxruntime.tools.make_dynamic_shape_fixed --dim_param batch --dim_value 1 end2end.onnx rtmpose-t_body7_256x192.onnx
    /// python -m onnxruntime.tools.convert_onnx_models_to_ort rtmpose-t_body7_256x192.onnx --optimization_style Runtime
    /// </summary>
    public sealed class RTMPose : ImageInference<float>
    {
        [Serializable]
        public class Options : ImageInferenceOptions
        {
            [Header("RTMPose options")]
            [Range(1, 10)]
            public int maxPoses = 5;
            [Range(0f, 1f)]
            public float keypointThreshold = 0.3f;
            public float bboxPadding = 1.25f;
        }

        public readonly struct Keypoint
        {
            /// <summary>
            /// Position in the viewport space, 0-1 in Y-up coordinates
            /// </summary>
            public readonly Vector2 position;
            public readonly float score;

            public Keypoint(Vector2 position, float score)
            {
                this.position = position;
                this.score = score;
            }
        }

        public class Pose
        {
            public Rect rect;
            public float probability;
            public readonly Keypoint[] keypoints = new Keypoint[KeypointCount];
        }

        /// <summary>
        /// COCO 17 keypoints:
        /// 0:nose, 1:left_eye, 2:right_eye, 3:left_ear, 4:right_ear,
        /// 5:left_shoulder, 6:right_shoulder, 7:left_elbow, 8:right_elbow,
        /// 9:left_wrist, 10:right_wrist, 11:left_hip, 12:right_hip,
        /// 13:left_knee, 14:right_knee, 15:left_ankle, 16:right_ankle
        /// </summary>
        public const int KeypointCount = 17;

        // Each axis is split into (input size * ratio) bins in SimCC
        private const float SIMCC_SPLIT_RATIO = 2f;

        private readonly Options options;
        private readonly Keypoint[] keypoints = new Keypoint[KeypointCount];
        private Matrix4x4 cropMatrix = Matrix4x4.identity;

        public ReadOnlySpan<Keypoint> Keypoints => keypoints;

        public RTMPose(byte[] model, Options options)
            : base(model, options)
        {
            this.options = options;
        }

        /// <summary>
        /// Run pose estimation for a single person cropped by GetCropRect().
        /// Returns keypoints in the viewport space.
        /// </summary>
        public ReadOnlySpan<Keypoint> Run(Texture texture, in Rect cropRect)
        {
            // Maps model input UV to the crop area of the source texture
            cropMatrix = Matrix4x4.Translate(cropRect.min)
                * Matrix4x4.Scale(new Vector3(cropRect.width, cropRect.height, 1f));
            Run(texture);
            return keypoints;
        }

        /// <summary>
        /// Convert a detected person rect to a crop rect for the pose model,
        /// following MMPose's TopDownGetBboxCenterScale:
        /// expand the box to the model input aspect ratio, then add padding.
        /// </summary>
        public Rect GetCropRect(in Rect viewportRect, float texWidth, float texHeight)
        {
            float inputAspect = (float)Width / Height;

            // Aspect ratio needs to be calculated in the source pixel space
            Vector2 center = new(viewportRect.center.x * texWidth, viewportRect.center.y * texHeight);
            float w = viewportRect.width * texWidth;
            float h = viewportRect.height * texHeight;
            if (w > h * inputAspect)
            {
                h = w / inputAspect;
            }
            else
            {
                w = h * inputAspect;
            }
            w *= options.bboxPadding;
            h *= options.bboxPadding;

            return new Rect(
                (center.x - w * 0.5f) / texWidth,
                (center.y - h * 0.5f) / texHeight,
                w / texWidth,
                h / texHeight);
        }

        protected override void PreProcess(Texture texture)
        {
            // Crop the person area instead of resizing the whole texture
            var tensorData = textureToTensor.Transform(texture, cropMatrix);
            tensorData.CopyTo(inputs[0].GetTensorMutableDataAsSpan<float>());
        }

        protected override void PostProcess(IReadOnlyList<OrtValue> outputs)
        {
            var simccX = outputs[0].GetTensorDataAsSpan<float>();
            var simccY = outputs[1].GetTensorDataAsSpan<float>();

            int xBins = simccX.Length / KeypointCount;
            int yBins = simccY.Length / KeypointCount;
            Assert.AreEqual(Width * SIMCC_SPLIT_RATIO, xBins);
            Assert.AreEqual(Height * SIMCC_SPLIT_RATIO, yBins);

            for (int k = 0; k < KeypointCount; k++)
            {
                int xIdx = ArgMax(simccX.Slice(k * xBins, xBins), out float xMax);
                int yIdx = ArgMax(simccY.Slice(k * yBins, yBins), out float yMax);

                // Pixel position in the crop space, CV coordinates (Y-down)
                float px = xIdx / SIMCC_SPLIT_RATIO;
                float py = yIdx / SIMCC_SPLIT_RATIO;

                // Convert to the viewport space via the crop matrix (Y-up)
                Vector2 uv = new(px / Width, 1f - py / Height);
                keypoints[k] = new Keypoint(
                    cropMatrix.MultiplyPoint3x4(uv),
                    Mathf.Min(xMax, yMax));
            }
        }

        private static int ArgMax(in ReadOnlySpan<float> span, out float maxValue)
        {
            int maxIndex = 0;
            maxValue = float.MinValue;
            for (int i = 0; i < span.Length; i++)
            {
                if (span[i] > maxValue)
                {
                    maxValue = span[i];
                    maxIndex = i;
                }
            }
            return maxIndex;
        }
    }
}
