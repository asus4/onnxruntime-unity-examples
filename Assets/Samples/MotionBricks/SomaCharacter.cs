using Unity.Mathematics;
using UnityEngine;

namespace Microsoft.ML.OnnxRuntime.Examples
{
    /// <summary>
    /// Poses the humanoid SOMA character with SOMA poses.
    /// Each bone turns by the rotation of its joint relative to the bind pose of the SOMA mesh.
    /// </summary>
    [RequireComponent(typeof(Animator))]
    public sealed class SomaCharacter : MonoBehaviour
    {
        // The humanoid bone of each SOMA joint. LastBone: not a humanoid bone.
        static readonly HumanBodyBones[] HumanBones =
        {
            HumanBodyBones.Hips, HumanBodyBones.Spine, HumanBodyBones.Chest, HumanBodyBones.UpperChest,
            HumanBodyBones.Neck, HumanBodyBones.LastBone, HumanBodyBones.Head, HumanBodyBones.Jaw,
            HumanBodyBones.LeftEye, HumanBodyBones.RightEye,
            HumanBodyBones.LeftShoulder, HumanBodyBones.LeftUpperArm, HumanBodyBones.LeftLowerArm, HumanBodyBones.LeftHand,
            HumanBodyBones.LastBone, HumanBodyBones.LastBone,
            HumanBodyBones.RightShoulder, HumanBodyBones.RightUpperArm, HumanBodyBones.RightLowerArm, HumanBodyBones.RightHand,
            HumanBodyBones.LastBone, HumanBodyBones.LastBone,
            HumanBodyBones.LeftUpperLeg, HumanBodyBones.LeftLowerLeg, HumanBodyBones.LeftFoot, HumanBodyBones.LeftToes,
            HumanBodyBones.RightUpperLeg, HumanBodyBones.RightLowerLeg, HumanBodyBones.RightFoot, HumanBodyBones.RightToes,
        };

        readonly Transform[] bones = new Transform[SomaSkeleton.JointCount];
        readonly Quaternion[] bindRotations = new Quaternion[SomaSkeleton.JointCount];
        readonly float3[] positions = new float3[SomaSkeleton.JointCount];
        readonly quaternion[] rotations = new quaternion[SomaSkeleton.JointCount];

        /// <summary>
        /// SOMA model space is right-handed and Unity is left-handed. Both are Y up.
        /// Mirror X, like the FBX importer does for the character.
        /// </summary>
        public static Vector3 ToUnity(float3 p) => new(-p.x, p.y, p.z);

        public static float3 ToSoma(Vector3 p) => new(-p.x, p.y, p.z);

        static Quaternion ToUnity(quaternion q) => new(q.value.x, -q.value.y, -q.value.z, q.value.w);

        void Awake()
        {
            // The character starts in its bind pose.
            var animator = GetComponent<Animator>();
            for (int j = 0; j < SomaSkeleton.JointCount; j++)
            {
                if (HumanBones[j] != HumanBodyBones.LastBone)
                {
                    bones[j] = animator.GetBoneTransform(HumanBones[j]);
                    bindRotations[j] = Quaternion.Inverse(transform.rotation) * bones[j].rotation;
                }
            }
        }

        public void SetPose(SomaPose pose)
        {
            pose.ComputeWorld(positions, rotations);
            for (int j = 0; j < SomaSkeleton.JointCount; j++)
            {
                if (bones[j] != null)
                {
                    quaternion fromBind = math.mul(rotations[j], math.inverse(SomaSkeleton.BindRotations[j]));
                    bones[j].rotation = transform.rotation * ToUnity(fromBind) * bindRotations[j];
                }
            }
            bones[0].position = transform.TransformPoint(ToUnity(positions[0]));
        }
    }
}
