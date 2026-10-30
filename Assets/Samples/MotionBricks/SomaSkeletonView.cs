using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Microsoft.ML.OnnxRuntime.Examples
{
    /// <summary>
    /// Draws a SOMA pose as a capsule mannequin. Heel and toe spheres show the predicted foot contacts.
    /// </summary>
    public sealed class SomaSkeletonView : MonoBehaviour
    {
        [SerializeField]
        Material material;

        [SerializeField]
        Mesh capsule;

        [SerializeField]
        Mesh sphere;

        [SerializeField]
        bool showFootContacts = true;

        // Bones end at the joint and start at its parent. Radius in meters.
        static readonly (int joint, float radius)[] Bones =
        {
            (1, 0.11f), (2, 0.115f), (3, 0.12f), (4, 0.115f), (5, 0.05f), (6, 0.05f), // spine, neck
            (10, 0.05f), (11, 0.05f), (12, 0.05f), (13, 0.04f), (15, 0.035f), // left arm
            (16, 0.05f), (17, 0.05f), (18, 0.05f), (19, 0.04f), (21, 0.035f), // right arm
            (22, 0.08f), (23, 0.07f), (24, 0.05f), (25, 0.04f), // left leg
            (26, 0.08f), (27, 0.07f), (28, 0.05f), (29, 0.04f), // right leg
        };
        static readonly int[] ContactJoints =
        {
            SomaSkeleton.LeftFoot, SomaSkeleton.LeftToeBase, SomaSkeleton.RightFoot, SomaSkeleton.RightToeBase,
        };
        static readonly int BaseColor = Shader.PropertyToID("_BaseColor");

        readonly float3[] positions = new float3[SomaSkeleton.JointCount];
        readonly quaternion[] rotations = new quaternion[SomaSkeleton.JointCount];
        readonly MaterialPropertyBlock[] contactColors = new MaterialPropertyBlock[4];

        /// <summary>SOMA model space is right-handed and Unity is left-handed. Both are Y up.</summary>
        public static Vector3 ToUnity(float3 p) => new(p.x, p.y, -p.z);

        public static float3 ToSoma(Vector3 p) => new(p.x, p.y, -p.z);

        /// <summary>Draws the pose in this frame.</summary>
        public void Draw(SomaPose pose, float4 footContacts)
        {
            pose.ComputeWorld(positions, rotations);
            var renderParams = new RenderParams(material) { shadowCastingMode = ShadowCastingMode.On };
            Matrix4x4 toWorld = transform.localToWorldMatrix;

            foreach (var (joint, radius) in Bones)
            {
                Vector3 a = ToUnity(positions[SomaSkeleton.Parents[joint]]);
                Vector3 b = ToUnity(positions[joint]);
                var scale = new Vector3(radius * 2, Vector3.Distance(a, b) * 0.5f, radius * 2);
                var bone = Matrix4x4.TRS((a + b) * 0.5f, Quaternion.FromToRotation(Vector3.up, b - a), scale);
                Graphics.RenderMesh(renderParams, capsule, 0, toWorld * bone);
            }

            // The head sphere sits between the head joint and the eyes.
            float3 eyes = (positions[SomaSkeleton.LeftEye] + positions[SomaSkeleton.RightEye]) * 0.5f;
            Vector3 head = ToUnity(math.lerp(positions[SomaSkeleton.Head], eyes, 0.5f));
            Graphics.RenderMesh(renderParams, sphere, 0, toWorld * Matrix4x4.TRS(head, Quaternion.identity, Vector3.one * 0.22f));

            if (!showFootContacts)
            {
                return;
            }
            for (int i = 0; i < ContactJoints.Length; i++)
            {
                contactColors[i] ??= new MaterialPropertyBlock();
                contactColors[i].SetColor(BaseColor, Color.Lerp(Color.white, new Color(1f, 0.45f, 0.1f), footContacts[i]));
                renderParams.matProps = contactColors[i];
                Vector3 contact = ToUnity(positions[ContactJoints[i]]);
                Graphics.RenderMesh(renderParams, sphere, 0, toWorld * Matrix4x4.TRS(contact, Quaternion.identity, Vector3.one * 0.07f));
            }
        }
    }
}
