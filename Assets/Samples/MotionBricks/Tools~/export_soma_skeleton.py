"""Export the 30-joint MotionBricks SOMA skeleton and an idle pose as SomaSkeleton.cs.

The skeleton comes from the USD character in https://github.com/NVlabs/motionbricks
(Apache-2.0): motionbricks/assets/human/soma_base_skel_minimal_no_root.usd.
The joint order comes from the model manifest: motionbricks/assets/onnx/soma/soma_n1.json.

The rest pose of the USD is a T-pose. The idle pose lowers the arms and bends the elbows a
little, so it looks like a person standing still.

Usage (from a clone of NVlabs/motionbricks, after `git lfs pull`):
    uv run --with usd-core --with numpy python export_soma_skeleton.py \
        <motionbricks>/motionbricks/assets/human/soma_base_skel_minimal_no_root.usd \
        <motionbricks>/motionbricks/assets/onnx/soma/soma_n1.json \
        ../SomaSkeleton.cs
"""

import json
import sys

import numpy as np
from pxr import Usd, UsdGeom, UsdSkel

# Degrees. The rest pose faces +Z with the left arm along +X.
ARM_DOWN = 76.0
ELBOW_BEND = 14.0


def quat_from_matrix(m):
    """Column-vector rotation matrix to a quaternion (x, y, z, w) with w >= 0."""
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q) / np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def axis_angle_matrix(axis, degrees):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.radians(degrees)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * (k @ k)


def load_skeleton(usd_path, joint_names):
    """Return the parents and the rest world transforms of the model joints, in meters."""
    stage = Usd.Stage.Open(usd_path)
    meters = UsdGeom.GetStageMetersPerUnit(stage)
    skel = next(UsdSkel.Skeleton(p) for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton))
    paths = [str(p) for p in skel.GetJointsAttr().Get()]
    rest = skel.GetRestTransformsAttr().Get()
    index = {p: i for i, p in enumerate(paths)}

    world_r, world_t = [None] * len(paths), [None] * len(paths)
    for i, path in enumerate(paths):
        m = np.array(rest[i], dtype=np.float64)  # USD matrices use the row-vector convention
        r, t = m[:3, :3].T, m[3, :3] * meters
        parent = path.rsplit("/", 1)[0] if "/" in path else None
        if parent is None:
            world_r[i], world_t[i] = r, t
        else:
            p = index[parent]
            world_r[i] = world_r[p] @ r
            world_t[i] = world_r[p] @ t + world_t[p]

    names = [p.rsplit("/", 1)[-1] for p in paths]
    usd_index = [names.index(n) for n in joint_names]

    def model_parent(j):
        # The nearest ancestor that is also a model joint. Finger joints are skipped.
        path = paths[usd_index[j]]
        while "/" in path:
            path = path.rsplit("/", 1)[0]
            name = path.rsplit("/", 1)[-1]
            if name in joint_names:
                return joint_names.index(name)
        return -1

    parents = [model_parent(j) for j in range(len(joint_names))]
    return parents, [world_r[i] for i in usd_index], [world_t[i] for i in usd_index]


def author_idle(joint_names, parents, rest_r, rest_t):
    idle_r = [r.copy() for r in rest_r]
    idle_t = [t.copy() for t in rest_t]

    def rotate_subtree(root_name, rot):
        root = joint_names.index(root_name)
        pivot = idle_t[root].copy()
        for j in range(len(joint_names)):
            k = j
            while k != -1 and k != root:
                k = parents[k]
            if k == root:
                idle_r[j] = rot @ idle_r[j]
                idle_t[j] = rot @ (idle_t[j] - pivot) + pivot

    rotate_subtree("LeftArm", axis_angle_matrix([0, 0, 1], -ARM_DOWN))
    rotate_subtree("RightArm", axis_angle_matrix([0, 0, 1], ARM_DOWN))
    rotate_subtree("LeftForeArm", axis_angle_matrix([1, 0, 0], -ELBOW_BEND))
    rotate_subtree("RightForeArm", axis_angle_matrix([1, 0, 0], -ELBOW_BEND))
    return idle_r, idle_t


def to_local(parents, world_r, world_t):
    rotations, positions = [], []
    for j, p in enumerate(parents):
        if p < 0:
            r, t = world_r[j], world_t[j]
        else:
            r = world_r[p].T @ world_r[j]
            t = world_r[p].T @ (world_t[j] - world_t[p])
        rotations.append(quat_from_matrix(r))
        positions.append(t)
    return rotations, positions


def f(v):
    return f"{v:.6f}f"


def emit(joint_names, parents, positions, rotations):
    lines = []
    w = lines.append
    w("// <auto-generated>")
    w("// Generated by Tools~/export_soma_skeleton.py from NVlabs/motionbricks (Apache-2.0).")
    w("// </auto-generated>")
    w("using Unity.Mathematics;")
    w("")
    w("namespace Microsoft.ML.OnnxRuntime.Examples")
    w("{")
    w("    /// <summary>")
    w("    /// The 30-joint SOMA skeleton of the MotionBricks SOMA model.")
    w("    /// SOMA model space is Y up, in meters and right-handed. The rest pose faces +Z.")
    w("    /// </summary>")
    w("    public static class SomaSkeleton")
    w("    {")
    w(f"        public const int JointCount = {len(joint_names)};")
    for name in ("Head", "LeftEye", "RightEye", "LeftLeg", "RightLeg",
                 "LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"):
        w(f"        public const int {name} = {joint_names.index(name)};")
    w("")
    w("        public static readonly string[] JointNames =")
    w("        {")
    for n in joint_names:
        w(f"            \"{n}\",")
    w("        };")
    w("")
    w("        /// <summary>Parent joint index, or -1 for the root (Hips).</summary>")
    w("        public static readonly int[] Parents =")
    w("        {")
    w("            " + ", ".join(str(p) for p in parents) + ",")
    w("        };")
    w("")
    w("        /// <summary>")
    w("        /// Offset of each joint from its parent, in the parent's space.")
    w("        /// Joint 0 holds the hips position of the idle pose.")
    w("        /// </summary>")
    w("        public static readonly float3[] LocalPositions =")
    w("        {")
    for n, t in zip(joint_names, positions):
        w(f"            new({f(t[0])}, {f(t[1])}, {f(t[2])}), // {n}")
    w("        };")
    w("")
    w("        /// <summary>")
    w("        /// Standing idle pose, made from the rest T-pose by lowering the arms.")
    w("        /// Joint 0 is the world rotation of the hips. The others are relative to their parent.")
    w("        /// </summary>")
    w("        public static readonly quaternion[] IdleLocalRotations =")
    w("        {")
    for n, q in zip(joint_names, rotations):
        w(f"            new({f(q[0])}, {f(q[1])}, {f(q[2])}, {f(q[3])}), // {n}")
    w("        };")
    w("    }")
    w("}")
    return "\n".join(lines) + "\n"


def main(usd_path, manifest_path, out_path):
    joint_names = json.load(open(manifest_path))["joint_names"]
    parents, rest_r, rest_t = load_skeleton(usd_path, joint_names)
    idle_r, idle_t = author_idle(joint_names, parents, rest_r, rest_t)
    rotations, positions = to_local(parents, idle_r, idle_t)
    with open(out_path, "w", newline="\n") as fp:
        fp.write(emit(joint_names, parents, positions, rotations))
    print("wrote", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:4])
