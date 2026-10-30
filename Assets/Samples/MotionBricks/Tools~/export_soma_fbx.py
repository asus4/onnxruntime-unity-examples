"""Convert the SOMA body model to SOMA.fbx for Unity.

The source is the SOMA template rig from https://github.com/NVlabs/SOMA-X (Apache-2.0):
assets/SOMA_template_rig.usda. This keeps its low-detail skin mesh (c_skin_lo, 4.5k vertices)
without UVs, and the skeleton without its twist bones, in the bind pose (an A-pose), in meters.

Usage (from a clone of NVlabs/SOMA-X, after `git lfs pull`):
    blender --background --python export_soma_fbx.py -- <SOMA-X>/assets/SOMA_template_rig.usda <out.fbx>
"""

import sys

import bpy

usd_path, fbx_path = sys.argv[sys.argv.index("--") + 1:][:2]

bpy.ops.wm.read_factory_settings(use_empty=True)
# The low-detail mesh is invisible in the USD, so import it by its path.
bpy.ops.wm.usd_import(
    filepath=usd_path,
    import_cameras=False,
    import_lights=False,
    create_world_material=False,
    import_visible_only=False,
    prim_path_mask="/OUTPUT/c_skeleton_grp;/OUTPUT/c_geometry_grp/MainMesh/Meshes/c_skin_lo",
)
scene = bpy.context.scene
armature = scene.objects["Root"]
mesh = scene.objects["c_skin_lo"]

# Bake the parent transform (centimeters to meters, Y up to Z up) into the armature and the mesh.
for obj in (armature, mesh):
    matrix = obj.matrix_world.copy()
    obj.parent = None
    obj.matrix_world = matrix
bpy.ops.object.select_all(action="DESELECT")
for obj in (armature, mesh):
    obj.select_set(True)
bpy.context.view_layer.objects.active = armature
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
mesh.parent = armature
mesh.matrix_parent_inverse = armature.matrix_world.inverted()

# The importer poses the skeleton in the USD rest pose (a T-pose). Keep the bind pose instead.
for bone in armature.pose.bones:
    bone.location = (0, 0, 0)
    bone.rotation_quaternion = (1, 0, 0, 0)
    bone.rotation_euler = (0, 0, 0)
    bone.scale = (1, 1, 1)

# Remove the empty parents, and the UV sets that the sample does not use.
for obj in [o for o in scene.objects if o.type == "EMPTY"]:
    bpy.data.objects.remove(obj, do_unlink=True)
for uv in list(mesh.data.uv_layers):
    mesh.data.uv_layers.remove(uv)

# Move the weights of the twist bones to their parent limb bone, then delete bones without weights.
# The twist bones are not animated, so the mesh deforms the same way with fewer bones.
groups = mesh.vertex_groups
for group in [g for g in groups if "Twist" in g.name]:
    parent = armature.data.bones[group.name].parent.name
    target = groups.get(parent) or groups.new(name=parent)
    for vertex in mesh.data.vertices:
        for g in vertex.groups:
            if g.group == group.index:
                target.add([vertex.index], g.weight, "ADD")
    groups.remove(group)
bpy.context.view_layer.objects.active = armature
bpy.ops.object.mode_set(mode="EDIT")
unused = ("Root", "HeadEnd", "LeftToeEnd", "RightToeEnd")
for name in [b.name for b in armature.data.edit_bones if "Twist" in b.name or b.name in unused]:
    armature.data.edit_bones.remove(armature.data.edit_bones[name])
bpy.ops.object.mode_set(mode="OBJECT")

armature.name = "SOMA"
mesh.name = "SOMA_Body"

bpy.ops.object.select_all(action="DESELECT")
armature.select_set(True)
mesh.select_set(True)
bpy.ops.export_scene.fbx(
    filepath=fbx_path,
    use_selection=True,
    object_types={"ARMATURE", "MESH"},
    apply_scale_options="FBX_SCALE_ALL",
    axis_forward="-Z",
    axis_up="Y",
    bake_space_transform=True,
    add_leaf_bones=False,
    bake_anim=False,
    mesh_smooth_type="FACE",
)
print("wrote", fbx_path)
