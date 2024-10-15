import bpy
import os
import h5py
import numpy as np
import warnings
import math
import glob
import json
from mathutils import Vector, Matrix

warnings.filterwarnings("ignore", category=UserWarning, module='numpy')

# Specify the folder containing the .h5 files
h5_folder = '/home/shaun/projects/text2mc-dataprocessor/test_builds'  # Update this path

# Output folder for rendered images
output_folder = '/home/shaun/projects/text2mc-dataprocessor/rendering/output'  # Update this path

# Path to the tok2block.json file
tok2block_path = '/home/shaun/projects/text2mc-dataprocessor/world2vec/tok2block.json'  # Update this path

# Path to the Minecraft texture pack block textures
texture_folder = '/home/shaun/projects/text2mc-dataprocessor/rendering/VanillaDefault 1.21/assets/minecraft/textures/block'  # Update this path

# Block size
block_size = 1  # Adjust if necessary

# Air block token
AIR_TOKEN = 102  # The integer representing air blocks

# Load the token to block mapping
with open(tok2block_path, 'r') as f:
    tok2block = json.load(f)

# Build a mapping from texture names to file paths
texture_files = glob.glob(os.path.join(texture_folder, '*.png'))
texture_name_to_path = {}
for texture_file in texture_files:
    texture_name = os.path.splitext(os.path.basename(texture_file))[0]  # e.g., 'acacia_log'
    texture_name_to_path[texture_name] = texture_file

def get_texture_paths(block_name):
    # Remove 'minecraft:' prefix
    if block_name.startswith('minecraft:'):
        block_name = block_name[len('minecraft:'):]
    textures = {
        'top': None,
        'bottom': None,
        'side': None,
        'all': None
    }
    # Check for specific textures
    if block_name + '_top' in texture_name_to_path:
        textures['top'] = texture_name_to_path[block_name + '_top']
    if block_name + '_bottom' in texture_name_to_path:
        textures['bottom'] = texture_name_to_path[block_name + '_bottom']
    if block_name + '_side' in texture_name_to_path:
        textures['side'] = texture_name_to_path[block_name + '_side']
    if block_name in texture_name_to_path:
        textures['all'] = texture_name_to_path[block_name]
    return textures

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def create_voxel_mesh(block_data):
    # Create a new mesh and object
    mesh = bpy.data.meshes.new('VoxelMesh')
    obj = bpy.data.objects.new('VoxelObject', mesh)
    bpy.context.collection.objects.link(obj)

    vertices = []
    faces = []
    uvs = []
    material_indices = []
    vertex_index = 0

    # Get positions and block types of non-air blocks
    positions = np.argwhere(block_data != AIR_TOKEN)
    block_types = block_data[block_data != AIR_TOKEN]

    print(f"Number of non-air blocks: {len(positions)}")

    # Create materials dictionary
    materials_dict = {}
    for block_token in np.unique(block_types):
        block_token_str = str(block_token)
        block_name = tok2block.get(block_token_str, 'minecraft:unknown')
        texture_paths = get_texture_paths(block_name)
        if not any(texture_paths.values()):
            continue  # Skip if no texture found
        materials_dict[block_token_str] = {}
        for face_type in ['top', 'bottom', 'side']:
            texture_path = texture_paths.get(face_type)
            if texture_path is None and texture_paths['all'] is not None:
                texture_path = texture_paths['all']
            if texture_path is not None:
                # Create material
                mat = bpy.data.materials.new(name=f"Material_{block_name}_{face_type}")
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links
                bsdf = nodes.get('Principled BSDF')
                
                # Create texture node
                tex_image_node = nodes.new('ShaderNodeTexImage')
                try:
                    tex_image_node.image = bpy.data.images.load(texture_path)
                    tex_image_node.interpolation = 'Closest'  # Prevent blurring
                    tex_image_node.extension = 'CLIP'  # Prevent tiling beyond texture size
                except:
                    print(f"Failed to load texture for {block_name}_{face_type}")
                    continue
                # Connect the texture node to the base color of the BSDF shader
                links.new(bsdf.inputs['Base Color'], tex_image_node.outputs['Color'])
                materials_dict[block_token_str][face_type] = mat
    if not materials_dict:
        print("No materials were created. Check your textures and tok2block.json mappings.")
        return obj

    # Assign materials to the object
    for block_materials in materials_dict.values():
        for mat in block_materials.values():
            if mat.name not in obj.data.materials:
                obj.data.materials.append(mat)

    # Build the mesh
    for i, pos in enumerate(positions):
        x, y, z = pos * block_size
        block_token = block_types[i]
        block_token_str = str(block_token)
        block_materials = materials_dict.get(block_token_str)
        if block_materials is None:
            continue  # No materials found for this block type

        # Define vertices of the cube
        verts = [
            (x, y, z),
            (x + block_size, y, z),
            (x + block_size, y + block_size, z),
            (x, y + block_size, z),
            (x, y, z + block_size),
            (x + block_size, y, z + block_size),
            (x + block_size, y + block_size, z + block_size),
            (x, y + block_size, z + block_size),
        ]
        vertices.extend(verts)

        # Define faces of the cube with face types
        cube_faces = [
            ((vertex_index, vertex_index + 1, vertex_index + 2, vertex_index + 3), 'bottom'),  # Bottom
            ((vertex_index + 4, vertex_index + 5, vertex_index + 6, vertex_index + 7), 'top'),  # Top
            ((vertex_index + 3, vertex_index + 2, vertex_index + 6, vertex_index + 7), 'side'),  # Front
            ((vertex_index + 1, vertex_index + 0, vertex_index + 4, vertex_index + 5), 'side'),  # Back
            ((vertex_index + 0, vertex_index + 3, vertex_index + 7, vertex_index + 4), 'side'),  # Left
            ((vertex_index + 2, vertex_index + 1, vertex_index + 5, vertex_index + 6), 'side'),  # Right
        ]
        for face_vertices, face_type in cube_faces:
            # Get the material for this face type
            material = block_materials.get(face_type)
            if material is None:
                material = block_materials.get('all')
            if material is None:
                continue  # No material for this face, skip

            # Only append the face if we have a valid material
            faces.append(face_vertices)

            material_index = obj.data.materials.find(material.name)
            if material_index == -1:
                # Material not found in object's materials, add it
                obj.data.materials.append(material)
                material_index = obj.data.materials.find(material.name)
            material_indices.append(material_index)
            # Assign UVs
            uv_face = [(0, 0), (1, 0), (1, 1), (0, 1)]
            uvs.append(uv_face)
        vertex_index += 8

    # Create the mesh
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    # Assign materials to faces
    if len(material_indices) != len(mesh.polygons):
        print("Warning: Number of material indices does not match number of polygons.")
    for i, poly in enumerate(mesh.polygons):
        if i < len(material_indices):
            poly.material_index = material_indices[i]
        else:
            poly.material_index = 0  # Assign default material if out of range

    # Create UV map
    if not mesh.uv_layers:
        uv_layer = mesh.uv_layers.new(name='UVMap')
    else:
        uv_layer = mesh.uv_layers.active

    # Assign UV coordinates to each face
    for i, poly in enumerate(mesh.polygons):
        if i >= len(uvs):
            continue
        for j in range(poly.loop_total):
            loop_index = poly.loop_start + j
            uv_layer.data[loop_index].uv = uvs[i][j]

    # Recalculate normals to ensure they are facing outward
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Ensure object transformations are updated
    bpy.context.view_layer.update()

    return obj

def calculate_camera_position(grid_shape, fov_deg):
    # Calculate dimensions of the grid
    width, height, depth = np.array(grid_shape) * block_size

    # Determine the center of the grid
    center_x = width / 2
    center_y = height / 2
    center_z = depth / 2

    # Determine the maximum dimension
    max_dim = max(width, height, depth)

    # Calculate the distance from the center to fit the entire build in view
    fov_rad = math.radians(fov_deg)
    distance = (max_dim / 2) / math.tan(fov_rad / 2)

    return distance, Vector((center_x, center_y, center_z))

def setup_scene(grid_shape):
    # Set the field of view
    fov_deg = 50

    # Calculate camera distance and grid center
    distance, grid_center = calculate_camera_position(grid_shape, fov_deg)

    # Set up the camera
    cam_data = bpy.data.cameras.new('Camera')
    cam_obj = bpy.data.objects.new('Camera', cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # Position the camera diagonally above and in front of the build
    initial_cam_location = grid_center + Vector((distance, -distance, distance))
    cam_obj.location = initial_cam_location

    # Rotate the camera position horizontally by 90 degrees to the right
    rotation_angle = math.radians(90)
    rot_matrix = Matrix.Rotation(rotation_angle, 4, 'Z')
    relative_cam_loc = cam_obj.location - grid_center
    new_relative_cam_loc = rot_matrix @ relative_cam_loc
    cam_obj.location = grid_center + new_relative_cam_loc

    # Point the camera at the grid center
    direction = grid_center - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

    # Set camera field of view
    cam_data.angle = math.radians(fov_deg)

    # Remove existing lights
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)

    # Set up lighting directly above the center of the build
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_obj = bpy.data.objects.new(name="Sun", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = grid_center + Vector((0, 150, 100))  # Directly above

    light_obj.rotation_euler = (
        math.radians(-60),  # Pointing straight down
        0,
        0,
    )
    light_data.energy = 10  # Adjust energy as needed

    # Optional: Adjust world background color
    bpy.context.scene.world.use_nodes = True
    world_nodes = bpy.context.scene.world.node_tree.nodes
    bg_node = world_nodes.get('Background')
    if bg_node:
        bg_node.inputs[0].default_value = (1, 1, 1, 1)  # White background
        bg_node.inputs[1].default_value = 0.1  # Strength

    # Set background to transparent
    bpy.context.scene.render.film_transparent = True

    # Set output format to PNG with RGBA
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'

def process_h5_files():
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all .h5 files in the folder
    h5_files = [f for f in os.listdir(h5_folder) if f.endswith('.h5')]

    for h5_file in h5_files:
        print(f"Processing {h5_file}...")
        clear_scene()

        # Load block data from .h5 file
        h5_path = os.path.join(h5_folder, h5_file)
        with h5py.File(h5_path, 'r') as hf:
            build_folder_in_hdf5 = list(hf.keys())[0]
            block_data = hf[build_folder_in_hdf5][()]
            # Transpose to match Blender's coordinate system
            block_data = np.transpose(block_data, (0, 2, 1))  # Adjusted transpose

        # Create voxel mesh
        obj = create_voxel_mesh(block_data)

        # Get grid shape
        grid_shape = block_data.shape

        # Set up the scene
        setup_scene(grid_shape)

        # Render settings
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'  # Use GPU if available

        # Set output path
        output_path = os.path.join(output_folder, os.path.splitext(os.path.basename(h5_file))[0] + '.png')
        bpy.context.scene.render.filepath = output_path

        # Set square render resolution
        bpy.context.scene.render.resolution_x = 6000
        bpy.context.scene.render.resolution_y = 6000

        # Increase samples for better quality
        bpy.context.scene.cycles.samples = 256  # Increase as needed

        # Enable denoising
        bpy.context.scene.cycles.use_denoising = True

        # Render the image
        bpy.ops.render.render(write_still=True)
        print(f"Rendered {output_path}")

# Entry point
if __name__ == "__main__":
    process_h5_files()
