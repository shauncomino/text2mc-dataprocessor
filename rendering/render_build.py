import bpy
import os
import h5py
import numpy as np
import warnings
import math
warnings.filterwarnings("ignore", category=UserWarning, module='numpy')

# Specify the folder containing the .h5 files
h5_folder = '/home/shaun/projects/text2mc-dataprocessor/test_builds'  # Replace with your folder path

# Output folder for rendered images
output_folder = '/home/shaun/projects/text2mc-dataprocessor/rendering/output'  # Replace with your desired output path

# Block size
block_size = 1  # You can adjust the size of each block

# Air block token
AIR_TOKEN = 102  # The integer representing air blocks

# Function to clear the scene
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

# Function to create voxel mesh from block data
def create_voxel_mesh(block_data):
    # Create a new mesh
    mesh = bpy.data.meshes.new('VoxelMesh')
    obj = bpy.data.objects.new('VoxelObject', mesh)
    bpy.context.collection.objects.link(obj)

    vertices = []
    faces = []
    vertex_index = 0

    # Get the non-air block positions
    positions = np.argwhere(block_data != AIR_TOKEN)
    print(f"Number of non-air blocks: {len(positions)}")  # Debugging to check non-air blocks

    for pos in positions:
        x, y, z = pos * block_size
        # Define the 8 vertices of the cube
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
        # Define the 6 faces of the cube
        face = [
            (vertex_index, vertex_index + 1, vertex_index + 2, vertex_index + 3),
            (vertex_index + 4, vertex_index + 5, vertex_index + 6, vertex_index + 7),
            (vertex_index, vertex_index + 1, vertex_index + 5, vertex_index + 4),
            (vertex_index + 1, vertex_index + 2, vertex_index + 6, vertex_index + 5),
            (vertex_index + 2, vertex_index + 3, vertex_index + 7, vertex_index + 6),
            (vertex_index + 3, vertex_index, vertex_index + 4, vertex_index + 7),
        ]
        vertices.extend(verts)
        faces.extend(face)
        vertex_index += 8

    # Create the mesh
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    # Optionally, set a material or color
    mat = bpy.data.materials.new(name="VoxelMaterial")
    mat.use_nodes = True  # Enable node-based material for Cycles
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.8, 0.1, 0.1, 1)  # Red color

    obj.data.materials.append(mat)

    # Ensure object transformations are updated
    bpy.context.view_layer.update()

    # Ensure the object is visible in the render
    obj.hide_render = False

    return obj

# Function to calculate the camera position and distance
def calculate_camera_position(grid_shape, fov_deg):
    # Calculate the dimensions of the grid
    width, height, depth = np.array(grid_shape) * block_size

    # Determine the camera distance based on the largest dimension (diagonal)
    diagonal = math.sqrt(width**2 + depth**2)
    fov_rad = math.radians(fov_deg)
    distance = (diagonal / 2) / math.tan(fov_rad / 2)

    # Return the calculated camera position and the center of the grid
    return distance, (width / 2, height / 2, depth / 2)

# Function to set up the scene (camera and lighting)
def setup_scene(grid_shape):
    # Set the field of view in degrees
    fov_deg = 50

    # Calculate the necessary camera distance and grid center
    distance, grid_center = calculate_camera_position(grid_shape, fov_deg)
    center_x, center_y, center_z = grid_center

    # Set up the camera
    cam_data = bpy.data.cameras.new('Camera')
    cam_obj = bpy.data.objects.new('Camera', cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # Position the camera based on the calculated distance
    cam_obj.location = (center_x, center_y + distance, center_z + distance)

    # Rotate the camera to face the voxel grid correctly
    cam_obj.rotation_euler[0] = math.radians(-45)  # Tilt down 45 degrees
    cam_obj.rotation_euler[1] = 0
    cam_obj.rotation_euler[2] = math.radians(0)  # Face directly along the Y-axis

    # Set the camera's field of view
    cam_data.angle = math.radians(fov_deg)

    # Set up lighting
    light_data = bpy.data.lights.new(name="Light", type='SUN')
    light_obj = bpy.data.objects.new(name="Light", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (50, -50, 150)
    light_data.energy = 10

# Main function to process .h5 files
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

        # Create voxel mesh
        create_voxel_mesh(block_data)

        # Get the shape of the voxel grid for camera positioning
        grid_shape = block_data.shape

        # Set up the scene based on voxel grid size
        setup_scene(grid_shape)

        # Render settings
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'  # Use GPU if available

        # Set output path
        output_path = os.path.join(output_folder, os.path.splitext(h5_file)[0] + '.png')
        bpy.context.scene.render.filepath = output_path

        # Set render resolution
        bpy.context.scene.render.resolution_x = 512
        bpy.context.scene.render.resolution_y = 512

        # Render the image
        bpy.ops.render.render(write_still=True)
        print(f"Rendered {output_path}")

# Entry point for the script
if __name__ == "__main__":
    process_h5_files()
