import anvil
import os
import mcschematic
from typing import List
import sys
from sklearn.cluster import DBSCAN
from typing import Union
import numpy as np
import json
import h5py
import math
import glob

# Now you can use mcschematic


# Class to parse data files and vectorize them into information the model can train on
class World2Vec:

    # Converts old blocks to new ones
    @staticmethod
    def convert_if_old(block) -> anvil.Block:
        if isinstance(block, anvil.OldBlock):
            try:
                block = anvil.OldBlock.convert(block)
            except:
                return None
        return block

    # Finds the subdirectory containing region files
    def find_regions_dir(directory: str) -> list:
        folders_with_mca = []
        # Walk through all directories and files in the given directory
        for root, dirs, files in os.walk(directory):
            # Skip Nether and End dimension folders
            if "DIM-1" in root or "DIM1" in root or "poi" in root:
                continue
            # Check if any file ends with .mca in the current directory
            if any(file.endswith(".mca") for file in files):
                # If so, add the current directory to the list
                folders_with_mca.append(root)
        return folders_with_mca

    @staticmethod
    def find_inhabited_time_exists(dir: str) -> bool:
        for filename in os.listdir(dir):
            if filename.endswith(".mca"):
                # Retrieve the region
                region = anvil.Region.from_file(os.path.join(dir, filename))
                # Only search the region file if it is not empty (because apparently sometimes they are empty?)
                if region.data:
                    # Set search sections
                    # Retrieve each chunk in the region
                    for x in range(0, 32):
                        for z in range(0, 32):
                            # Region files need not contain 32x32 chunks, so we must check if the chunk exists
                            chunk_data = region.chunk_data(x, z)
                            if chunk_data:
                                # Calculate the time the chunk has been inhabited
                                if (
                                    "Level" in chunk_data
                                    and "InhabitedTime" in chunk_data["Level"]
                                ):
                                    inhabited_time = (
                                        chunk_data["Level"]["InhabitedTime"].value / 20
                                    )
                                elif "InhabitedTime" in chunk_data:
                                    inhabited_time = (
                                        chunk_data["InhabitedTime"].value / 20
                                    )
                                else:
                                    return False

                                if inhabited_time > 0:
                                    return True
        return False

    @staticmethod
    def find_surface_section(
        chunk: anvil.Chunk, low_section: int, high_section: int, superflat: bool
    ) -> Union[bool, int]:
        surface_section = None
        for s in range(high_section, low_section, -1):
            good_section = False
            superflat_void = False
            air_count = 0
            section = anvil.Chunk.get_section(chunk, s)
            for block in anvil.Chunk.stream_blocks(chunk, section=section):
                block = World2Vec.convert_if_old(block)
                if block != None and anvil.Block.name(block) == "minecraft:air":
                    air_count += 1
                    # We'll check for a section to have a good portion of air, testing says 1024 blocks is a good fit
                    if surface_section is not None and air_count == 1024:
                        surface_section = section
                        good_section = True

                    if (
                        surface_section is not None
                        and air_count == 4096
                        and s <= low_section + 4
                    ):
                        surface_section = anvil.Chunk.get_section(chunk, s + 1)
                        superflat_void = True
                        superflat = True
                        break
            if surface_section is None and air_count != 4096:
                surface_section = section
            elif superflat_void:
                return superflat, s + 1
            elif surface_section is not None and not good_section and not superflat:
                return superflat, s + 1
        return True, low_section + 1

    # Reads all region files in dir and returns a Generator of Chunks, all of which contain blocks that are not in natural_blocks.txt
    def get_build(
        dir: str, save_dir: str, build_name: str, natural_blocks_path: str = None
    ) -> List[str]:
        print("Searching directories: " + dir)  # Read in the natural blocks to an array
        if natural_blocks_path is None:
            nb_file = open("natural_blocks.txt", "r")
        else:
            nb_file = open(natural_blocks_path, "r")
        natural_blocks = nb_file.read().splitlines()
        nb_file.close()
        # This is the list of all the build chunks
        build_chunks = []
        relevant_regions = []

        # Flag for superflat worlds
        superflat = False
        superflat_markers = []
        # Iterate through .mca files in dir
        inhabited_time_exist = World2Vec.find_inhabited_time_exists(dir)
        inhabited_time_check = 0
        if inhabited_time_exist:
            inhabited_time_check = 1.5
        for filename in os.listdir(dir):
            if filename.endswith(".mca"):
                # print("Now in:" +  filename + "\n")
                # Retrieve the region
                region = anvil.Region.from_file(os.path.join(dir, filename))
                # Only search the region file if it is not empty (because apparently sometimes they are empty?)
                if region.data:
                    # Set search sections
                    inhabited_time_exist = True
                    search_sections = range(15, -5, -1)
                    # Retrieve each chunk in the region
                    for x in range(0, 32):
                        for z in range(0, 32):
                            # Region files need not contain 32x32 chunks, so we must check if the chunk exists
                            chunk_data = region.chunk_data(x, z)
                            if chunk_data:
                                try:
                                    chunk = anvil.Region.get_chunk(region, x, z)
                                    if chunk.version and chunk.version < 2844:
                                        search_sections = range(15, -1, -1)
                                except Exception as e:
                                    print(
                                        "Error: Could not read chunk",
                                        x,
                                        z,
                                        "in",
                                        filename,
                                        "due to",
                                        e,
                                    )
                                    return
                                # Calculate the time the chunk has been inhabited
                                if (
                                    "Level" in chunk_data
                                    and "InhabitedTime" in chunk_data["Level"]
                                ):
                                    inhabited_time = (
                                        chunk_data["Level"]["InhabitedTime"].value / 20
                                    )
                                elif "InhabitedTime" in chunk_data:
                                    inhabited_time = (
                                        chunk_data["InhabitedTime"].value / 20
                                    )
                                else:
                                    inhabited_time_exist = False
                                # Check whether the chunk has been visited at all, if not we can skip checking it
                                if (
                                    inhabited_time_exist == False
                                    or inhabited_time >= inhabited_time_check
                                ):

                                    superflat_mode = False
                                    if len(superflat_markers) != 0:
                                        superflat_mode = max(
                                            set(superflat_markers),
                                            key=superflat_markers.count,
                                        )

                                    superflat, surface_section = (
                                        World2Vec.find_surface_section(
                                            chunk,
                                            search_sections.stop,
                                            search_sections.start,
                                            superflat_mode,
                                        )
                                    )

                                    superflat_markers.append(superflat)
                                    # Search the relevant sections
                                    chunk_added = False
                                    for s in range(
                                        surface_section, surface_section + 4
                                    ):
                                        section = anvil.Chunk.get_section(chunk, s)
                                        # Check each block in the section
                                        for block in anvil.Chunk.stream_blocks(
                                            chunk, section=section
                                        ):
                                            block = World2Vec.convert_if_old(block)
                                            if block != None and not anvil.Block.name(
                                                block
                                            ).startswith("minecraft"):
                                                # This is a modded block, and we should skip the build
                                                print(
                                                    "Modded block found in build. Skipping..."
                                                )
                                                return
                                            # If it's not a natural block, add this chunk to the list
                                            if (
                                                block != None
                                                and anvil.Block.name(block)
                                                not in natural_blocks
                                            ):
                                                print(anvil.Block.name(block) + "at " + str(chunk.x) + ", " + str(chunk.z) + " added to build!")
                                                build_chunks.append(chunk)
                                                if filename not in relevant_regions:
                                                    region_x = int(
                                                        filename.split("r.")[1].split(
                                                            "."
                                                        )[0]
                                                    )
                                                    region_z = int(
                                                        filename.split("r.")[1].split(
                                                            "."
                                                        )[1]
                                                    )
                                                    if chunk.x == 0:
                                                        new_file = (
                                                            "r."
                                                            + str(region_x - 1)
                                                            + "."
                                                            + str(region_z)
                                                            + ".mca"
                                                        )
                                                        if new_file in os.listdir(dir):
                                                            relevant_regions.append(
                                                                new_file
                                                            )
                                                    elif chunk.x == 31:
                                                        new_file = (
                                                            "r."
                                                            + str(region_x + 1)
                                                            + "."
                                                            + str(region_z)
                                                            + ".mca"
                                                        )
                                                        if new_file in os.listdir(dir):
                                                            relevant_regions.append(
                                                                new_file
                                                            )
                                                    if chunk.z == 0:
                                                        new_file = (
                                                            "r."
                                                            + str(region_x)
                                                            + "."
                                                            + str(region_z - 1)
                                                            + ".mca"
                                                        )
                                                        if new_file in os.listdir(dir):
                                                            relevant_regions.append(
                                                                new_file
                                                            )
                                                    elif chunk.z == 31:
                                                        new_file = (
                                                            "r."
                                                            + str(region_x)
                                                            + "."
                                                            + str(region_z + 1)
                                                            + ".mca"
                                                        )
                                                        if new_file in os.listdir(dir):
                                                            relevant_regions.append(
                                                                new_file
                                                            )
                                                    relevant_regions.append(filename)
                                                chunk_added = True
                                                break
                                        if chunk_added:
                                            break
        # Check for failure and send error message
        if len(build_chunks) == 0:
            print("Error: Build could not be found in region files")
            return

        if build_chunks:
            data = np.array([(chunk.x, chunk.z) for chunk in build_chunks])

            # Apply DBSCAN clustering
            dbscan = DBSCAN(eps=5, min_samples=5).fit(data)
            labels = dbscan.labels_

            main_cluster_label = np.argmax(np.bincount(labels[labels != -1]))  # Exclude noise points labeled as -1

            # Remove chunks that are outside the main cluster
            build_chunks = [chunk for chunk, label in zip(build_chunks, labels) if label == main_cluster_label]

            low_x = min(chunk.x for chunk in build_chunks)
            high_x = max(chunk.x for chunk in build_chunks)
            low_z = min(chunk.z for chunk in build_chunks)
            high_z = max(chunk.z for chunk in build_chunks)

            # Loop through the clusters
            schem_paths = []

            for filename in relevant_regions:
                if filename.endswith(".mca"):
                    # Retrieve the region
                    region = anvil.Region.from_file(os.path.join(dir, filename))
                    # Only search the region file if it is not empty (because apparently sometimes they are empty?)
                    if (region.data):
                        # Retrieve each chunk in the region
                        for x in range(0, 32):
                            for z in range(0, 32):
                                # Region files need not contain 32x32 chunks, so we must check if the chunk exists
                                if region.chunk_data(x, z):
                                    chunk = anvil.Region.get_chunk(region, x, z)
                                    if chunk not in build_chunks:
                                        if (chunk.x >= low_x and chunk.x <= high_x) and (chunk.z >= low_z and chunk.z <= high_z):
                                            build_chunks.append(chunk)
                print("Build chunks found!")
                World2Vec.extract_build(
                    build_chunks,
                    superflat,
                    save_dir,
                    build_name,
                    schem_paths,
                )

                # Call the extract_build function on cluster_chunks, pass in the builds_extracted integer
            return schem_paths
            # Iterate through .mca files in dir to fill in missing chunks

    # Extracts a build from a list of chunks and writes a file containing block info and coordinates
    def extract_build(
        chunks: List,
        superflat: bool,
        save_dir: str,
        build_name: str,
        schem_paths: List[str],
    ):
        print("Extracting build from chunks into " + build_name + "_" + ".schematic...")
        # Open the output file
        schem = mcschematic.MCSchematic()
        # Part of this process is finding the lowest y-value that can be considered the "surface"
        # This will almost certainly never be y=-100, so if this value is unchanged, we know something went wrong
        lowest_surface_y = 0
        # Iterate through the chunks
        min_range = 0
        if chunks[0].version is not None and chunks[0].version > 1451:
            min_range = -5
        level = 0
        all_surface_sections = []
        surface_section_mode = None
        # If it's a superflat world, we need to search the lower sections
        if superflat:
            lowest_surface_y = -100
            level = -100
        for chunk in chunks:
            superflat, surface_section = World2Vec.find_surface_section(
                chunk, min_range, 15, superflat
            )
            all_surface_sections.append(surface_section)
        # Find the mode (most common) surface section among the build chunks
        surface_section_mode = max(
            set(all_surface_sections), key=all_surface_sections.count
        )
        all_ys = []
        start_y = -8
        if superflat:
            start_y = 0
        for chunk in chunks:
            chunk_lowest_y = level
            # Iterate through the surface section and find the lowest surface block
            # Because we are specifying the section, we are using relative coordinates in the 0-16 range, rather than global coordinates
            # (this is better for us, as it is world-agnostic)
            # We start at -8 in the y level just in case the surface block is close to the section border (if it is not superflat)
            for x in range(0, 16):
                for z in range(0, 16):
                    for y in range(start_y, 16):
                        # Here we calculate the true y value, in order to compare against other sections
                        true_y = y + (surface_section_mode * 16)
                        block = World2Vec.convert_if_old(
                            anvil.Chunk.get_block(chunk, x, true_y, z)
                        )
                        block_above = World2Vec.convert_if_old(
                            anvil.Chunk.get_block(chunk, x, true_y + 1, z)
                        )
                        # Check if there is an air block above it, to confirm it is a surface block
                        if (
                            block != None
                            and block_above != None
                            and anvil.Block.name(block) != "minecraft:air"
                            and anvil.Block.name(block_above) == "minecraft:air"
                        ):
                            if chunk_lowest_y == level or true_y < chunk_lowest_y:
                                chunk_lowest_y = true_y
            all_ys.append(chunk_lowest_y)

        lowest_surface_y = int(sum(all_ys) / len(all_ys))
        min_y = 0
        if chunks[0].version is not None and chunks[0].version > 1451:
            min_y = -64
        if lowest_surface_y < min_y:
            lowest_surface_y = min_y
        elif surface_section_mode != min_range + 1:
            lowest_surface_y -= 1

        # Again, we don't need global coordinates, but we do need the blocks to be in the right places relative to each other
        # So, we're going to "create" our own (0, 0) and place everything relative to that point
        # To do this, we're just going to pick one of the chunks and call it the (0, 0) chunk, then map all the other chunks accordingly
        origin_x = chunks[0].x
        origin_z = chunks[0].z
        # Additionally, we are going to find blocks layer by layer, so we need to keep track of which y we are searching, starting with the value we found earlier
        current_y = lowest_surface_y
        # We also need a stopping point, so we need a flag to tell us when to stop searching for blocks (we don't want to spend time searching the sky)
        searching = True
        empty_layers = 0
        while searching:
            empty_layer = True
            for chunk in chunks:
                relative_chunk_x = chunk.x - origin_x
                relative_chunk_z = chunk.z - origin_z
                for x in range(0, 16):
                    for z in range(0, 16):
                        # This function CAN take global y values, so we don't have to worry about finding specific sections
                        block = World2Vec.convert_if_old(
                            anvil.Chunk.get_block(chunk, x, current_y, z)
                        )
                        # We're going to ignore air blocks, as we can just fill in empty coordinates later with air blocks
                        # This just cleans up the output file for better readability
                        if block != None and anvil.Block.name(block) != "minecraft:air":
                            if not anvil.Block.name(block).startswith("minecraft"):
                                # This is a modded block, and we should skip the build
                                print("Modded block found in build. Skipping...")
                                return
                            # We've found a non-air block, so this isn't an empty layer
                            empty_layer = False
                            # We need to map the coordinates to our new system now
                            new_x = (relative_chunk_x * 16) + x
                            new_y = current_y - lowest_surface_y
                            new_z = (relative_chunk_z * 16) + z
                            # Extract and format the block's properties (if they exist)
                            block_properties = "["
                            if len(block.properties) > 0:
                                for prop in sorted(block.properties):
                                    block_properties = (
                                        block_properties
                                        + prop
                                        + "="
                                        + str(block.properties.get(prop))
                                        + ","
                                    )
                            block_properties = block_properties[:-1] + "]"
                            # Finally, we write to the output file
                            if len(block.properties) > 0:
                                schem.setBlock(
                                    (int(new_x), int(new_y), int(new_z)),
                                    anvil.Block.name(block) + block_properties,
                                )
                            else:
                                schem.setBlock(
                                    (int(new_x), int(new_y), int(new_z)),
                                    anvil.Block.name(block),
                                )
            # If this layer is empty, stop searching
            if empty_layer:
                empty_layers += 1
                if empty_layers == 3:
                    searching = False
            # Otherwise, increase to the next y layer
            else:
                current_y += 1

        schem.save(
            save_dir, build_name, mcschematic.Version.JE_1_20_1
        )
        schem_file_path = os.path.join(
            save_dir, build_name + ".schem"
        )

        schem_paths.append(schem_file_path)

        print(
            "Build extracted to "
            + build_name
            + ".schematic...!\n"
        )

    def export_json_to_npy(input_file_path: str):
        # Load JSON data
        with open(input_file_path) as f:
            data = json.load(f)
        # Extract dimensions from JSON
        dimensions = data["worldDimensions"]
        width = dimensions["width"]
        height = dimensions["height"]
        length = dimensions["length"]
        # Create a 3D array with dimensions from JSON
        world_array = np.zeros((width, height, length), dtype=object)
        # Initialize counters for skipped and modded blocks

        # Fill the array with block names based on JSON data
        for block in data["blocks"]:
            x, y, z = block["x"], block["y"], block["z"]
            block_name = block["name"]

            try:
                if block_name[0].isupper():
                    continue
                elif not block_name.startswith("minecraft:"):
                    return None

                world_array[x, y, z] = block_name

            except IndexError:
                world_array[x, y, z] = "minecraft:air"
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                return None
        return world_array

    def export_npy_to_hdf5(output_file_prefix: str, world_array: np.ndarray):
        # Open HDF5 file in write mode
        with h5py.File(f"{output_file_prefix}.h5", "w") as f:
            # Create a dataset in the HDF5 file with the same name as the file name and write the array data
            f.create_dataset(output_file_prefix, data=world_array)
