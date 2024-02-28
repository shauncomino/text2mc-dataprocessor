import anvil
import os
from typing import Generator
import mcschematic

# Now you can use mcschematic

# Class to parse data files and vectorize them into information the model can train on
class World2Vec:

    # Reads all region files in dir and returns a Generator of Chunks, all of which contain blocks that are not in natural_blocks.txt
    def get_build_chunks(dir: str) -> Generator[anvil.Chunk, None, None]:
        print("Searching directory " + dir + "...")
        # Read in the natural blocks to an array
        nb_file = open("natural_blocks.txt", 'r')
        natural_blocks = nb_file.read().splitlines()
        nb_file.close()
        # This variable tracks the coordinates of the last identified build chunk, used to reduce computation time 
        # when faraway chunks are reached
        last_build_chunk = [None, None]
        # Iterate through .mca files in dir
        for filename in os.listdir(dir):
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
                                # If there is already an identified build chunk
                                if last_build_chunk[0] != None:
                                    # If this chunk is too far away, just skip it
                                    if (abs(chunk.x - last_build_chunk[0]) >= 3) or (abs(chunk.z - last_build_chunk[1]) >= 3):
                                        continue
                                # Only check sections 3-10, eliminating sections deep underground or high in the sky
                                chunk_added = False
                                for s in range(3, 10):
                                    section = anvil.Chunk.get_section(chunk, s)
                                    # Check each block in the section
                                    for block in anvil.Chunk.stream_blocks(chunk, section=section):
                                        # If it's not a natural block, add this chunk to the Generator
                                        if anvil.Block.name(block) not in natural_blocks:
                                            yield chunk
                                            last_build_chunk[0] = chunk.x
                                            last_build_chunk[1] = chunk.z
                                            chunk_added = True
                                            break
                                    if chunk_added:
                                        break
        # Check for failure and send error message
        if last_build_chunk[0] == None:
            print("Error: Build could not be found in region files")
            return
        print("Build chunks found!")
    
    # Extracts a build from a list of chunks and writes a file containing block info and coordinates
    def extract_build(chunk_gen: Generator[anvil.Chunk, None, None], build_no: int):
        print("Extracting build from chunks into " + "my_schematics" + ".schematic...")
        # Read in terrain blocks to an array
        t_file = open("terrain.txt", 'r')
        terrain = t_file.read().splitlines()
        t_file.close()
        # Open the output file
        schem = mcschematic.MCSchematic()
        #                   build_file = open(filename + ".txt", 'w')
        # Part of this process is finding the lowest y-value that can be considered the "surface"
        # This will almost certainly never by y=0, so if this value is unchanged, we know something went wrong
        lowest_surface_y = 0
        # Since we need to iterate through the chunks multiple times, we need to convert the generator to a list
        chunks = list(chunk_gen)
        # Iterate through the chunks
        for chunk in chunks:
            surface_section = None
            surface_section_y = 0
            # Begin with section 3 and find the first section up from there that contains a large amount of air (the "surface" section)
            # We stop at section 10 because that is the highest section that get_build_chunks() searches
            for s in range(3, 10):
                air_count = 0
                section = anvil.Chunk.get_section(chunk, s)
                for block in anvil.Chunk.stream_blocks(chunk, section=section):
                    if anvil.Block.name(block) == "minecraft:air":
                        air_count += 1
                        # We'll check for a section to have a good portion of air, testing says 1024 blocks is a good fit
                        if air_count == 1024:
                            surface_section = section
                            surface_section_y = s
                            break
                # If we've already found a surface section, stop searching
                if surface_section != None:
                    break
            # Check for failure and output an error message
            if surface_section is None:
                print("Error: No surface section found in chunk", chunk.x, chunk.z)
                return
            # Iterate through the surface section and find the lowest surface (terrain) block
            # Because we are specifying the section, we are using relative coordinates in the 0-16 range, rather than global coordinates 
            # (this is better for us, as it is world-agnostic)
            for x in range(0, 16):
                for z in range(0, 16):
                    for y in range(0, 16):
                        # Here we calculate the true y value, in order to compare against other sections
                        true_y = y + (surface_section_y * 16)
                        block = anvil.Chunk.get_block(chunk, x, y, z, section=surface_section)
                        if anvil.Block.name(block) in terrain:
                            # We know the block is a terrain block, now we need to check if there is an air block above it, to confirm it is a surface terrain block
                            if anvil.Block.name(anvil.Chunk.get_block(chunk, x, true_y + 1, z)) == "minecraft:air":
                                if lowest_surface_y == 0 or true_y < lowest_surface_y:
                                    lowest_surface_y = true_y
        # Check for failure and output an error message
        if lowest_surface_y == 0:
            print("Error: Could not find the surface y-value")
            return
        # Now, we have the y-value of the lowest surface block. Next, we have to write the build blocks (from that y-value up) to the output file
        # Since the anvil-parser library only uses relative x and z values, we need to be careful about how we iterate through the chunks
        # Again, we don't need global coordinates, but we do need the blocks to be in the right places relative to each other
        # So, we're going to "create" our own (0, 0) and place everything relative to that point
        # To do this, we're just going to pick one of the chunks and call it the (0, 0) chunk, then map all the other chunks accordingly
        origin_x = chunks[0].x
        origin_z = chunks[0].z
        # Additionally, we are going to find blocks layer by layer, so we need to keep track of which y we are searching, starting with the value we found earlier
        current_y = lowest_surface_y
        # We also need a stopping point, so we need a flag to tell us when to stop searching for blocks (we don't want to spend time searching the sky)
        searching = True
        while (searching):
            empty_layer = True
            for chunk in chunks:
                relative_chunk_x = chunk.x - origin_x
                relative_chunk_z = chunk.z - origin_z
                for x in range(0, 16):
                    for z in range(0, 16):
                        # This function CAN take global y values, so we don't have to worry about finding specific sections
                        block = anvil.Chunk.get_block(chunk, x, current_y, z)
                        # We're going to ignore air blocks, as we can just fill in empty coordinates later with air blocks
                        # This just cleans up the output file for better readability
                        if anvil.Block.name(block) != "minecraft:air":
                            # We've found a non-air block, so this isn't an empty layer
                            empty_layer = False
                            # We need to map the coordinates to our new system now
                            new_x = (relative_chunk_x * 16) + x
                            new_y = current_y - lowest_surface_y
                            new_z = (relative_chunk_z * 16) + z
                            # Finally, we write to the output file
                            schem.setBlock((int(new_x), int(new_y), int(new_z)),anvil.Block.name(block))
                            #                       build_file.write(anvil.Block.name(block) + " " + str(new_x) + " " + str(new_y) + " " + str(new_z) + "\n")
            # If this layer is empty, stop searching
            if (empty_layer):
                searching = False
            # Otherwise, increase to the next y layer
            else:
                current_y += 1
        # Close the output file
        schem.save("testbuilds", "my_schematic_" + str(build_no), mcschematic.Version.JE_1_20_1)
                        #build_file.close()
        print("Build extracted to " + "my_schematics" + str(build_no) + ".schematic...!\n")