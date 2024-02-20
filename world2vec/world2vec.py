import anvil
import os
from typing import Generator

# Class to parse data files and vectorize them into information the model can train on
class World2Vec:
    # Reads all region files in dir and returns a Generator of Chunks, all of which contain blocks that are not in natural_blocks.txt
    def get_build_chunks(dir: str) -> Generator[anvil.Chunk, None, None]:
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
                # Retrieve each chunk in the region
                for x in range(0, 31):
                    for z in range(0, 31):
                        # Region files need not contain 32x32 chunks, so we must check if the chunk exists
                        if region.chunk_data(x, z):
                            chunk = anvil.Region.get_chunk(region, x, z)
                            # If there is already an identified build chunk
                            if last_build_chunk[0]:
                                # If this chunk is too far away, just skip it
                                if (abs(chunk.x - last_build_chunk[0]) >= 3) or (abs(chunk.z - last_build_chunk[1]) >= 3):
                                    continue
                            # Only check sections 3-10, eliminating sections deep underground or high in the sky
                            for s in range(3, 10):
                                section = anvil.Chunk.get_section(chunk, s)
                                # Check each block in the chunk
                                for block in anvil.Chunk.stream_blocks(chunk, section=section):
                                    # If it's not a natural block, add this chunk to the Generator
                                    if anvil.Block.name(block) not in natural_blocks:
                                        # TEST
                                        # print(anvil.Block.name(block))
                                        yield chunk
                                        last_build_chunk[0] = chunk.x
                                        last_build_chunk[1] = chunk.z
                                        break