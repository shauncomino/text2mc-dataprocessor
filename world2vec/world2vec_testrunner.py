import json
import os
from itertools import product
from world2vec import World2Vec
import subprocess
from loguru import logger
import numpy as np
import h5py
import json
import h5py

# Not in vocab token
NIV_TOK = 4000

# Load the blockname -> integer lookup dictionary
this_running_python_file_path = os.path.abspath(__file__)
cwd = os.path.dirname(this_running_python_file_path)
block2tok_filepath = os.path.join(cwd, "block2tok.json")
with open(block2tok_filepath, "r") as file:
    block2tok = json.load(file)

""" Find closest matching string to given query out of options by number of common words """


def find_closest_match(query, options):
    query_words = set(query)

    best_option = None
    max_matching = 0

    for option in options:
        option_words = set(option.split(","))
        shared_words = query_words.intersection(option_words)
        matching = len(shared_words)

        if matching > max_matching:
            best_option = option
            max_matching = matching

    return best_option


def convert_block_names_to_integers(build_array: np.ndarray):
    x_dim, y_dim, z_dim = build_array.shape
    integerized_build = np.zeros((x_dim, y_dim, z_dim), dtype=np.uint16)

    for x, y, z in product(range(0, x_dim), range(0, y_dim), range(0, z_dim)):
        blockname = build_array[x, y, z]
        token = None

        # If there are block states, separate from block name
        if "[" in blockname:
            blockstates = blockname.replace("[", ",").replace("]", "").split(",")
            blockname = blockstates.pop(0)

        value = block2tok.get(blockname)

        # Blockname maps to nothing
        if value is None:
            logger.error("Couldn't find: \"" + blockname + '" in block2tok')
            token = NIV_TOK

        # Blockname maps to dictionary
        elif isinstance(value, dict):
            standard_blockstates = find_closest_match(blockstates, value.keys())

            if standard_blockstates is None:
                standard_blockstates = list(value.keys())[0]
                logger.warning(
                    "Couldn't find blockstates for blockname: "
                    + blockname
                    + " with blockstates: "
                    + str(blockstates)
                    + " . Using default: "
                    + str(standard_blockstates)
                )

            token = value.get(standard_blockstates)

        # Blockname maps directly to token
        else:
            token = value

        integerized_build[x, y, z] = token

    return integerized_build


def main():
    """
    world -> schem,
    schem -> JSON,
    JSON -> numpy array,
    numpy array -> HDF5
    """

    builds_raw_dir = os.path.join(cwd, "builds_raw")
    build_name = "Boulevardier's_Shophouse"  # This is the name of the build folder for your current run
    region_dir = os.path.join(builds_raw_dir, build_name, "region")
    print("region dir: " + region_dir)

    schem_dir = os.path.join(cwd, "builds_schem")
    schem_filename = build_name + "_1.schem"
    schem_filepath = os.path.join(schem_dir, schem_filename)

    """
    if not os.path.isfile(schem_filepath):
        print("creating schem file: " + schem_filepath)
        with open(schem_filepath, 'w') as file:
            pass 
    """

    json_dir = os.path.join(cwd, "builds_json")
    json_filename = build_name + ".json"
    json_filepath = os.path.join(json_dir, json_filename)

    natural_blocks_path = os.path.join(cwd, "natural_blocks.txt")
    schematic_loader_jar_path = os.path.join(cwd, "schematic-loader.jar")
    print(schem_filepath)
    hdf5_filepath = os.path.join(cwd, "builds_hdf5")
    hdf5_filename = build_name + ".h5"
    hdf5_filepath = os.path.join(hdf5_filepath, hdf5_filename)

    # Get schematic file for build
    if not os.path.exists(schem_dir):
        World2Vec.get_build(region_dir, schem_dir, build_name, natural_blocks_path)

    if not os.path.exists(schem_dir):
        logger.error("schem file for " + build_name + " was not created.")
    else:
        logger.info("Schem file for " + build_name + " is ready.")

    # Get json file from schematic file
    subprocess.call(
        [
            "java",
            "-jar",
            schematic_loader_jar_path,
            schem_filepath,
            json_filepath,
        ]
    )

    if not os.path.exists(json_filepath):
        logger.error("JSON for " + build_name + " was not created.")
    else:
        logger.info("JSON for " + build_name + " is ready.")

    # Get numpy array from json file
    build_npy_array = World2Vec.export_json_to_npy(json_filepath)
    logger.info("Build npy array for " + build_name + " is ready.")

    # Integerize the numpy array, instead of doing back-and-forth conversion
    integerized_build = convert_block_names_to_integers(build_npy_array)
    logger.info("Integerized build npy array for " + build_name + " is ready.")

    # Get HDF5 file from numpy array
    with h5py.File(hdf5_filepath, "w") as file:
        file.create_dataset(os.path.split(hdf5_filepath)[-1], data=integerized_build)
    if not os.path.exists(hdf5_filepath):
        logger.error("HDF5 for " + build_name + " was not created.")
    else:
        logger.info("HDF5 for " + build_name + " is ready.")


if __name__ == "__main__":
    main()
