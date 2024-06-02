import pandas as pd
import json
import time
import os, shutil
from dataclasses import dataclass, field
from typing import Optional, List
from itertools import product
import re
import traceback
from world2vec import World2Vec
import sys
import subprocess
import traceback
import zipfile
import numpy as np
import h5py
import glob
import patoolib
import json
import h5py
import sys 

# Load the blockname -> integer lookup dictionary
cwd = os.getcwd() 
block2tok_filepath = os.path.join(cwd, "block2tok_mc.json") 
with open(block2tok_filepath, 'r') as file:
    block2tok = json.load(file)

def has_blockstates(blockname: str): 
    regex_square_brackets = r'\[.*?\]'

    if re.search(regex_square_brackets, blockname):
        return True
    
    return False 

def convert_block_names_to_integers(build_array: np.ndarray):
    x_dim, y_dim, z_dim = build_array.shape
    blockset = set() # For debug purposes (don't want 9000 of the same blocks coming up as err)

    for x, y, z in product(range(0, x_dim), range(0, y_dim), range(0, z_dim)): 
        blockname = build_array[x, y, z]
        
        if blockname not in blockset:
            if (block2tok.get(blockname) is None): 
                blockname_split = blockname.split('[')

                if block2tok.get(blockname_split[0]) is None: 
                    print("couldn't find " + blockname_split[0] + " in block2tok")
                else: 
                    print("found " + blockname_split[0] + " in block2tok")
            else : 
                print("found " + blockname + " in block2tok")
            blockset.add(blockname)

    return build_array

def main(): 
    """ 
        world -> schem, 
        schem -> JSON, 
        JSON -> numpy array, 
        numpy array -> HDF5
    """

    builds_raw_dir = os.path.join(cwd, "builds_raw")
    build_name = "Boulevardier's_Sanctuary_of_All_Times" # This is the name of the build folder for your current run
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
        print("Error: schem file for " + build_name + " was not created.")
    else: 
        print("Schem file for " + build_name + " is ready.")

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
        print("Error: JSON for " + build_name + " was not created.")
    else: 
        print("JSON for " + build_name  + " is ready.")
        
    # Get numpy array from json file
    build_npy_array = World2Vec.export_json_to_npy(json_filepath)
    print("Build npy array for " + build_name  + " is ready.")

    # Integerize the numpy array, instead of doing back-and-forth conversion 
    build_npy_array = convert_block_names_to_integers(build_npy_array) 
    print("Integerized build npy array for " + build_name  + " is ready.")

    # Get HDF5 file from numpy array
    with h5py.File(hdf5_filepath, "w") as file:
        file.create_dataset(os.path.split(hdf5_filepath)[-1], data=build_npy_array)
    if not os.path.exists(hdf5_filepath):
        print("Error: HDF5 for " + build_name + " was not created.")
    else: 
        print("HDF5 for " + build_name  + " is ready.")
    

if __name__ == "__main__":
    main()
