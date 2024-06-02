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

# Not in vocab token 
NIV_TOK = 4000

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
    integerized_build = np.zeros((x_dim, y_dim, z_dim), dtype=np.uint16)

    for x, y, z in product(range(0, x_dim), range(0, y_dim), range(0, z_dim)): 
        blockname = build_array[x, y, z]
        value = block2tok.get(blockname)
       
        if (value is None): 
            blockname_and_states = blockname.split('[')
            split_blockname = blockname_and_states[0]
            value = block2tok.get(split_blockname)

            if value is None: 
                print("Couldn't find " + split_blockname + " aka " + blockname + " in block2tok")
                integerized_build[x, y, z] = NIV_TOK
            else: 
                if (isinstance(value, dict)): 
                    block_states = blockname_and_states[1].replace(']', "")
                    token = value.get(block_states)
                    if (token is None): 
                        print("For block: \"" + split_blockname + "\", couldn't find state string: \"" + block_states + "\" in block2tok_mc.json")
                        integerized_build[x, y, z] = NIV_TOK
                    else: 
                        integerized_build[x, y, z] = token
                else: 
                    integerized_build[x, y, z] = value
        else : 
            integerized_build[x, y, z] = block2tok.get(blockname)
        
    return integerized_build

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
    integerized_build = convert_block_names_to_integers(build_npy_array) 
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
