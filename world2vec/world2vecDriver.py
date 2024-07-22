import pandas as pd
import json
import time
import os, shutil
from dataclasses import dataclass, field
from typing import Optional, List
import re
from itertools import product
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


@dataclass
class world2vecDriverConfig:
    DOWNLOADED_BUILDS_FOLDER: str = None
    """ Downloaded builds """

    PROCESSED_BUILDS_FOLDER: str = None
    """ Processed .hdf5 builds """

    BLOCK_TO_TOKEN_JSON_PATH: str = None
    """ Path to the JSON block name to token lookup file """

    NATURAL_BLOCKS_PATH: str = None
    """ Path to the .txt file that delineates the naturally spawning blocks """

    JAR_RUNNER_PATH: str = None

    cwd: str = os.path.dirname(os.path.abspath(__file__))

    # Not in vocab token
    NIV_TOK = 4000

    def __post_init__(self):

        if not self.NATURAL_BLOCKS_PATH or not os.path.exists(self.NATURAL_BLOCKS_PATH):
            self.NATURAL_BLOCKS_PATH = os.path.join(self.cwd, "natural_blocks.txt")

        if not self.JAR_RUNNER_PATH or not os.path.exists(self.JAR_RUNNER_PATH):
            self.JAR_RUNNER_PATH = os.path.join(self.cwd, "schematic-loader.jar")
        if not self.DOWNLOADED_BUILDS_FOLDER or not os.path.exists(
            self.DOWNLOADED_BUILDS_FOLDER
        ):
            print("Invalid build source folder or not provided")

        if self.PROCESSED_BUILDS_FOLDER is None or not os.path.exists(
            self.PROCESSED_BUILDS_FOLDER
        ):
            print(
                f"Processed build folder not passed or incorrect, creating a new folder. Passed path: {self.PROCESSED_BUILDS_FOLDER}"
            )
            self.PROCESSED_BUILDS_FOLDER = os.path.join(
                os.path.abspath("./"), "vectorized_builds"
            )

        if self.BLOCK_TO_TOKEN_JSON_PATH is None or not os.path.exists(
            self.BLOCK_TO_TOKEN_JSON_PATH
        ):
            try:
                this_running_python_file_path = os.path.abspath(__file__)
                cwd = os.path.dirname(this_running_python_file_path)
                block2tok_filepath = os.path.join(cwd, "block2tok.json")
                with open(block2tok_filepath, "r") as file:
                    self.block2tok = json.load(file)
            except:
                print("Couldn't load block2tok.json file")
                exit(1)

        os.makedirs(self.PROCESSED_BUILDS_FOLDER, exist_ok=True)


class world2vecDriver:

    def __init__(self, cfg: world2vecDriverConfig = None):
        self.cfg = cfg

    def process_batch(
        self, dataframe_path: str, start_index: int, end_index: int, batch_num: int
    ):
        """
        Process a batch of builds from a DataFrame.

        :param dataframe_path: The path to the DataFrame CSV file containing the builds information.
        :param start_index: The start index of the batch.
        :param end_index: The end index of the batch.
        :param batch_num: The batch number for this processing batch, used for temporary directory naming.
        """
        dataframe = pd.read_csv(dataframe_path, on_bad_lines="warn")
        dataframe["PROCESSED_PATHS"] = pd.Series(dtype="object")
        temp_dir_name = f"temp" + str(batch_num)
        temp_dir_path = os.path.join(
            os.path.dirname(self.cfg.PROCESSED_BUILDS_FOLDER), temp_dir_name
        )
        self.create_directory(temp_dir_path)

        successes = 0

        for i, row in dataframe.iloc[start_index:end_index].iterrows():
            try:
                unique_name = f"batch_{batch_num}_{str(i)}"
                filename = row["FILENAME"]
                processed_paths = self.process_build(
                    filename,
                    processed_file_name=unique_name,
                    temp_dir_path=temp_dir_path,
                )
                if not processed_paths:
                    continue
                dataframe.at[i, "PROCESSED_PATHS"] = processed_paths
                successes += 1
            except Exception as e:
                print(e)
                traceback.format_exc()
        shutil.rmtree(temp_dir_path)
        print("Batch %d: %d builds successfully processed out of %d\n"(batch_num, successes, end_index - start_index))

    def process_build(
        self,
        filename: str,
        processed_file_name: str = "temp_schem",
        temp_dir_path: str = "temp",
        straight_to_hdf5=True,
    ) -> List[str]:
        """
        Process a single build file and return the processed paths.

        :param filename: The filename of the build to be processed.
        :param processed_file_name: A suffix for the temporary directory to avoid conflicts.
        :return: A list of processed file paths.
        """
        processed_paths = []
        if not os.path.exists(temp_dir_path):
            os.mkdir(temp_dir_path)

        temp_extract = os.path.join(temp_dir_path, "extract")

        try:
            unprocessed_build_path = os.path.join(
                self.cfg.DOWNLOADED_BUILDS_FOLDER, filename
            )
            if filename.endswith(".zip") or filename.endswith(".rar"):
                # Names of .zip and .rar files have '+' instead of spaces, which causes them not to be found
                # unprocessed_build_path = os.path.join(
                #     self.cfg.DOWNLOADED_BUILDS_FOLDER, filename.replace('+', ' ')
                # )
                self.extract_archive_to_temporary_directory(
                    unprocessed_build_path, temp_extract
                )

                # Search all files within the temporary directory once
                all_files = glob.glob(
                    os.path.join(temp_extract, "**/*"), recursive=True
                )

                schems_paths = []
                mca_paths = []

                for path in all_files:
                    if path.endswith(".schem") or path.endswith(".schematic"):
                        schems_paths.append(path)
                    if path.endswith(".mca"):
                        mca_paths.append(path)

                # If '.schem' files or '.schematic' files are present, use them
                if len(schems_paths) > 0:
                    processed_paths = schems_paths

                # If neither '.schem' nor '.schematic' files are found, but '.mca' files are, convert them
                if len(mca_paths) > 0 and len(schems_paths) == 0:
                    schem_paths = self.convert_build_to_schemfile(
                        temp_extract, f"build_{processed_file_name}"
                    )
                    if schem_paths is None:
                        return []
                    processed_paths = schem_paths

            elif filename.endswith(".schematic") or filename.endswith(".schem"):
                processed_paths = [
                    os.path.join(self.cfg.DOWNLOADED_BUILDS_FOLDER, filename)
                ]

            if straight_to_hdf5:
                new_paths = list()
                for path in processed_paths:
                    try:
                        temp_json_path = os.path.join(
                            temp_dir_path, f"{processed_file_name}.json"
                        )
                        hdf5_path = os.path.join(
                            self.cfg.PROCESSED_BUILDS_FOLDER,
                            f"{processed_file_name}.h5",
                        )
                        print(f"Schempaths: {path}")
                        self.convert_schemfile_to_json(path, temp_json_path)
                        npy_array = self.convert_json_to_npy(temp_json_path)
                        npy_array = self.convert_block_names_to_integers(npy_array, processed_file_name)
                        self.convert_vector_to_hdf5(npy_array, hdf5_path)
                        if os.path.exists(hdf5_path):
                            new_paths.append(hdf5_path)
                    except Exception as e:
                        print(
                            f"Error processing schem to hdf5: {os.path.split(path)[-1]}"
                        )
                        print(e)
                        traceback.format_exc()

                processed_paths = new_paths

            print(f"Processed paths: {processed_paths}")
            self.delete_directory_contents(temp_extract)

        except Exception as e:
            print(f"Error processing build {filename}: {e}")
            traceback.print_exc()

        return processed_paths

    def extract_archive_to_temporary_directory(
        self, source_archive_path: str = None, outfolder_path: str = None
    ):
        patoolib.extract_archive(
            source_archive_path, outdir=outfolder_path, verbosity=-1
        )

    def delete_directory_contents(self, folder: str = None):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))

    def convert_build_to_schemfile(self, folder_or_build_path, processed_file_prefix):
        regions_dir = World2Vec.find_regions_dir(folder_or_build_path)[0]
        return World2Vec.get_build(
            regions_dir,
            self.cfg.PROCESSED_BUILDS_FOLDER,
            processed_file_prefix,
            natural_blocks_path=self.cfg.NATURAL_BLOCKS_PATH,
        )

    def convert_schemfile_to_json(self, schem_file_path: str, json_export_path: str):
        print("Calling subprocess")
        subprocess.call(
            [
                "java",
                '-Xms512m',  # Set initial Java heap size
                '-Xmx4096m',
                "-jar",
                self.cfg.JAR_RUNNER_PATH,
                schem_file_path,
                json_export_path,
            ]
        )

    def convert_json_to_npy(self, json_file_path) -> np.ndarray:
        return World2Vec.export_json_to_npy(json_file_path)

    def convert_vector_to_hdf5(self, vector, path):
        with h5py.File(path, "w") as file:
            file.create_dataset(
                os.path.split(path)[-1],
                data=vector,
                dtype="uint16",  # Specify the data type as unsigned 16-bit integer
                compression="gzip",  # Use gzip compression
                compression_opts=9,  # Maximum compression level
            )

    # Used to create processed and temporary directories
    def create_directory(self, dir_path: str):
        try:
            if os.path.exists(dir_path):
                return
            print("Temporary directory does not exist, creating")
            os.mkdir(dir_path)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            print("An error occured when creating directory")

    # Deletes a directory after it has been extracted/used
    def delete_directory(self, dir_path: str):
        if os.path.exists(dir_path):
            if os.path.isdir(dir_path):
                os.rmdir(dir_path)

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

        # Fill the array with block names based on JSON data
        for block in data["blocks"]:
            x, y, z = block["x"], block["y"], block["z"]
            block_name = block["name"]
            world_array[x, y, z] = block_name

        return world_array

    def export_npy_to_hdf5(output_file_prefix: str, world_array: np.ndarray):
        # Open HDF5 file in write mode
        with h5py.File(f"{output_file_prefix}.h5", "w") as f:
            # Create a dataset in the HDF5 file with the same name as the file name and write the array data
            f.create_dataset(output_file_prefix, data=world_array)

    def find_closest_match(self, query, options):
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

    def convert_block_names_to_integers(self, build_array: np.ndarray, filename):
        block2tok = self.cfg.block2tok
        x_dim, y_dim, z_dim = build_array.shape
        integerized_build = np.zeros((x_dim, y_dim, z_dim), dtype=np.uint16)
        missing_blocks = []
        count = 0

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
                token = self.cfg.NIV_TOK
                if blockname not in missing_blocks:
                    missing_blocks.append(blockname)
                    count += 1                  
            
            # Blockname maps to dictionary
            elif isinstance(value, dict):
                standard_blockstates = self.find_closest_match(
                    blockstates, value.keys()
                )

                if standard_blockstates is None:
                    standard_blockstates = list(value.keys())[0]

                token = value.get(standard_blockstates)

            # Blockname maps directly to token
            else:
                token = value

            integerized_build[x, y, z] = token
        if(count != 0):
            with open("/lustre/fs1/groups/jaedo/world2vec/missing_blocks/" + filename + ".json",'w') as f:
                json.dump(missing_blocks,f)
        return integerized_build


def main():
    # Code to load from command line parameters

    args = sys.argv
    source_df_path = args[1]
    source_builds_dir = args[2]
    processed_builds_folder = args[3]
    batch_num = args[4]
    start_index = (int(batch_num) - 1) * 10
    end_index =  int(batch_num) * 10 - 1
    
    print(f"Source Dataframe Path: {source_df_path}")
    print(f"Source Unprocessed Builds Directory: {source_builds_dir}")
    print(f"Processed Builds Directory: {processed_builds_folder}")
    print(f"Starting Index: {start_index}")
    print(f"Ending Index: {end_index}")
    print(f"Batch Number: {batch_num}")

    config = world2vecDriverConfig(DOWNLOADED_BUILDS_FOLDER=source_builds_dir, PROCESSED_BUILDS_FOLDER=processed_builds_folder)
    world2vecdriver = world2vecDriver(cfg=config)

    world2vecdriver.process_batch(source_df_path, start_index, end_index, batch_num)

    # world2vecdriver.process_batch(
    #     dataframe_path=r"C:\Users\shaun\OneDrive\Desktop\personal\CS classes\CS classes\COP4934\text2mc\text2mc-dataprocessor\projects_df_processed.csv",
    #     start_index=0,
    #     end_index=100,
    #     batch_num=1,
    # )

    # Code to test the driver manually

    # config = world2vecDriverConfig(
    #     DOWNLOADED_BUILDS_FOLDER=r"C:\Users\skepp\source\repos\text2mc-dataprocessor\world2vec\builds_raw",
    #     PROCESSED_BUILDS_FOLDER=r"C:\Users\skepp\source\repos\text2mc-dataprocessor\world2vec\builds_hdf5",
    # )
    # world2vecdriver = world2vecDriver(cfg=config)

    # projects_df = pd.read_csv(
    #     r"C:\Users\skepp\source\repos\text2mc-dataprocessor\projects_df_processed.csv"
    # )

    # num_to_process = 5

    # Process .schem files
    # print("Processings .schem files")
    # schem_df = projects_df[projects_df["SUFFIX"] == ".schem"]
    # for i, row in schem_df[0:num_to_process].iterrows():
    #     world2vecdriver.process_build(
    #         row["FILENAME"],
    #         f"schem_test_{i}",
    #         straight_to_hdf5=True,
    #     )

    # Process .zip archives
    # print("Processing .zip files")
    # zip_df = projects_df[projects_df["SUFFIX"] == ".zip"]
    # for i, row in zip_df[0:num_to_process].iterrows():
    #     world2vecdriver.process_build(
    #         row["FILENAME"], f"zip_test_{i}", straight_to_hdf5=True
    #     )

    # Process .schematic files
    # print("Processing .schematic files")
    # schematic_df = projects_df[projects_df["SUFFIX"] == ".schematic"]
    # for i, row in schematic_df[0:num_to_process].iterrows():
    #     world2vecdriver.process_build(
    #         row["FILENAME"], f"schematic_test_{i}", straight_to_hdf5=True
    #     )

    # Process .rar archives
    # print("Processing .rar files")
    # rar_df = projects_df[projects_df["SUFFIX"] == ".rar"]
    # for i, row in rar_df[0:num_to_process].iterrows():
    #     world2vecdriver.process_build(
    #         row["FILENAME"],
    #         f"rar_test{i}",
    #         straight_to_hdf5=True,
    #     )


if __name__ == "__main__":
    main()