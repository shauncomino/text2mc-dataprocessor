import pandas as pd
import json
import time
import os, shutil
from dataclasses import dataclass, field
from typing import Optional, List
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


@dataclass
class world2vecDriverConfig:
    DOWNLOADED_BUILDS_FOLDER: str = None
    """ Downloaded builds """

    PROCESSED_BUILDS_FOLDER: str = None
    """ Processed .hdf5 builds """

    BLOCK_JSON_PATH: str = None
    """ Path to the JSON blocks path """

    BLOCK_TO_TOKEN_JSON_PATH: str = None
    """ Path to the JSON block name to token lookup file """

    def __post_init__(self):

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

        if self.BLOCK_JSON_PATH is None or not os.path.exists(self.BLOCK_JSON_PATH):
            print("Block JSON not passed or invalid path, making new .json")

        if self.BLOCK_TO_TOKEN_JSON_PATH is None or not os.path.exists(
            self.BLOCK_TO_TOKEN_JSON_PATH
        ):
            print("Block to token JSON not passed or invalid path.")

        if self.BLOCK_TO_TOKEN_JSON_PATH is None or not os.path.exists(
            self.BLOCK_TO_TOKEN_JSON_PATH
        ):
            print("Block to token JSON not passed or invalid path.")

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
        temp_dir_name = f"temp"
        temp_dir_path = os.path.join(
            os.path.dirname(self.cfg.PROCESSED_BUILDS_FOLDER), temp_dir_name
        )
        self.create_directory(temp_dir_path)

        for i, row in dataframe.iloc[start_index:end_index].iterrows():
            try:
                unique_name = f"batch_{batch_num}_{str(i)}"
                filename = row["FILENAME"]
                processed_paths = self.process_build(
                    filename,
                    processed_file_name=unique_name,
                    temp_dir_path=temp_dir_path,
                )
                dataframe.at[i, "PROCESSED_PATHS"] = processed_paths
                for path in processed_paths:
                    self.convert_schemfile_to_hdf5(
                        path,
                        temp_dir_path,
                        os.path.join(temp_dir_path, unique_name),
                        unique_name,
                    )
            except Exception as e:
                print(e)
                traceback.format_exc()

    def process_build(
        self,
        filename: str,
        processed_file_name: str = "temp_schem",
        temp_dir_path: str = "temp",
        straight_to_hdf5=False,
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
        try:
            unprocessed_build_path = os.path.join(
                self.cfg.DOWNLOADED_BUILDS_FOLDER, filename
            )
            if filename.endswith(".zip") or filename.endswith(".rar"):
                self.extract_archive_to_temporary_directory(
                    unprocessed_build_path, temp_dir_path
                )

                # Search all files within the temporary directory once
                all_files = glob.glob(
                    os.path.join(temp_dir_path, "**/*"), recursive=True
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
                        temp_dir_path, f"build_{processed_file_name}"
                    )
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
                        vfunc_integerize = np.vectorize(self.integerize_build)
                        npy_array = vfunc_integerize(npy_array)
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
            self.delete_directory_contents(temp_dir_path)

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

    def integerize_build(self, block_name_array):
        block2tok_lookup = dict()

        integerized_array = np.empty(block_name_array.shape, dtype=int)
        with open(self.cfg.BLOCK_TO_TOKEN_JSON_PATH, "r") as file:
            block2tok_lookup = json.load(file)

        for i in range(block_name_array.shape[0]):
            for j in range(block_name_array.shape[1]):
                for k in range(block_name_array.shape[2]):
                    block_name = block_name_array[i, j, k]

                    token = block2tok_lookup.get(block_name, -1)
                    integerized_array[i, j, k] = token
                    print("Block name: {}, Token: {}".format(block_name, token))

        return integerized_array

    def convert_build_to_schemfile(self, folder_or_build_path, processed_file_prefix):
        print(folder_or_build_path)
        regions_dir = World2Vec.find_regions_dir(folder_or_build_path)[0]
        return World2Vec.get_build(
            regions_dir,
            self.cfg.PROCESSED_BUILDS_FOLDER,
            processed_file_prefix,
            natural_blocks_path=r"C:\Projects\text2mc\SCRUM-whatever\text2mc-dataprocessor\world2vec\natural_blocks.txt",
        )

    def convert_schemfile_to_json(self, schem_file_path: str, json_export_path: str):
        print("Calling subprocess")
        subprocess.call(
            [
                "java",
                "-jar",
                r"C:\Projects\text2mc\New folder\text2mc-dataprocessor\world2vec\schematic-loader.jar",
                schem_file_path,
                json_export_path,
            ]
        )

    def convert_json_to_npy(self, json_file_path) -> np.ndarray:
        return World2Vec.export_json_to_npy(json_file_path)

    def convert_vector_to_hdf5(self, vector, path):
        with h5py.File(path, "w") as file:
            file.create_dataset(os.path.split(path)[-1], data=vector)

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


def main():
    # Code to load from command line parameters

    # args = sys.argv
    # source_df_path = args[1]
    # source_builds_dir = args[2]
    # processed_builds_folder = args[3]
    # start_index = args[4]
    # end_index = args[5]
    # batch_num = args[6]
    # print(f"Source Dataframe Path: {source_df_path}")
    # print(f"Source Unprocessed Builds Directory: {source_builds_dir}")
    # print(f"Processed Builds Directory: {processed_builds_folder}")
    # print(f"Starting Index: {start_index}")
    # print(f"Ending Index: {end_index}")
    # print(f"Batch Number: {batch_num}")

    # config = world2vecDriverConfig(DOWNLOADED_BUILDS_FOLDER=source_builds_dir, PROCESSED_BUILDS_FOLDER=processed_builds_folder)
    # world2vecdriver = world2vecDriver(cfg=config)

    # world2vecdriver.process_batch(source_df_path, start_index, end_index, batch_num)

    # Code to test the driver manually

    config = world2vecDriverConfig(
        DOWNLOADED_BUILDS_FOLDER=r"C:\Projects\text2mc\builds",
        PROCESSED_BUILDS_FOLDER=r"C:\Projects\text2mc\processed_builds",
        BLOCK_JSON_PATH=r"C:\Projects\text2mc\SCRUM-whatever\text2mc-dataprocessor\world2vec\block_ints.json",
    )
    world2vecdriver = world2vecDriver(cfg=config)

    # world2vecdriver.process_batch(
    #     dataframe_path=r"C:\Users\shaun\OneDrive\Desktop\personal\CS classes\CS classes\COP4934\text2mc\text2mc-dataprocessor\projects_df_processed.csv",
    #     start_index=0,
    #     end_index=100,
    #     batch_num=1,
    # )

    projects_df = pd.read_csv(
        r"C:\Projects\text2mc\SCRUM-whatever\text2mc-dataprocessor\projects_df_processed.csv"
    )

    num_to_process = 20

    # Process .schem files
    print("Processings .schem files")
    schem_df = projects_df[projects_df["SUFFIX"] == ".schem"]
    for i, row in schem_df[0:num_to_process].iterrows():
        world2vecdriver.process_build(
            row["FILENAME"],
            f"schem_test_{i}",
            r"C:\Projects\text2mc\SCRUM-whatever\text2mc-dataprocessor\world2vec",
            straight_to_hdf5=True,
        )
    """
    # Process a single .zip archive
    # print("Processing .zip files")
    # zip_df = projects_df[projects_df["SUFFIX"] == ".zip"]
    # for i, row in zip_df[0:num_to_process].iterrows():
    #     world2vecdriver.process_build(row["FILENAME"], f"zip_test_{i}", r"C:\Projects\text2mc\SCRUM-whatever\text2mc-dataprocessor\world2vec")

    # Process a single .schematic file
    print("Processing .schematic files")
    schematic_df = projects_df[projects_df["SUFFIX"] == ".schematic"]
    for i, row in schematic_df[0:num_to_process].iterrows():
        world2vecdriver.process_build(
            row["FILENAME"], f"schematic_test_{row["FILENAME"]}", r"C:\Projects\text2mc\SCRUM-whatever\text2mc-dataprocessor\world2vec", straight_to_hdf5=True
        )

    # Process a single .rar archive
    print("Processing .rar files")
    rar_df = projects_df[projects_df["SUFFIX"] == ".rar"]
    for i, row in rar_df[0:num_to_process].iterrows():
        world2vecdriver.process_build(
            row["FILENAME"],
            f"schematic_test_{i}",
            "C:\Projects\text2mc\SCRUM-whatever\text2mc-dataprocessor\world2vec",
            straight_to_hdf5=True,
        )
        """


if __name__ == "__main__":
    main()
