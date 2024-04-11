from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import pandas as pd
from openai import OpenAI
import time
from pandas import DataFrame
import os, shutil
from dataclasses import dataclass, field
from typeguard import typechecked
from typing import Optional, List
import requests
from tqdm import tqdm
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
from pyunpack import Archive
from unrar import rarfile


@dataclass
class world2vecDriverConfig:
    DOWNLOADED_BUILDS_FOLDER: str = None
    """ Downloaded builds """

    PROCESSED_BUILDS_FOLDER: str = None
    """ Processed .hdf5 builds """

    def __post_init__(self):

        if not self.DOWNLOADED_BUILDS_FOLDER or not os.path.exists(
            self.DOWNLOADED_BUILDS_FOLDER
        ):
            print("Invalid build source folder or not provided")

        if self.PROCESSED_BUILDS_FOLDER is None or not os.path.exists(
            self.PROCESSED_BUILDS_FOLDER
        ):
            print(
                "Processed build folder not passed or incorrect, creating a new folder"
            )
            self.PROCESSED_BUILDS_FOLDER = os.path.join(
                os.path.abspath("./"), "vectorized_builds"
            )

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
                filename = row["FILENAME"]
                processed_paths = self.process_build(
                    filename,
                    processed_file_name=f"batch_{batch_num}_{str(i)}",
                    temp_dir_path=temp_dir_path,
                )
                dataframe.at[i, "PROCESSED_PATHS"] = processed_paths
            except Exception as e:
                print(e)
                traceback.format_exc()

    def process_build(
        self,
        filename: str,
        processed_file_name: str = "temp_schem",
        temp_dir_path: str = "temp",
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

                if (
                    len(
                        glob.glob(
                            os.path.join(temp_dir_path, "/**/*.mca"), recursive=True
                        )
                    )
                    > 0
                ):
                    schem_paths = self.convert_build_to_schemfile(
                        temp_dir_path, f"build_{processed_file_name}"
                    )
                    processed_paths = schem_paths

                elif (
                    len(
                        glob.glob(
                            os.path.join(temp_dir_path, "/**/*.schem"), recursive=True
                        )
                    )
                    > 0
                ):
                    processed_paths = glob.glob(
                        os.path.join(temp_dir_path, "/**/*.schem"), recursive=True
                    )
                elif len(
                    glob.glob(
                        os.path.join(temp_dir_path, "/**/*.schematic"), recursive=True
                    )
                ):
                    processed_paths = glob.glob(
                        os.path.join(temp_dir_path, "/**/*.schematic"), recursive=True
                    )

                self.delete_directory_contents(temp_dir_path)

            elif filename.endswith(".schem"):
                processed_paths = [filename]
            else:
                # If the file format is not supported, return an empty list
                return processed_paths

        except Exception as e:
            print(f"Error processing build {filename}: {e}")
            traceback.print_exc()

        return processed_paths

    def extract_archive_to_temporary_directory(
        self, source_archive_path: str = None, outfolder_path: str = None
    ):
        if source_archive_path.endswith(".zip"):
            # For .zip files, use Archive from pyunpack
            Archive(source_archive_path).extractall(outfolder_path)
        elif source_archive_path.endswith(".rar"):
            # For .rar files, use rarfile
            with rarfile.RarFile(source_archive_path) as rf:
                rf.extractall(outfolder_path)

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
            natural_blocks_path=r"C:\Users\shaun\OneDrive\Desktop\personal\CS classes\CS classes\COP4934\text2mc\text2mc-dataprocessor\world2vec\natural_blocks.txt",
        )

    def convert_schemfile_to_json(
        self, schem_file_path: str, json_export_directory: str
    ):
        subprocess.call(
            [
                "java",
                "-jar",
                "schematic-loader.jar",
                schem_file_path,
                json_export_directory,
            ]
        )

    def convert_json_to_npy(self, json_file_path) -> np.ndarray:
        return World2Vec.export_json_to_npy(json_file_path)

    def convert_vector_to_hdf5(self, vector, processed_file_prefix):
        hdf5_out_path = os.path.join(
            self.cfg.PROCESSED_BUILDS_FOLDER, f"{processed_file_prefix}.h5"
        )
        hdf5_file = h5py.File(f"{processed_file_prefix}.h5", "w")

    def convert_schemfile_to_hdf5(
        self,
        schem_file_path,
        json_export_directory,
        json_file_path,
        processed_file_prefix,
    ):
        self.convert_schemfile_to_json(self, schem_file_path, json_export_directory)
        vector = self.convert_json_to_npy(self, json_file_path)
        self.convert_vector_to_hdf5(self, vector, processed_file_prefix)

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
        DOWNLOADED_BUILDS_FOLDER=r"D:\builds",
        PROCESSED_BUILDS_FOLDER=r"D:\processed_schems",
    )
    world2vecdriver = world2vecDriver(cfg=config)

    # world2vecdriver.process_batch(
    #     dataframe_path=r"C:\Users\shaun\OneDrive\Desktop\personal\CS classes\CS classes\COP4934\text2mc\text2mc-dataprocessor\projects_df_processed.csv",
    #     start_index=0,
    #     end_index=100,
    #     batch_num=1,
    # )

    projects_df = pd.read_csv(
        r"C:\Users\shaun\OneDrive\Desktop\personal\CS classes\CS classes\COP4934\text2mc\text2mc-dataprocessor\projects_df_processed.csv"
    )

    num_to_process = 5

    # # Process a single .schem file
    # print("Processings .schem files")
    # schem_df = projects_df[projects_df["SUFFIX"] == ".schem"]
    # for i, row in schem_df[0:num_to_process].iterrows():
    #     world2vecdriver.process_build(row["FILENAME"], f"schem_test_{i}", r"D:\\temp")

    # Process a single .zip archive
    print("Processing .zip files")
    zip_df = projects_df[projects_df["SUFFIX"] == ".zip"]
    for i, row in zip_df[0:num_to_process].iterrows():
        world2vecdriver.process_build(row["FILENAME"], f"zip_test_{i}", r"D:\\temp")

    # # Process a single .schematic file
    # print("Processing .schematic files")
    # schematic_df = projects_df[projects_df["SUFFIX"] == ".schematic"]
    # for i, row in schematic_df[0:num_to_process].iterrows():
    #     world2vecdriver.process_build(
    #         row["FILENAME"], f"schematic_test_{i}", r"D:\\temp"
    #     )

    # Process a single .rar archive
    # print("Processing .rar files")
    # rar_df = projects_df[projects_df["SUFFIX"] == ".rar"]
    # for i, row in rar_df[0:num_to_process].iterrows():
    #     world2vecdriver.process_build(
    #         row["FILENAME"], f"schematic_test_{i}", r"D:\\temp"
    #     )


if __name__ == "__main__":
    main()
