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
from typing import Optional
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
            print("Processed build folder not passed or incorrect, creating a new folder")
            self.PROCESSED_BUILDS_FOLDER = os.path.join(
                os.path.abspath("./"), "vectorized_builds"
            )

        os.makedirs(self.PROCESSED_BUILDS_FOLDER, exist_ok=True)


class world2vecDriver:
    
    def __init__(self, cfg: world2vecDriverConfig = None):
        self.cfg = cfg

    def process_batch(self, 
        dataframe_path: str = None, start_index: int = None, end_index: int = None, batch_num: int = None
    ):
        temp_dir_path = os.path.join(os.path.dirname(self.cfg.PROCESSED_BUILDS_FOLDER), str("temp" + str(batch_num)))
        self.create_directory(temp_dir_path)
        dataframe = pd.read_csv(dataframe_path, on_bad_lines="warn")
        dataframe["PROCESSED_PATHS"] = pd.Series()
        for i, row in dataframe.iterrows():
            try:
                unprocessed_build_path = os.path.join(self.cfg.DOWNLOADED_BUILDS_FOLDER, row["FILENAME"])

                if row["FILENAME"].endswith(".zip") or row["FILENAME"].endswith(".rar"):
                    self.extract_archive_to_temporary_directory(unprocessed_build_path, temp_dir_path)

                    if (len(glob.glob(os.path.join(temp_dir_path, '/**/*.mca'), recursive=True)) > 0):
                        # .mca files means it's a standard Minecraft save
                        schem_paths = self.convert_build_to_schemfile(temp_dir_path, f"build_{i}_{batch_num}")
                        row["PROCESSED_PATHS"] = schem_paths

                    elif (len(glob.glob(os.path.join(temp_dir_path, '/**/*.schem'), recursive=True)) > 0):
                        # Handles the case when there are 0 or more .schem files in the .zip archive
                        row["PROCESSED_PATHS"] = glob.glob(os.path.join(temp_dir_path, '/**/*.schem'), recursive=True)

                    self.delete_directory_contents(temp_dir_path)

                elif row["FILENAME"].endswith(".schem"):
                    row["PROCESSED_PATHS"] = row["FILENAME"]
                    # schem_name = row["FILENAME"].split(".")[0]
                    # self.convert_schemfile_to_hdf5(
                    #     self, unprocessed_build_path, temp_dir_path, os.path.join(temp_dir_path, schem_name + ".json"), 
                    #     str("build_" + str(batch_num)))
                    self.delete_directory_contents(temp_dir_path)

                else:
                    row["PROCESSED_PATHS"] = None
                    continue             

            except Exception as e:
                print(f"Error processing build {row["FILENAME"]}")
                print(e)
                print(traceback.format_exc())
            
    def extract_archive_to_temporary_directory(self, source_archive_path: str = None, outfolder_path: str = None):
        Archive(source_archive_path).extractall(outfolder_path)
        
    def delete_directory_contents(self, folder: str = None):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        
    def convert_build_to_schemfile(self, folder_or_build_path, processed_file_prefix):
        regions_dir = World2Vec.find_regions_dir(folder_or_build_path)[0]
        return World2Vec.get_build(regions_dir, self.cfg.PROCESSED_BUILDS_FOLDER, processed_file_prefix)  # function needs to be rewritten, but will eventually work like this

    def convert_schemfile_to_json(self, schem_file_path: str, json_export_directory: str):
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
        hdf5_file = h5py.File(f"{processed_file_prefix}.h5", 'w')
        
    def convert_schemfile_to_hdf5(self, schem_file_path, json_export_directory, json_file_path, processed_file_prefix):
        self.convert_schemfile_to_json(self, schem_file_path, json_export_directory)
        vector = self.convert_json_to_npy(self, json_file_path)
        self.convert_vector_to_hdf5(self, vector, processed_file_prefix)

    def file_exists(self, dir_name: str = None, f_name: str = None): 
        return os.path.exists(dir_name) or os.path.exists(f_name)
        
    # Used to create processed and temporary directories
    def create_directory(self, dir_path: str):
        try:
            if (self.file_exists(dir_path)):
                return
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

    config = world2vecDriverConfig(DOWNLOADED_BUILDS_FOLDER=r'D:\builds', PROCESSED_BUILDS_FOLDER=r'D:\processed_schems')
    world2vecdriver = world2vecDriver(cfg=config)

    world2vecdriver.process_batch(dataframe_path=r'C:\Users\shaun\OneDrive\Desktop\personal\CS classes\CS classes\COP4934\text2mc\text2mc-dataprocessor\projects_df_processed.csv', start_index=0, end_index=100, batch_num=1)

if __name__ == "__main__":
    main()
