from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import pandas as pd
from openai import OpenAI
import time
import os
from dataclasses import dataclass, field
from typeguard import typechecked
from typing import Optional
import requests
from tqdm import tqdm
import re
import traceback
from world2vec import World2Vec


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
            print("Invalid or not provided build source folder")

        if self.PROCESSED_BUILDS_FOLDER is None or not os.path.exists(
            self.PROCESSED_BUILD_FOLDER
        ):
            self.PROCESSED_BUILDS_FOLDER = os.path.join(
                os.path.abspath("./"), "vectorized_builds"
            )

        os.makedirs(self.PROCESSED_BUILDS_FOLDER, exist_ok=True)


class world2vecDriver:

    def __init__(self, cfg: world2vecDriverConfig = None):
        self.cfg = cfg

    def convert_build_to_vector(self, folder_or_build_path, processed_file_prefix):
        pass

    def convert_vector_to_hdf5(self, vector, processed_file_prefix):
        hdf5_out_path = os.path.join(
            self.cfg.PROCESSED_BUILDS_FOLDER, f"{processed_file_prefix}.hdf5"
        )

    def convert_schemfile_to_vector():
        pass


def main():
    config = world2vecDriverConfig()

    world2vec_instance = world2vecDriver(cfg=config)


if __name__ == "__main__":
    main()
