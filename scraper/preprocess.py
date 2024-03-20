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


@dataclass
class PreprocessorConfig:
    CSV_FILE_PATH: str = None
    """ CSV file path."""

    DOWNLOADED_BUILDS_DIRECTORY: str = None
    """ Directory that contains all the heterogenous build paths """

    def __post_init__(self):

        if not self.CSV_FILE_PATH or not os.path.exists(self.CSV_FILE_PATH):
            raise Exception(
                f"Must provide correct projects.csv filepath, got: {self.CSV_FILE_PATH}"
            )


class Preprocessor:

    def __init__(self, cfg: PreprocessorConfig = None):
        self.cfg = cfg
        self.projects_df = pd.read_csv(self.cfg.CSV_FILE_PATH)

    def preprocess_csv(self):
        self.projects_df = self.projects_df.dropna(subset=["RAW_DOWNLOAD_LINK"])

    def save_preprocessed(self):
        print("Writing processed CSV to new filepath")
        new_path = os.path.join(
            os.path.dirname(self.cfg.CSV_FILE_PATH), "projects_df_processed.csv"
        )
        self.projects_df.to_csv(new_path, index=False)


def main():
    config = PreprocessorConfig(
        CSV_FILE_PATH=r"C:\Users\shaun\OneDrive\Desktop\personal\CS classes\CS classes\COP4934\text2mc\text2mc-dataprocessor\projects.csv"
    )

    preprocessor = Preprocessor(cfg=config)
    preprocessor.preprocess_csv()
    preprocessor.save_preprocessed()


if __name__ == "__main__":
    main()
