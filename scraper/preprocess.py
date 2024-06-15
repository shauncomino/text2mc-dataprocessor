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
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from ast import literal_eval
import glob


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
        self.projects_df["FILENAME"] = self.projects_df["RAW_DOWNLOAD_LINK"].apply(
            lambda x: os.path.split(x)[-1]
        )
        # Filter based on DOWNLOAD_SIZE <= 50MB and extract file extension
        mask = self.projects_df["DOWNLOAD_SIZE"] <= 50000000
        self.projects_df = self.projects_df[mask]

        self.projects_df["SUFFIX"] = self.projects_df["FILENAME"].apply(
            lambda x: os.path.splitext(x)[-1]
        )

        filetypes_mask = (
            (self.projects_df["SUFFIX"] == ".zip")
            | (self.projects_df["SUFFIX"] == ".rar")
            | (self.projects_df["SUFFIX"] == ".schematic")
            | (self.projects_df["SUFFIX"] == ".schem")
        )
        self.projects_df = self.projects_df[filetypes_mask]

        files_in_build_dir = glob.glob(
            os.path.join(self.cfg.DOWNLOADED_BUILDS_DIRECTORY, "*")
        )
        all_filenames = [os.path.basename(path) for path in files_in_build_dir]
        self.projects_df = self.projects_df[
            self.projects_df["FILENAME"].isin(all_filenames)
        ]

    def save_preprocessed(self):
        print("Writing processed CSV to new filepath")
        new_path = os.path.join(
            os.path.dirname(self.cfg.CSV_FILE_PATH), "projects_df_processed.csv"
        )
        self.projects_df.to_csv(new_path, index=False)

    def plot_stuff(self):
        self.projects_df["SUFFIX"] = self.projects_df["FILENAME"].apply(
            lambda x: os.path.splitext(x)[-1]
        )

        file_type_counts = (
            self.projects_df["SUFFIX"].value_counts(normalize=True) * 100
        )  # normalize=True gives the percentage

        # Plotting
        plt.figure(figsize=(10, 6))
        file_type_counts[0:9].plot(kind="bar")
        plt.title("Dataset Percentage of Each File Type")
        plt.ylabel("Percentage (%)")
        plt.xlabel("File Types")
        plt.xticks(rotation=45)  # Rotate file types for better readability

        # Filter for archives and schematic files
        archives_mask = (self.projects_df["SUFFIX"] == ".zip") | (
            self.projects_df["SUFFIX"] == ".rar"
        )
        archives_files_df = self.projects_df[archives_mask]
        schem_files_df = self.projects_df[
            (self.projects_df["SUFFIX"] == ".schem")
            | (self.projects_df["SUFFIX"] == ".schematic")
        ]

        # Removing outliers from both DataFrames
        archives_files_df_no_outliers = self.remove_outliers(
            archives_files_df, "DOWNLOAD_SIZE"
        )
        schem_files_df_no_outliers = self.remove_outliers(
            schem_files_df, "DOWNLOAD_SIZE"
        )

        # Plotting
        plt.figure(figsize=(10, 8))

        # Histogram for archive files without outliers
        plt.subplot(2, 1, 1)
        plt.hist(
            archives_files_df_no_outliers["DOWNLOAD_SIZE"],
            bins=110,
            color="skyblue",
            edgecolor="black",
        )
        plt.title("Build Sizes Histogram: .zip, .rar")
        plt.xlabel("Size in Bytes")
        plt.ylabel("Frequency")
        plt.gca().xaxis.set_major_formatter(
            ticker.ScalarFormatter(useOffset=False, useMathText=False)
        )
        plt.ticklabel_format(
            style="plain", axis="x"
        )  # This line turns off scientific notation

        # Histogram for schematic files without outliers
        plt.subplot(2, 1, 2)
        plt.hist(
            schem_files_df_no_outliers["DOWNLOAD_SIZE"],
            bins=110,
            color="salmon",
            edgecolor="black",
        )
        plt.title("Build Sizes Histogram: .schem")
        plt.xlabel("Size in Bytes")
        plt.ylabel("Frequency")
        plt.gca().xaxis.set_major_formatter(
            ticker.ScalarFormatter(useOffset=False, useMathText=False)
        )
        plt.ticklabel_format(
            style="plain", axis="x"
        )  # And this line does the same for the second plot

        plt.tight_layout()

        self.projects_df["TAGS"] = self.projects_df["TAGS"].apply(
            lambda x: literal_eval(x)
        )

        # Flatten the list of tags into a single list
        all_tags = [tag for sublist in self.projects_df["TAGS"] for tag in sublist]

        # Convert list to a Series and count values
        tag_counts = pd.Series(all_tags).value_counts()

        # Select the top 20 most common tags
        top_tags = tag_counts.head(30)
        top_tags = top_tags[1:]

        # Plotting
        plt.figure(figsize=(10, 8))
        top_tags.plot(kind="bar")
        plt.title("Top 20 Most Common Tags")
        plt.ylabel("Frequency")
        plt.xlabel("Tags")
        plt.xticks(rotation=45, ha="right")  # Rotate tags for better readability
        plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels

        # plt.show()

    # Function to remove outliers using IQR

    def remove_outliers(self, df, column_name):
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]


def main():
    config = PreprocessorConfig(
        CSV_FILE_PATH=r"C:\Users\shaun\OneDrive\Desktop\personal\CS classes\CS classes\COP4934\text2mc\text2mc-dataprocessor\projects.csv",
        DOWNLOADED_BUILDS_DIRECTORY=r"D:\builds",
    )

    preprocessor = Preprocessor(cfg=config)
    preprocessor.preprocess_csv()
    preprocessor.save_preprocessed()


if __name__ == "__main__":
    main()
