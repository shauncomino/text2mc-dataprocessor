import pandas as pd
import glob
import os
from zipfile import ZipFile
import shutil
import patoolib
from tqdm import tqdm
import traceback

df_path = r"C:\Users\shaun\OneDrive\Desktop\personal\CS classes\CS classes\COP4934\text2mc\text2mc-dataprocessor\projects_filtered.csv"
out_df_path = r"C:\Users\shaun\OneDrive\Desktop\personal\CS classes\CS classes\COP4934\text2mc\text2mc-dataprocessor\projects_unzipped.csv"
filtered_df = pd.read_csv(df_path)
filtered_df = filtered_df.reset_index(drop=True)
filtered_df["NEW_FILENAME"] = pd.Series()

predownloaded_builds_directory = "D:\\builds\\"
actually_downloaded = list(filtered_df["FILENAME"])

target_dir = "D:\\kept_builds\\"

j = 0
for i in tqdm(range(0, len(actually_downloaded)), desc="Unzipping and moving process"):
    new_dir = None

    try:
        path = os.path.join(predownloaded_builds_directory, actually_downloaded[i])
        file = os.path.split(path)[-1]
        if file.endswith(".zip"):
            prefix = f"build-{j}"
            new_dir = os.path.join(target_dir, prefix)
            os.mkdir(new_dir)
            ZipFile(path).extractall(new_dir)
            filtered_df.at[i, "NEW_FILENAME"] = prefix
            j += 1
        elif file.endswith(".rar"):
            prefix = f"build-{j}"
            patoolib.extract_archive(path, outdir=os.path.join(target_dir, prefix))
            filtered_df.at[i, "NEW_FILENAME"] = prefix
            j += 1
        else:
            # .schem/.schematic files, simply move them
            suffix = os.path.splitext(file)[-1]
            suffix_ = f"build-{j}.{suffix}"
            new_path = os.path.join(target_dir, suffix_)
            os.copyfile(path, new_path)
            filtered_df.at[i, "NEW_FILENAME"] = suffix_
            j += 1

        if i % 100:
            filtered_df.to_csv(out_df_path)

    except Exception as e:
        try:
            print(f"Error processing archive/file: {file}")
            print(e)
            print(traceback.format_exc())
            shutil.rmtree(new_dir)
        except Exception as e:
            print("Error removing erroneous folder")

filtered_df.to_csv(out_df_path)
