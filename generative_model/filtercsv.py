import os
import pandas as pd

# Step 1: Read the CSV file
csv_file_path = "/mnt/c/Users/grees/Downloads/merged_output.csv"  # Converted to Unix path

df = pd.read_csv(csv_file_path)

# Step 2: Identify and remove duplicates
# Keep only the first occurrence of each duplicate row based on 'FILENAME', and remove duplicates without paths
df['PROCESSED_PATHS'] = df['PROCESSED_PATHS'].apply(lambda x: [] if pd.isna(x) else x.strip("[]").replace("'", "").split(','))

# Identify duplicates based on the 'FILENAME' column
duplicates = df.duplicated(subset=['FILENAME'], keep=False)

# Remove duplicates that do not have a path in 'PROCESSED_PATHS'
df = df[~(duplicates & df['PROCESSED_PATHS'].apply(lambda x: len(x) == 0))]

# Step 3: Save the modified DataFrame to a new CSV file
output_csv_file_path = "/mnt/c/Users/grees/Downloads/filtered_output.csv"

# Ensure the directory exists
os.makedirs(os.path.dirname(output_csv_file_path), exist_ok=True)

df.to_csv(output_csv_file_path, index=False)