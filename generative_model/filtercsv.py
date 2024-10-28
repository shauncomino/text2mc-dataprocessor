import os
import pandas as pd

# Step 1: Read the CSV file
csv_file_path = "/lustre/fs1/groups/jaedo/all_version_2.csv"  # Converted to Unix path

df = pd.read_csv(csv_file_path)

# Remove duplicates, keeping only rows with a non-empty 'PROCESSED_PATH' column
# Subset specifies columns to consider for duplicates; keep=False to mark all duplicates
# Filter on 'PROCESSED_PATH' to drop rows without a path
df_cleaned = df[df.duplicated(subset=['PAGE_URL', 'DOWNLOAD_URL', 'FILENAME'], keep=False) & df['PROCESSED_PATHS'].notna()]

# Step 3: Save the modified DataFrame to a new CSV file
output_csv_file_path = "/lustre/fs1/groups/jaedo/filtered.csv"

# Ensure the directory exists
os.makedirs(os.path.dirname(output_csv_file_path), exist_ok=True)

# Save the cleaned data back to a CSV
df_cleaned.to_csv(output_csv_file_path, index=False)
