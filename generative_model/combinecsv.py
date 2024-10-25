import os
import pandas as pd

# Define the folder containing the CSV files
folder_path = "/lustre/fs1/groups/jaedo/batch_csvs/"  # Replace with your actual path to the folder containing CSVs

# List to store data from each CSV
dataframes = []

# Loop over each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):  # Only process CSV files
        file_path = os.path.join(folder_path, filename)
        # Read the CSV into a DataFrame without a header (to maintain original structure)
        df = pd.read_csv(file_path, header=None)
        dataframes.append(df)

# Start by using the first CSV as the base for merging
merged_csv = dataframes[0]

# Loop through the remaining DataFrames and merge them
for df in dataframes[1:]:
    # Merge on all columns except the 'processed_paths' column (which I assume is the last column, adjust index if needed)
    merged_csv = pd.merge(merged_csv, df, how='outer', on=list(range(len(df.columns) - 1)), suffixes=('', '_extra'))

    # Fill missing values in the 'processed_paths' column (last column in both DataFrames)
    processed_col = len(merged_csv.columns) - 2  # Last column before the _extra
    processed_col_extra = len(merged_csv.columns) - 1  # Last column from the extra suffix

    merged_csv.iloc[:, processed_col] = merged_csv.iloc[:, processed_col].combine_first(merged_csv.iloc[:, processed_col_extra])

    # Drop the extra 'processed_paths' column after merging
    merged_csv.drop(columns=[merged_csv.columns[processed_col_extra]], inplace=True)

# Save the final merged CSV without modifying the original files
output_path = 'merged_output.csv'  # Set the output path for the final merged CSV
merged_csv.to_csv(output_path, index=False, header=False)

print(f"All CSVs have been merged and saved to '{output_path}'.")
