import pandas as pd
import ast

# Load the CSV file
df = pd.read_csv('metrics.csv')

# Convert string representations of lists to actual lists
df['F1_Score'] = df['F1_Score'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df['PROCESSED_PATHS'] = df['PROCESSED_PATHS'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Check a sample of the data to verify format
print("Sample F1_Score column data:", df['F1_Score'].head())
print("Sample PROCESSED_PATHS column data:", df['PROCESSED_PATHS'].head())

# Ensure F1_Score and PROCESSED_PATHS are lists and calculate highest scores only for valid rows
def get_highest_score_and_path(row):
    if isinstance(row['F1_Score'], list) and isinstance(row['PROCESSED_PATHS'], list):
        highest_score = max(row['F1_Score'])
        best_file_name = row['PROCESSED_PATHS'][row['F1_Score'].index(highest_score)]
        return pd.Series([highest_score, best_file_name])
    return pd.Series([None, None])

# Apply the function to get Highest_Score and Best_File_Name columns
df[['Highest_Score', 'Best_File_Name']] = df.apply(get_highest_score_and_path, axis=1)

# Filter out rows where calculations could not be performed
df = df.dropna(subset=['Highest_Score', 'Best_File_Name'])

# Sort by the highest score in descending order
df = df.sort_values(by='Highest_Score', ascending=False)

# Save to a new CSV or display the result
df.to_csv('sorted_by_highest_score.csv', index=False)
print("Final result:", df[['Highest_Score', 'Best_File_Name']].head())