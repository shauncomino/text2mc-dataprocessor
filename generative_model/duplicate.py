import pandas as pd

# Step 2: Read the CSV file into a DataFrame
df = pd.read_csv('/mnt/c/Users/grees/OneDrive/Desktop/SD1/text2mc-dataprocessor/projects_filtered.csv')
# Step 3: Identify duplicate rows
duplicates = df[df.duplicated()]

# Step 4: Print or handle the duplicate rows
if not duplicates.empty:
    print("Duplicate rows found:")
    print(duplicates)
else:
    print("No duplicate rows found.")