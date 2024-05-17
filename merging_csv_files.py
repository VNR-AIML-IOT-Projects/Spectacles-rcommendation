import pandas as pd
import glob

# File pattern to match CSV files
file_pattern = "Data_2/*.csv"  # Update with your CSV files' location
csv_files = glob.glob(file_pattern)

# List comprehension to load all CSV files into DataFrames
dfs = [pd.read_csv(file) for file in csv_files]

# Concatenate all DataFrames into one
merged_df = pd.concat(dfs, ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv("merged_file_2.csv", index=False)
