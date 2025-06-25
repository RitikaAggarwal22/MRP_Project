
"""Step 1: Data Collection, Integration & Initial Setup
"""

import pandas as pd
import os

# Define input and output directories
input_folder = 'Dataset'
output_folder = 'Cleaned Datasets'

file_names  = [
    'ob_report_2016.csv',
    'ob_report_2017.csv',
    'ob_report_2018.csv',
    'ob_report_2019.csv',
    'ob_report_2020.csv',
    'ob_report_2021.csv',
    'ob_report_2022.csv',
    'ob_report_2023.csv',
    'ob_report_2024.csv',
    'ob_report_2025.csv'
]

# Read each file, add 'Year' column, and append to list
dataframes = []

for file in file_names:
    file_path = os.path.join(input_folder, file)
    year = int(file.split('_')[2].split('.')[0])  # Extract year from filename
    df = pd.read_csv(file_path)
    df['Year'] = year
    dataframes.append(df)

master_df = pd.concat(dataframes, ignore_index=True)

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

output_path = os.path.join(output_folder, 'master_outbreak_dataset.csv')
master_df.to_csv(output_path, index=False)

print("Master dataset saved as 'master_outbreak_dataset.csv'")
print("Shape:", master_df.shape)
print("Columns:", master_df.columns.tolist())