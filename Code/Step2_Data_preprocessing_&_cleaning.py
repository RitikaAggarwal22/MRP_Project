#Step 2: Data Preprocessing & Cleaning

import pandas as pd
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
clean_dir = os.path.join(base_dir, '..', 'Cleaned Datasets')

master_df = pd.read_csv(os.path.join(clean_dir, 'master_outbreak_dataset.csv'))

# Fill missing values in main columns using backup columns
master_df['Causative Agent-1'] = master_df['Causative Agent-1'].fillna(master_df.get('Causative Agent - 1'))
master_df['Causative Agent-2'] = master_df['Causative Agent-2'].fillna(master_df.get('Causative Agent - 2'))

# Drop duplicate columns if they exist
columns_to_drop = ['Causative Agent - 1', 'Causative Agent - 2']
master_df = master_df.drop(columns=[col for col in columns_to_drop if col in master_df.columns])

# Reset `_id` to be a consistent, unique index starting from 1
master_df.reset_index(drop=True, inplace=True)
master_df['_id'] = master_df.index + 1 

# Convert date columns to datetime format
master_df['Date Outbreak Began'] = pd.to_datetime(master_df['Date Outbreak Began'], errors='coerce')
master_df['Date Declared Over'] = pd.to_datetime(master_df['Date Declared Over'], errors='coerce')
master_df[['Date Outbreak Began', 'Date Declared Over']].dtypes

# Calculate outbreak duration
master_df['Outbreak Duration (days)'] = (master_df['Date Declared Over'] - master_df['Date Outbreak Began']).dt.days
master_df[['Date Outbreak Began', 'Date Declared Over', 'Outbreak Duration (days)']].head()

# Fill missing values
median_began = master_df.groupby('Year')['Date Outbreak Began'].transform('median')
median_over  = master_df.groupby('Year')['Date Declared Over'].transform('median')

# Global medians
global_began = master_df['Date Outbreak Began'].median()
global_over  = master_df['Date Declared Over'].median()

master_df['Date Outbreak Began'] = (master_df['Date Outbreak Began'].fillna(median_began).fillna(global_began))
current_date = pd.Timestamp.now().normalize()

mask_active   = master_df['Active'].str.lower() == 'y'
mask_inactive = master_df['Active'].str.lower() == 'n'

master_df.loc[mask_active, 'Date Declared Over'] = master_df.loc[mask_active, 'Date Declared Over'].fillna(current_date)
master_df.loc[mask_inactive, 'Date Declared Over'] = master_df.loc[mask_inactive, 'Date Declared Over'].fillna(median_over)
master_df['Date Declared Over'] = master_df['Date Declared Over'].fillna(global_over)

# Recompute duration (at least 1 day)
master_df['Outbreak Duration (days)'] = (master_df['Date Declared Over'] - master_df['Date Outbreak Began']).dt.days.clip(lower=1)

# Fill Causative Agent-2
master_df['Causative Agent-2'] = master_df['Causative Agent-2'].fillna('N/A')

master_df['Date Outbreak Began']  = master_df['Date Outbreak Began'].dt.date
master_df['Date Declared Over']   = master_df['Date Declared Over'].dt.date

print(master_df[['Date Outbreak Began','Date Declared Over','Outbreak Duration (days)']].isnull().sum())

# Standardize categorical fields
cat_cols = ['Outbreak Setting', 'Type of Outbreak', 'Active']

for col in cat_cols:
    master_df[col] = master_df[col].str.strip().str.lower().astype('category')

for col in cat_cols:
      print(f"{col} categories: {master_df[col].cat.categories.tolist()}")

print("\nData types after standardization:")
print(master_df[cat_cols].dtypes)

# Drop duplicate rows
print("Before dropping duplicates:", master_df.shape)
master_df = master_df.drop_duplicates()

print("After dropping duplicates:", master_df.shape)

master_df.to_csv(os.path.join(clean_dir, 'preprocessed_dataset.csv'), index=False)