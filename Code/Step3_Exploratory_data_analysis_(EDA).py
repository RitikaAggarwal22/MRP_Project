#Step 3: Exploratory Data Analysis (EDA)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Folder paths
base_dir = os.path.dirname(os.path.abspath(__file__))
clean_dir = os.path.join(base_dir, '..', 'Cleaned Datasets')
eda_dir   = os.path.join(base_dir, '..', 'EDA Outputs')

os.makedirs(eda_dir, exist_ok=True)

df = pd.read_csv(os.path.join(clean_dir, 'preprocessed_dataset.csv'))

# Inspect overall structure
print(" DataFrame Information")
print(df.info())

print("\n\n First 5 Rows")
print(df.head())

print("\n\n Missing Values per Column")
print(df.isnull().sum())

print("\n\n Data Types")
print(df.dtypes)

# Summary Statistics
print("\n\n Numeric Summary for Outbreak Duration (days)")
print(df['Outbreak Duration (days)'].describe())

print("\n\n Skewness:", df['Outbreak Duration (days)'].skew())
print("\nKurtosis:", df['Outbreak Duration (days)'].kurtosis())

# Categorical distributions for key fields
cat_cols = ['Outbreak Setting', 'Type of Outbreak', 'Active', 'Year']
for col in cat_cols:
    print(f"\nValue Counts for {col}")
    print(df[col].value_counts().sort_index())

# Univariate Visualizations
# Histogram of Outbreak Duration
plt.figure(figsize=(8, 4))
ax=sns.histplot(df['Outbreak Duration (days)'], bins=30, kde=True)
plt.title('Distribution of Outbreak Duration (days)')
plt.xlabel('Duration (days)')
plt.ylabel('Count')
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.text(p.get_x() + p.get_width()/2, height + 1, int(height), ha='center', va='bottom', fontsize=8)
plt.savefig(os.path.join(eda_dir, 'hist_outbreak_duration.png'))
plt.show()

# Boxplot of Outbreak Duration
plt.figure(figsize=(8, 3))
ax=sns.boxplot(x=df['Outbreak Duration (days)'])
plt.title('Boxplot of Outbreak Duration (days)')
plt.xlabel('Duration (days)')
median = df['Outbreak Duration (days)'].median()
ax.text(median, 0.6, f'Median = {median:.0f}', ha='center', va='center', color='white', weight='bold')
plt.savefig(os.path.join(eda_dir, 'box_outbreak_duration.png'))
plt.show()

# Bar plots for categorical variables
for col in ['Outbreak Setting', 'Type of Outbreak', 'Active', 'Year']:
    plt.figure(figsize=(10, 4))
    order = df[col].value_counts().index
    ax=sns.countplot(data=df, x=col, order=order)
    plt.title(f'Count of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2, height + 1, int(height), ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, f'count_{col.lower().replace(" ", "_")}.png'))
    plt.show()

# Bivariate Analyses
# Scatter plot: Outbreak Duration vs Year
plt.figure(figsize=(8, 4))
ax=sns.scatterplot(data=df, x='Year', y='Outbreak Duration (days)', alpha=0.6)
plt.title('Outbreak Duration vs Year')
plt.xlabel('Year')
plt.ylabel('Duration (days)')
for x, y in zip(df['Year'], df['Outbreak Duration (days)']):
    ax.text(x, y, f'{int(y)}', fontsize=6, alpha=0.6, ha='center', va='bottom')
plt.savefig(os.path.join(eda_dir, 'scatter_duration_vs_year.png'))
plt.show()

# Box plot: Outbreak Duration by Type of Outbreak
plt.figure(figsize=(8, 4))
ax=sns.boxplot(data=df, x='Type of Outbreak', y='Outbreak Duration (days)')
plt.title('Duration by Outbreak Type')
plt.xlabel('Type of Outbreak')
plt.ylabel('Duration (days)')
medians = df.groupby('Type of Outbreak')['Outbreak Duration (days)'].median()
for i, (cat, m) in enumerate(medians.items()):
    ax.text(i, m, f'{int(m)}', ha='center', va='bottom', color='white', weight='bold')
plt.savefig(os.path.join(eda_dir, 'box_duration_by_type.png'))
plt.show()

# Box plot: Outbreak Duration by Active Status
plt.figure(figsize=(6, 4))
ax=sns.boxplot(data=df, x='Active', y='Outbreak Duration (days)')
plt.title('Duration by Active Status')
plt.xlabel('Active (n/y)')
plt.ylabel('Duration (days)')

medians_active = df.groupby('Active')['Outbreak Duration (days)'].median()
for i, (cat, m) in enumerate(medians_active.items()):
    ax.text(i, m, f'{int(m)}', ha='center', va='bottom', color='white', weight='bold')
plt.savefig(os.path.join(eda_dir, 'box_duration_by_active.png'))
plt.show()

# Count plot: Outbreak Setting by Type of Outbreak
plt.figure(figsize=(12, 5))
ax=sns.countplot(data=df, x='Outbreak Setting', hue='Type of Outbreak')
plt.title('Count of Outbreaks by Setting and Type')
plt.xlabel('Outbreak Setting')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Type of Outbreak')
plt.tight_layout()

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2, height + 1, int(height), ha='center', va='bottom')
plt.savefig(os.path.join(eda_dir, 'count_setting_by_type.png'))
plt.show()

# Correlation Analysis for Numeric Features
# Compute correlation matrix for numeric columns
num_cols = ['Year', 'Outbreak Duration (days)']
corr_matrix = df[num_cols].corr()

print("Correlation Matrix")
print(corr_matrix)

# Plot heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix for Numeric Features')
plt.savefig(os.path.join(eda_dir, 'heatmap_correlation.png'))
plt.show()

# Pairplot for numeric features
sns.pairplot(df[num_cols], diag_kind='kde')
plt.suptitle("Pairplot of Numeric Features", y=1.02)
plt.savefig(os.path.join(eda_dir, 'pairplot_numeric_features.png'))
plt.show()

# Outlier Detection for Outbreak Duration
# Compute Q1, Q3 and IQR
q1 = df['Outbreak Duration (days)'].quantile(0.25)
q3 = df['Outbreak Duration (days)'].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print(f"Lower bound: {lower_bound:.1f} days")
print(f"Upper bound: {upper_bound:.1f} days")

# Identify outliers
outliers = df[(df['Outbreak Duration (days)'] < lower_bound) | (df['Outbreak Duration (days)'] > upper_bound)]

print(f"Number of outlier records: {outliers.shape[0]}")

print("\nSample outliers:")
print(outliers[['Year','Outbreak Setting','Type of Outbreak','Outbreak Duration (days)']].head())

# Group Summaries
# Mean and median duration by Type of Outbreak
print(" Duration by Type of Outbreak ")
print(df.groupby('Type of Outbreak')['Outbreak Duration (days)'].agg(['count','mean','median','std']).sort_values('mean', ascending=False))

# Mean and median duration by Outbreak Setting
print("\n Duration by Outbreak Setting ")
print(df.groupby('Outbreak Setting')['Outbreak Duration (days)'].agg(['count','mean','median','std']).sort_values('mean', ascending=False))

# Pivot table: average duration for each Setting × Type
pivot = df.pivot_table(
    index='Outbreak Setting',
    columns='Type of Outbreak',
    values='Outbreak Duration (days)',
    aggfunc='mean'
)
print("\n Pivot Table: Avg Duration (days) by Setting Type")
print(pivot)

# Temporal & Geospatial Patterns
# Ensure date columns are datetime
df['Date Outbreak Began'] = pd.to_datetime(df['Date Outbreak Began'], errors='coerce')
df['Date Declared Over']  = pd.to_datetime(df['Date Declared Over'], errors='coerce')

# Temporal Patterns: Monthly outbreak counts
df['Year-Month'] = df['Date Outbreak Began'].dt.to_period('M').dt.to_timestamp()
monthly_counts = df.groupby('Year-Month').size().reset_index(name='Count')

plt.figure(figsize=(10, 4))
plt.plot(monthly_counts['Year-Month'], monthly_counts['Count'], marker='o')
plt.title('Monthly Outbreak Counts')
plt.xlabel('Month')
plt.ylabel('Number of Outbreaks')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(eda_dir, 'monthly_outbreak_counts.png'))
plt.show()

# Geospatial: Top 10 Institutions by outbreak frequency
top_institutions = df['Institution Name'].value_counts().head(10)

plt.figure(figsize=(8, 5))
sns.barplot(x=top_institutions.values, y=top_institutions.index,hue=top_institutions.index, palette='viridis',legend=False)
plt.title('Top 10 Institutions by Number of Outbreaks')
plt.xlabel('Outbreak Count')
plt.ylabel('Institution Name')
plt.tight_layout()
plt.savefig(os.path.join(eda_dir, 'top10_institutions.png'))
plt.show()

# Recompute IQR bounds for Outbreak Duration
q1 = df['Outbreak Duration (days)'].quantile(0.25)
q3 = df['Outbreak Duration (days)'].quantile(0.75)
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr

# Top 10 longest outbreak durations
print(" Top 10 Longest Outbreaks ")
print(df.nlargest(10, 'Outbreak Duration (days)')[['Year', 'Institution Name', 'Outbreak Setting', 'Type of Outbreak', 'Outbreak Duration (days)']])

# Active vs Resolved counts
print("\n Active vs Resolved ")
print(df['Active'].value_counts())

# Number of extreme outliers (> upper IQR bound)
outlier_count = df[df['Outbreak Duration (days)'] > upper_bound].shape[0]
print(f"\nNumber of outliers (duration > {upper_bound:.1f} days): {outlier_count}")

# Very long outbreaks (> 100 days)
long_100 = df[df['Outbreak Duration (days)'] > 100]
print(f"\nOutbreaks > 100 days: {long_100.shape[0]}")
print(long_100[['Year','Institution Name','Outbreak Duration (days)']].head())

# Median duration by Active status
print("\n Median Duration by Active Status ")
print(df.groupby('Active')['Outbreak Duration (days)'].median())

# Top 5 primary causative agents
print("\n Top 5 Primary Agents ")
print(df['Causative Agent-1'].value_counts().head(5))