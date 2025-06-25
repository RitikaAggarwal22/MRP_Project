
"""Step 4: Feature Engineering
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv(
    'Cleaned Datasets/preprocessed_dataset.csv',
    parse_dates=['Date Outbreak Began', 'Date Declared Over']
)

# Temporal features
df['Begin Month']   = df['Date Outbreak Began'].dt.month
df['Begin Weekday'] = df['Date Outbreak Began'].dt.weekday

def month_to_season(m):
    if m in (12, 1, 2): return 'Winter'
    if m in (3, 4, 5):  return 'Spring'
    if m in (6, 7, 8):  return 'Summer'
    return 'Fall'

df['Begin Season']  = df['Begin Month'].apply(month_to_season)

# Severity labeling
def categorize_severity(d):
    if d <= 7:   return 'Mild'
    if d <= 21:  return 'Moderate'
    return 'Severe'

df['Severity'] = df['Outbreak Duration (days)'].apply(categorize_severity).astype('category')

# Encode severity to numeric
le = LabelEncoder()
df['Severity_Label'] = le.fit_transform(df['Severity'])

print("=== Severity Distribution ===")
print(df['Severity'].value_counts().sort_index())

print("\n=== Severity Mapping ===")
print(dict(zip(le.classes_, le.transform(le.classes_))))

# Create binary flags on the raw df BEFORE one-hot encoding
df['is_covid1'] = df['Causative Agent-1'].str.contains('covid', case=False, na=False).astype(int)
df['is_covid2'] = df['Causative Agent-2'].str.contains('covid', case=False, na=False).astype(int)
df['is_ltch']   = (df['Outbreak Setting'] == 'ltch').astype(int)

print("\nFlags added (sample):")
print(df[['is_covid1','is_covid2','is_ltch']].head())

# One-hot encode other categorical features
cat_cols = [
    'Outbreak Setting',
    'Type of Outbreak',
    'Active',
    'Begin Season',
    'Causative Agent-1',
    'Causative Agent-2'
]
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

print("\nEncoded DataFrame shape:", df_encoded.shape)
print("Sample agent dummies:",
      [c for c in df_encoded.columns if c.startswith('Causative Agent-1_')][:5],
      [c for c in df_encoded.columns if c.startswith('Causative Agent-2_')][:5])

# Define feature matrix X and target y
exclude = [
    '_id','Institution Name','Institution Address',
    'Date Outbreak Began','Date Declared Over',
    'Severity','Severity_Label'
]
feature_cols = [c for c in df_encoded.columns if c not in exclude]
X = df_encoded[feature_cols]
y = df_encoded['Severity_Label']

# Scale numeric features
numeric_cols = ['Outbreak Duration (days)','Begin Month','Begin Weekday']
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numeric_cols] = scaler.fit_transform(X_scaled[numeric_cols])

print("Scaled means:", X_scaled[numeric_cols].mean().round(4).to_dict())
print("Scaled stds: ", X_scaled[numeric_cols].std().round(4).to_dict())
print("Feature matrix shape:", X_scaled.shape)
print("Target distribution:", y.value_counts().to_dict())

# Combine X_scaled + y
processed_df = X_scaled.copy()
processed_df['Severity_Label'] = y.values

processed_df.to_csv('Cleaned Datasets/processed_outbreak_dataset.csv', index=False)
print("Saved 'processed_outbreak_dataset.csv'")

# Save scaler and label encoder
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'severity_label_encoder.pkl')
print("Saved 'scaler.pkl' and 'severity_label_encoder.pkl'")

