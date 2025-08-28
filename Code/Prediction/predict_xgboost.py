#Predict severity for May 2025–May 2026 using XGBoost

import os
import pandas as pd
import numpy as np
import joblib

base_dir   = os.path.dirname(os.path.abspath(__file__))
clean_dir  = os.path.join(base_dir, '..', 'Cleaned Datasets')
out_dir    = os.path.join(base_dir, '..', 'Output Files')
os.makedirs(out_dir, exist_ok=True)


Master_csv = os.path.join(clean_dir, 'master_outbreak_dataset.csv')
Train_processed = os.path.join(clean_dir, 'processed_outbreak_dataset.csv')
Scaler_pkl = os.path.join(out_dir, 'scaler.pkl')
Label_encoder_pkl = os.path.join(out_dir, 'severity_label_encoder.pkl')
Xgb_pkl = os.path.join(out_dir, 'xgboost.pkl')
Output_csv = os.path.join(out_dir, 'predictions_xgb_may2025_may2026.csv')

# Load master dataset and filter by target date range
df = pd.read_csv(Master_csv, parse_dates=False)
df["Date Outbreak Began"] = pd.to_datetime(df["Date Outbreak Began"], errors="coerce")

# Drop rows with invalid dates
invalidDateCount = df["Date Outbreak Began"].isna().sum()
if invalidDateCount:
    print(f"Warning: {invalidDateCount} rows have invalid dates and will be dropped.")
df = df.dropna(subset=["Date Outbreak Began"])

# Keep only rows within May 2025–May 2026
df = df[(df["Date Outbreak Began"] >= "2025-05-01") & (df["Date Outbreak Began"] <= "2026-05-31")].copy()
if df.empty:
    print("No rows in May 2025–May 2026. Nothing to predict.")
    raise SystemExit()

# Keep metadata columns to include in the output
metadataColumns = ["_id", "Institution Name", "Outbreak Setting", "Type of Outbreak", "Causative Agent-1", "Causative Agent-2", "Date Outbreak Began", "Outbreak Duration (days)"]
metadataColumns  = [col for col in metadataColumns if col in df.columns]
meta = df[metadataColumns].copy() if metadataColumns else pd.DataFrame(index=df.index)

# Feature engineering to match the training dataset
df["Begin Month"] = df["Date Outbreak Began"].dt.month
df["Begin Weekday"] = df["Date Outbreak Began"].dt.weekday

def month_to_season(month): 
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Spring"
    elif month in (6, 7, 8):
        return "Summer"
    else:
        return "Fall"
df["Begin Season"] = df["Begin Month"].apply(month_to_season)

# Create binary flags for COVID and LTCH settings
df["is_covid1"] = df.get("Causative Agent-1","").astype(str).str.contains("covid", case=False, na=False).astype(int)
df["is_covid2"] = df.get("Causative Agent-2","").astype(str).str.contains("covid", case=False, na=False).astype(int)
df["is_ltch"]   = (df.get("Outbreak Setting","").astype(str).str.lower() == "ltch").astype(int)

if "Active" not in df.columns:
    df["Active"] = "n" 

# One-hot categorical columns
cat_cols = ["Outbreak Setting", "Type of Outbreak", "Active", "Begin Season", "Causative Agent-1", "Causative Agent-2"]
cat_cols = [col for col in cat_cols if col in df.columns]
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Align features to match training dataset
train_df = pd.read_csv(Train_processed)
train_features = [c for c in train_df.columns if c != "Severity_Label"]
X = df_encoded.reindex(columns=train_features, fill_value=0)

# Load saved scaler and apply it to numeric columns
scaler = joblib.load(Scaler_pkl)
expected_numeric = list(getattr(scaler, "feature_names_in_", ["Outbreak Duration (days)", "Begin Month", "Begin Weekday"]))

# Add any missing numeric columns if necessary
for col in expected_numeric:
    if col not in X.columns:
        if col == "Outbreak Duration (days)" and col in df_encoded.columns:
            X[col] = df_encoded[col]
        else:
            X[col] = 0

# Scale numeric features
X[expected_numeric] = scaler.transform(X[expected_numeric])

# Load label encoder and trained XGBoost model
le = joblib.load(Label_encoder_pkl)
xgb = joblib.load(Xgb_pkl)

probabilities = xgb.predict_proba(X)
pred_ids = np.argmax(probabilities, axis=1)
pred_labels = le.inverse_transform(pred_ids)

# Save predictions to file
output_df = meta.copy()
output_df["Pred_Class_ID"] = pred_ids
output_df["Pred_Severity"] = pred_labels
output_df.to_csv(Output_csv, index=False)

print("XGBoost predictions saved to 'predictions_xgb_may2025_may2026.csv'")
print(output_df.head(10).to_string(index=False))
