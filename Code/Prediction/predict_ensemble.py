#Predict severity for May 2025–May 2026 using XGBoost + CNN ensemble

import os
import pandas as pd
import numpy as np
import joblib
import warnings
import tensorflow as tf

base_dir   = os.path.dirname(os.path.abspath(__file__))
clean_dir  = os.path.join(base_dir, '..', 'Cleaned Datasets')
out_dir    = os.path.join(base_dir, '..', 'Output Files')
os.makedirs(out_dir, exist_ok=True)

# File paths
Master_csv = os.path.join(clean_dir, 'master_outbreak_dataset.csv')
Train_processed = os.path.join(clean_dir, 'processed_outbreak_dataset.csv')
Scaler_pkl = os.path.join(out_dir, 'scaler.pkl')
Label_encoder_pkl = os.path.join(out_dir, 'severity_label_encoder.pkl')
Xgb_pkl = os.path.join(out_dir, 'xgboost.pkl')
Cnn_keras = os.path.join(out_dir, 'cnn_model.keras')
Output_csv = os.path.join(out_dir, 'predictions_ensemble_may2025_may2026.csv')


df_raw = pd.read_csv(Master_csv, dtype=str)

# Convert outbreak start date to datetime
def toDatetime(value):
    try:
        return pd.to_datetime(value, errors="coerce", infer_datetime_format=True)
    except TypeError:
        return pd.NaT

if "Date Outbreak Began" not in df_raw.columns:
    raise SystemExit("Column 'Date Outbreak Began' not found in master dataset")

df_raw["Date Outbreak Began"] = df_raw["Date Outbreak Began"].apply(toDatetime)

# Drop rows with invalid dates
invalidDateCount = df_raw["Date Outbreak Began"].isna().sum()
if invalidDateCount > 0:
    print(f"Warning: {invalidDateCount} rows have invalid dates and will be dropped.")

df = df_raw.dropna(subset=["Date Outbreak Began"]).copy()

# Keep only rows within May 2025–May 2026
df = df[(df["Date Outbreak Began"] >= "2025-05-01") & (df["Date Outbreak Began"] <= "2026-05-31")].copy()
if df.empty:
    print("No rows in May 2025–May 2026. Nothing to predict.")
    raise SystemExit()

# Keep metadata columns for output
id_cols = ["_id","Institution Name","Outbreak Setting","Type of Outbreak","Causative Agent-1","Causative Agent-2","Date Outbreak Began","Outbreak Duration (days)"]
metadataColumns = [col for col in id_cols if col in df.columns]
meta = df[metadataColumns].copy() if metadataColumns else pd.DataFrame(index=df.index)

# Feature engineering
df["Begin Month"]   = df["Date Outbreak Began"].dt.month
df["Begin Weekday"] = df["Date Outbreak Began"].dt.weekday
def month_to_season(m):
    if pd.isna(m): return np.nan
    m = int(m)
    if m in (12,1,2): return "Winter"
    if m in (3,4,5):  return "Spring"
    if m in (6,7,8):  return "Summer"
    return "Fall"
df["Begin Season"] = df["Begin Month"].apply(month_to_season)

# Convert Year column to numeric if it exists
if "Year" in df.columns:
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(0).astype(int)

df["is_covid1"] = df.get("Causative Agent-1", "").astype(str).str.contains("covid", case=False, na=False).astype(int)
df["is_covid2"] = df.get("Causative Agent-2", "").astype(str).str.contains("covid", case=False, na=False).astype(int)
df["is_ltch"]   = df.get("Outbreak Setting", "").astype(str).str.strip().str.lower().eq("ltch").astype(int)

if "Active" not in df.columns:
    df["Active"] = "n"

cat_cols = ["Outbreak Setting", "Type of Outbreak", "Active", "Begin Season", "Causative Agent-1", "Causative Agent-2"]
cat_cols = [col for col in cat_cols if col in df.columns]
df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Align features to match training dataset
train_df = pd.read_csv(Train_processed)
train_features = [c for c in train_df.columns if c != "Severity_Label"]
X = df_enc.reindex(columns=train_features, fill_value=0)

# Scale numeric columns
numeric_cols = ["Outbreak Duration (days)", "Begin Month", "Begin Weekday"]
numeric_cols = [col for col in numeric_cols if col in X.columns]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    scaler = joblib.load(Scaler_pkl)
if numeric_cols:
    X[numeric_cols] = scaler.transform(X[numeric_cols])

# Ensure all features are numeric
obj_cols = [c for c in X.columns if X[c].dtype == "O"]
for c in obj_cols:
    # try numeric first
    tmp = pd.to_numeric(X[c], errors="coerce")
    if tmp.notna().any():
        X[c] = tmp.fillna(0)
    else:
        # fallback: category codes
        X[c] = X[c].astype("category").cat.codes.astype("int32")
X = X.astype(np.float32)

# Load label encoder and both models
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    le  = joblib.load(Label_encoder_pkl)
    xgb = joblib.load(Xgb_pkl)
cnn = tf.keras.models.load_model(Cnn_keras)

# Predict probabilities from both models
proba_xgb = xgb.predict_proba(X)
X_cnn = X.values.reshape((X.shape[0], X.shape[1], 1))
proba_cnn = cnn.predict(X_cnn, verbose=0)
proba_ens = (proba_xgb + proba_cnn) / 2.0
pred_ids = np.argmax(proba_ens, axis=1)
pred_labels = le.inverse_transform(pred_ids)

# Save output
out = meta.copy()
out["Pred_Class_ID"] = pred_ids
out["Pred_Severity"] = pred_labels
out.to_csv(Output_csv, index=False)

print("Ensemble predictions saved to 'predictions_ensemble_may2025_may2026.csv'")
print(out.head(10).to_string(index=False))
