import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

base_dir   = os.path.dirname(os.path.abspath(__file__))
clean_dir  = os.path.join(base_dir, '..', 'Cleaned Datasets')
out_dir    = os.path.join(base_dir, '..', 'Output Files')
os.makedirs(out_dir, exist_ok=True)

data_path    = os.path.join(clean_dir, 'processed_outbreak_dataset.csv')
Numeric_Cols = ['Begin Month','Begin Weekday']

df = pd.read_csv(data_path)
X, y = df.drop('Severity_Label', axis=1), df['Severity_Label']

# Split into training set (70%) and temporary set (30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

imputer = SimpleImputer(strategy='median').fit(X_train[Numeric_Cols])

imputer_path = os.path.join(out_dir, 'imputer.pkl')
cols_path    = os.path.join(out_dir, 'feature_cols.pkl')

joblib.dump(imputer, imputer_path)
joblib.dump(X_train.columns.tolist(), cols_path)

print('Saved imputer.pkl and feature_cols.pkl')
