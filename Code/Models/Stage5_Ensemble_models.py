#Step 5: Model Development & Training (Stage 5 Ensemble Models)

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score, log_loss, cohen_kappa_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize
import os


base_dir = os.path.dirname(os.path.abspath(__file__))
clean_dir = os.path.join(base_dir, '..', 'Cleaned Datasets')
out_dir   = os.path.join(base_dir, '..', 'Output Files')

def _must_exist(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected model file not found: {os.path.basename(path)} at {path}")

def run_ensemble_model(data_path=os.path.join(clean_dir, 'processed_outbreak_dataset.csv')):
    xgb_path  = os.path.join(out_dir, 'xgboost.pkl')
    cnn_path  = os.path.join(out_dir, 'cnn_model.keras')

    _must_exist(xgb_path)
    _must_exist(cnn_path)

    xgb_model = joblib.load(xgb_path)
    cnn_model = tf.keras.models.load_model(cnn_path)

    df = pd.read_csv(data_path)
    X = df.drop('Severity_Label', axis=1)
    y = df['Severity_Label']

    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    # Fill missing values in numeric columns using median values
    numeric_cols = ['Begin Month', 'Begin Weekday']
    imputer = SimpleImputer(strategy='median')
    X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = imputer.transform(X_val[numeric_cols])
    X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])

    # Function to compute performance metrics for a given dataset
    def compute_metrics(X_input, y_input):
        proba_xgb = xgb_model.predict_proba(X_input)
        X_input_arr = X_input.values.astype('float32').reshape((X_input.shape[0], X_input.shape[1], 1))
        proba_cnn = cnn_model.predict(X_input_arr, verbose=0)
        ensemble_proba = (proba_xgb + proba_cnn) / 2.0
        y_pred = np.argmax(ensemble_proba, axis=1)

        classes = np.unique(y_train)
        y_bin = label_binarize(y_input, classes=classes)

        # Calculate metrics
        acc = accuracy_score(y_input, y_pred)
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_input, y_pred, average='macro', zero_division=0)
        roc_auc = roc_auc_score(y_bin, ensemble_proba, average='macro', multi_class='ovr')
        ap = average_precision_score(y_bin, ensemble_proba, average='macro')
        ll = log_loss(y_input, ensemble_proba)
        kappa = cohen_kappa_score(y_input, y_pred)
        mcc = matthews_corrcoef(y_input, y_pred)

        return {
            "Accuracy": acc,
            "Macro Precision": prec_m,
            "Macro Recall": rec_m,
            "Macro F1-score": f1_m,
            "AUC-ROC (macro)": roc_auc,
            "AUC-PR (macro)": ap,
            "Log Loss": ll,
            "Cohen’s Kappa": kappa,
            "Matthews CorrCoef": mcc
        }

    return {"val": compute_metrics(X_val, y_val), "test": compute_metrics(X_test, y_test)}