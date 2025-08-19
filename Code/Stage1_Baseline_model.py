#Step 5: Model Development & Training (Stage 1 Baseline Model)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score, log_loss, cohen_kappa_score, matthews_corrcoef
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


base_dir = os.path.dirname(os.path.abspath(__file__))
clean_dir = os.path.join(base_dir, '..', 'Cleaned Datasets')

def run_baseline_logistic_model(data_path=os.path.join(clean_dir, 'processed_outbreak_dataset.csv')):
    df = pd.read_csv(data_path)
    X = df.drop('Severity_Label', axis=1)
    y = df['Severity_Label']

    # Split into train, validation and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
   
    # Train logistic regression model
    lr = LogisticRegression(max_iter=10000, random_state=42)
    lr.fit(X_train, y_train)

    # Predict on validation set and compute metrics
    y_pred = lr.predict(X_val)
    y_proba = lr.predict_proba(X_val)
    acc = accuracy_score(y_val, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average=None)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_val, y_pred, average='macro')
    ll = log_loss(y_val, y_proba)
    kappa = cohen_kappa_score(y_val, y_pred)
    mcc = matthews_corrcoef(y_val, y_pred)

    classes = np.unique(y_train)
    y_val_bin = label_binarize(y_val, classes=classes)
    roc_auc = roc_auc_score(y_val_bin, y_proba, average='macro', multi_class='ovr')
    ap = average_precision_score(y_val_bin, y_proba, average='macro')

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

