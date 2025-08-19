#Step 5: Model Development & Training (Stage 2 ML Models)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score, log_loss, cohen_kappa_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
import joblib
import os

# Folder paths
base_dir = os.path.dirname(os.path.abspath(__file__))
clean_dir = os.path.join(base_dir, '..', 'Cleaned Datasets')
out_dir   = os.path.join(base_dir, '..', 'Output Files')

os.makedirs(clean_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

def run_advanced_ml_models(data_path=os.path.join(clean_dir, 'processed_outbreak_dataset.csv')):
    df = pd.read_csv(data_path)
    X = df.drop('Severity_Label', axis=1)
    y = df['Severity_Label']

    # Split into train, validation and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    # Fill missing values in numeric columns with median
    numeric_cols = ['Begin Month', 'Begin Weekday']
    imputer = SimpleImputer(strategy='median')
    X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = imputer.transform(X_val[numeric_cols])
    X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])

    # Train the Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)

    # Predict on validation set and calculate metrics
    y_pred_rf = rf.predict(X_val)
    y_proba_rf = rf.predict_proba(X_val)
    acc_rf = accuracy_score(y_val, y_pred_rf)
    prec_m_rf, rec_m_rf, f1_m_rf, _ = precision_recall_fscore_support(y_val, y_pred_rf, average='macro')
    ll_rf = log_loss(y_val, y_proba_rf)
    kappa_rf = cohen_kappa_score(y_val, y_pred_rf)
    mcc_rf = matthews_corrcoef(y_val, y_pred_rf)

    classes_rf = rf.classes_
    y_val_bin_rf = label_binarize(y_val, classes=classes_rf)
    roc_auc_rf = roc_auc_score(y_val_bin_rf, y_proba_rf, average='macro', multi_class='ovr')
    ap_rf = average_precision_score(y_val_bin_rf, y_proba_rf, average='macro')

    # Save the Random Forest model
    rf_path = os.path.join(out_dir, 'random_forest.pkl')
    joblib.dump(rf, rf_path)

    # Define parameter search space for XGBoost
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 1, 5]
    }

    # Run randomized search for best XGBoost model
    xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
    search = RandomizedSearchCV(estimator=xgb, param_distributions=param_dist, n_iter=20, scoring='f1_macro', cv=5, n_jobs=-1, verbose=0, random_state=42)
    search.fit(X_train, y_train)

    # Predict on validation set with best model and calculate metrics
    best_xgb = search.best_estimator_
    y_pred_xgb = best_xgb.predict(X_val)
    y_proba_xgb = best_xgb.predict_proba(X_val)
    acc = accuracy_score(y_val, y_pred_xgb)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_val, y_pred_xgb, average='macro')
    ll = log_loss(y_val, y_proba_xgb)
    kappa = cohen_kappa_score(y_val, y_pred_xgb)
    mcc = matthews_corrcoef(y_val, y_pred_xgb)

    classes_xgb = best_xgb.classes_
    y_val_bin_xgb = label_binarize(y_val, classes=classes_xgb)
    roc_auc = roc_auc_score(y_val_bin_xgb, y_proba_xgb, average='macro', multi_class='ovr')
    ap = average_precision_score(y_val_bin_xgb, y_proba_xgb, average='macro')

    # Save the XGBoost model
    xgb_path = os.path.join(out_dir, 'xgboost.pkl')
    joblib.dump(best_xgb, xgb_path)

    return {
        "Random Forest": {
            "Accuracy": acc_rf,
            "Macro Precision": prec_m_rf,
            "Macro Recall": rec_m_rf,
            "Macro F1-score": f1_m_rf,
            "AUC-ROC (macro)": roc_auc_rf,
            "AUC-PR (macro)": ap_rf,
            "Log Loss": ll_rf,
            "Cohen’s Kappa": kappa_rf,
            "Matthews CorrCoef": mcc_rf
        },
        "XGBoost": {
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
    }


