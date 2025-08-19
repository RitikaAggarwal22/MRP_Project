#Step 5: Model Development & Training (Stage 3 DL Models)

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score, log_loss, cohen_kappa_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

# Folder paths
base_dir = os.path.dirname(os.path.abspath(__file__))
clean_dir = os.path.join(base_dir, '..', 'Cleaned Datasets')
out_dir   = os.path.join(base_dir, '..', 'Output Files')

os.makedirs(clean_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

def run_deep_learning_models(data_path=os.path.join(clean_dir, 'processed_outbreak_dataset.csv')):
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

    # Train an MLP classifier
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=200, random_state=42)
    mlp.fit(X_train, y_train)

    # Predict and calculate metrics for MLP
    y_pred_mlp = mlp.predict(X_val)
    y_proba_mlp = mlp.predict_proba(X_val)
    acc_mlp = accuracy_score(y_val, y_pred_mlp)
    prec_mlp, rec_mlp, f1_mlp, _ = precision_recall_fscore_support(y_val, y_pred_mlp, average=None, zero_division=0)
    prec_mm, rec_mm, f1_mm, _ = precision_recall_fscore_support(y_val, y_pred_mlp, average='macro', zero_division=0)
    ll_mlp = log_loss(y_val, y_proba_mlp)
    kappa_mlp = cohen_kappa_score(y_val, y_pred_mlp)
    mcc_mlp = matthews_corrcoef(y_val, y_pred_mlp)

    classes_mlp = mlp.classes_
    y_val_bin = label_binarize(y_val, classes=classes_mlp)
    roc_auc_mlp = roc_auc_score(y_val_bin, y_proba_mlp, average='macro', multi_class='ovr')
    ap_mlp = average_precision_score(y_val_bin, y_proba_mlp, average='macro')

    # Save the MLP model
    mlp_path = os.path.join(out_dir, 'mlp_model.pkl')
    joblib.dump(mlp, mlp_path)

    # Prepare data for CNN
    X_train_arr = X_train.values.astype('float32')
    X_val_arr = X_val.values.astype('float32')
    n_train, n_feat = X_train_arr.shape
    X_train_cnn = X_train_arr.reshape((n_train, n_feat, 1))
    X_val_cnn = X_val_arr.reshape((X_val_arr.shape[0], n_feat, 1))

    # Determine number of classes dynamically
    num_classes = len(np.unique(y_train))

    # Define and train a CNN model
    model = Sequential([
        tf.keras.Input(shape=(n_feat, 1)),
        Conv1D(32, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.3),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.3),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_cnn, y_train, epochs=100, batch_size=32, validation_data=(X_val_cnn, y_val), verbose=0)

    # Predict and calculate metrics for CNN
    y_proba_cnn = model.predict(X_val_cnn, verbose=0)
    y_pred_cnn = np.argmax(y_proba_cnn, axis=1)
    acc_cnn = accuracy_score(y_val, y_pred_cnn)
    prec_cnn, rec_cnn, f1_cnn, _ = precision_recall_fscore_support(y_val, y_pred_cnn, average=None, zero_division=0)
    prec_m_cnn, rec_m_cnn, f1_m_cnn, _ = precision_recall_fscore_support(y_val, y_pred_cnn, average='macro', zero_division=0)
    ll_cnn = log_loss(y_val, y_proba_cnn)
    kappa_cnn = cohen_kappa_score(y_val, y_pred_cnn)
    mcc_cnn = matthews_corrcoef(y_val, y_pred_cnn)

    classes_cnn = np.unique(y_train)
    y_val_bin_cnn = label_binarize(y_val, classes=classes_cnn)
    roc_auc_cnn = roc_auc_score(y_val_bin_cnn, y_proba_cnn, average='macro', multi_class='ovr')
    ap_cnn = average_precision_score(y_val_bin_cnn, y_proba_cnn, average='macro')

    # Save the CNN model
    cnn_path = os.path.join(out_dir, 'cnn_model.keras')
    model.save(cnn_path)

    return {
        "MLP": {
            "Accuracy": acc_mlp,
            "Macro Precision": prec_mm,
            "Macro Recall": rec_mm,
            "Macro F1-score": f1_mm,
            "AUC-ROC (macro)": roc_auc_mlp,
            "AUC-PR (macro)": ap_mlp,
            "Log Loss": ll_mlp,
            "Cohen’s Kappa": kappa_mlp,
            "Matthews CorrCoef": mcc_mlp
        },
        "CNN": {
            "Accuracy": acc_cnn,
            "Macro Precision": prec_m_cnn,
            "Macro Recall": rec_m_cnn,
            "Macro F1-score": f1_m_cnn,
            "AUC-ROC (macro)": roc_auc_cnn,
            "AUC-PR (macro)": ap_cnn,
            "Log Loss": ll_cnn,
            "Cohen’s Kappa": kappa_cnn,
            "Matthews CorrCoef": mcc_cnn
        }
    }
