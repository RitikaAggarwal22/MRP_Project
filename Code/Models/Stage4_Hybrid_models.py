#Step 5: Model Development & Training (Stage 4 Hybrid Models)

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score, log_loss, cohen_kappa_score, matthews_corrcoef
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
import os


base_dir = os.path.dirname(os.path.abspath(__file__))
clean_dir = os.path.join(base_dir, '..', 'Cleaned Datasets')
out_dir   = os.path.join(base_dir, '..', 'Output Files')

os.makedirs(clean_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

def run_hybrid_model(data_path=os.path.join(clean_dir, 'processed_outbreak_dataset.csv')):
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

    # Prepare data for CNN+LSTM
    X_train_arr = X_train.values.astype('float32')
    X_val_arr = X_val.values.astype('float32')
    n_train, n_feat = X_train_arr.shape
    X_train_seq = X_train_arr.reshape((n_train, n_feat, 1))
    X_val_seq = X_val_arr.reshape((X_val_arr.shape[0], n_feat, 1))

    # Determine number of classes dynamically (e.g., 2 or 3)
    classes_hy = np.sort(np.unique(y_train))
    num_classes = len(classes_hy)

    # Build and train hybrid CNN+LSTM model
    model_hybrid = Sequential([
        Input(shape=(n_feat, 1)),
        Conv1D(32, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.3),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model_hybrid.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_hybrid.fit(X_train_seq, y_train, epochs=100, batch_size=32, validation_data=(X_val_seq, y_val), verbose=0)

    # Predict and calculate metrics
    y_proba_hybrid = model_hybrid.predict(X_val_seq, verbose=0)
    y_pred_hybrid  = np.argmax(y_proba_hybrid, axis=1)
    acc_hy = accuracy_score(y_val, y_pred_hybrid)
    prec_hy, rec_hy, f1_hy, _ = precision_recall_fscore_support(y_val, y_pred_hybrid, average=None, zero_division=0)
    prec_m_hy, rec_m_hy, f1_m_hy, _ = precision_recall_fscore_support(y_val, y_pred_hybrid, average='macro', zero_division=0)
    ll_hy = log_loss(y_val, y_proba_hybrid)
    kappa_hy = cohen_kappa_score(y_val, y_pred_hybrid)
    mcc_hy = matthews_corrcoef(y_val, y_pred_hybrid)

    #classes = np.unique(y_train)
    y_val_bin = label_binarize(y_val, classes=classes_hy)
    roc_auc_hy = roc_auc_score(y_val_bin, y_proba_hybrid, average='macro', multi_class='ovr')
    ap_hy = average_precision_score(y_val_bin, y_proba_hybrid, average='macro')

    # Save the model
    hybrid_path = os.path.join(out_dir, 'hybrid_cnn_lstm_model.keras')
    model_hybrid.save(hybrid_path)

    return {
        "Accuracy": acc_hy,
        "Macro Precision": prec_m_hy,
        "Macro Recall": rec_m_hy,
        "Macro F1-score": f1_m_hy,
        "AUC-ROC (macro)": roc_auc_hy,
        "AUC-PR (macro)": ap_hy,
        "Log Loss": ll_hy,
        "Cohen's Kappa": kappa_hy,
        "Matthews CorrCoef": mcc_hy
    }

