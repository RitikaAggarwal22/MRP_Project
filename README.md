# OUTBREAK SEVERITY PREDICTION USING ML, DL, HYBRID & ENSEMBLE MODELS  

This repository contains end-to-end experiments with **Machine Learning (Random Forest, XGBoost)**, **Deep Learning (MLP, CNN, LSTM)**, and **Hybrid/Ensemble models** for predicting the **severity of infectious disease outbreaks**.  

It includes data preprocessing, feature engineering, model training & evaluation, and an interactive **Streamlit dashboard** for visualizing outbreak patterns and predictions.  

---

## Repository Structure  

```
├── Code/
│   ├── Main/
│   │   ├── Dashboard.py               # Streamlit dashboard
│   │   └── main.py                    # Runs evaluation + dashboard
│   │
│   ├── Models/
│   │   ├── Stage1_Baseline_model.py   # Logistic Regression baseline
│   │   ├── Stage2_ML_models.py        # Random Forest, XGBoost
│   │   ├── Stage3_DL_models.py        # MLP, CNN
│   │   ├── Stage4_Hybrid_models.py    # CNN + LSTM hybrid
│   │   └── Stage5_Ensemble_models.py  # XGBoost + CNN ensemble
│   │
│   ├── Prediction/
│   │   ├── freeze_preprocessing.py    # Lock preprocessing pipeline
│   │   ├── predict_xgboost.py         # XGBoost predictions
│   │   └── predict_ensemble.py        # Ensemble predictions
│   │
│   ├── Step1_Data_collection_&_integration.py
│   ├── Step2_Data_preprocessing_&_cleaning.py
│   ├── Step3_Exploratory_data_analysis.py
│   └── Step4_Feature_Engineering.py
│
├── Dataset/
│   ├── Cleaned/
│   │   ├── master_outbreak_dataset.csv     # Combined raw data
│   │   ├── preprocessed_dataset.csv        # Final dataset for modeling
│   │   └── processed_outbreak_dataset.csv  # Processed structured dataset
│   └── Raw/                                # Original raw files (if any)
│
├── EDA Outputs/                            # Visualizations from EDA
│
├── Output Files/                           # Trained models & encoders
│
├── MRP Report_Ritika_Aggarwal.pdf          # Final research project report
├── Requirements.txt                        # Python dependencies
└── README.md                               # This file
```

---

## Quickstart  

### 1. Clone repository  
```bash
git clone https://github.com/RitikaAggarwal22/MRP_Project
```

### 2. Install dependencies  
```bash
pip install -r Requirements.txt
```

### 3. Preprocess data  
```bash
python Code/Step2_Data_preprocessing_&_cleaning.py
```

### 4. Train models  
Each stage has its own script:  
```bash
python Code/Models/Stage1_Baseline_model.py
python Code/Models/Stage2_ML_models.py
python Code/Models/Stage3_DL_models.py
python Code/Models/Stage4_Hybrid_models.py
python Code/Models/Stage5_Ensemble_models.py
```

### 5. Evaluate models  
```bash
python Code/Main/main.py
```

### 6. Launch dashboard  
```bash
streamlit run Code/Main/Dashboard.py
```

---

## Dataset  

- **Toronto Public Health Open Data (2016–2025)**  
- Settings: hospitals, long-term care homes, retirement homes  
- Features: outbreak type, setting, causative agent, start & end dates, duration  
- Severity labels: **Mild, Moderate, Severe** (based on outbreak duration thresholds)  

---

