# Import all model-running functions from different stages
from Stage1_Baseline_model import run_baseline_logistic_model
from Stage2_ML_models import run_advanced_ml_models
from Stage3_DL_models import run_deep_learning_models
from Stage4_Hybrid_models import run_hybrid_model
from Stage5_Ensemble_models import run_ensemble_model

import pandas as pd
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

base_dir = os.path.dirname(os.path.abspath(__file__))
out_dir  = os.path.join(base_dir, '..', 'Output Files')
os.makedirs(out_dir, exist_ok=True)

# Run all Models 
baseline_result = run_baseline_logistic_model()
ml_results = run_advanced_ml_models()
dl_results = run_deep_learning_models()
hybrid_result = run_hybrid_model()
ensemble_result = run_ensemble_model()

# Combine results from all models into a single dictionary
results = {
    "Logistic Regression (Baseline)": baseline_result,
    "Random Forest": ml_results["Random Forest"],
    "XGBoost": ml_results["XGBoost"],
    "MLPClassifier": dl_results["MLP"],
    "CNN": dl_results["CNN"],
    "CNN + LSTM (Hybrid)": hybrid_result,
    "XGB + CNN (Ensemble Val)": ensemble_result["val"],
    "XGB + CNN (Ensemble Test)": ensemble_result["test"]
}

# Print each model's metrics
for model_name, metrics in results.items():
    print(f"\n{'=' * 60}")
    print(f" Results for: {model_name}")
    print('-' * 60)
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            print(f"{metric_name:25}: {metric_value:.4f}")
        else:
            print(f"{metric_name:25}: {metric_value}")

# Create a comparison table
comparison_data = {model: metrics for model, metrics in results.items()}
df = pd.DataFrame(comparison_data)

# Save the metrics for later use
metrics_pkl_path = os.path.join(out_dir, "model_metrics.pkl")
with open(metrics_pkl_path, "wb") as f:
    pickle.dump(comparison_data, f)

# Print the comparison table
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)

print(f"\n{'=' * 60}")
print("\n FINAL COMPARISON TABLE ")
print('-' * 60)
print(df.to_string(float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
