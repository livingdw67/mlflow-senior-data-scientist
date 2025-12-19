import mlflow
import shutil
import os

# 1. Clean up old runs so we start fresh
if os.path.exists("mlruns"):
    shutil.rmtree("mlruns")

# 2. Set up the local registry folder
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("churn_prediction_v1")

print("🌱 Seeding registry with fake experiments...")

# 3. Log a BAD run (Linear Regression)
with mlflow.start_run(run_name="Run_1_Baseline"):
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("f1_score", 0.60)
    print("Logged Run 1 (Bad)")

# 4. Log a BEST run (XGBoost)
with mlflow.start_run(run_name="Run_3_XGB"):
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_metric("f1_score", 0.88)
    print("Logged Run 3 (Best)")

print("✅ Done! You now have a registry history.")