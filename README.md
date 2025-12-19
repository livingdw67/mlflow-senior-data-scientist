# MLflow Experiment Tracking Architecture

**A template for reproducible, scalable Machine Learning pipelines.**

## The Business Case
In enterprise environments, the challenge isn't just building a model—it's managing the chaos of hundreds of experiments. This project demonstrates a **Senior-level MLOps framework** designed to solve:
* **Reproducibility:** Eliminating "it works on my machine" syndrome.
* **Traceability:** Logging every hyperparameter, metric, and artifact version.
* **Standardization:** Creating a reusable pattern for model training and deployment.

## Technical Architecture
This repository implements a production-ready workflow using **MLflow** and **Scikit-Learn**:

* **train_best_model.py:** The main orchestration script. It loads data, trains a Random Forest model, and logs all metrics (RMSE, MAE) and artifacts to the MLflow tracking server.
* **seed_registry.py:** A utility module ensuring statistical reproducibility across different runs and environments.
* **PJME_hourly.csv:** The time-series dataset used for demonstrating the forecasting pipeline.

## Quick Start

1.  **Clone the Repo**
    \\\ash
    git clone https://github.com/livingdw67/mlflow-senior-data-scientist.git
    cd mlflow-senior-data-scientist
    \\\

2.  **Install Dependencies**
    \\\ash
    pip install -r requirements.txt
    \\\

3.  **Run the Training Pipeline**
    Execute the main script to train the model and log a new run:
    \\\ash
    python train_best_model.py
    \\\

4.  **View the Dashboard**
    Launch the MLflow UI to inspect model performance visually:
    \\\ash
    mlflow ui
    # Open http://localhost:5000 in your browser
    \\\

## Tech Stack
* **Python 3.10+**
* **MLflow** (Experiment Tracking & Model Registry)
* **Scikit-Learn** (Modeling)
* **Pandas** (Data Engineering)

---
*Built by Daniel Livingston | Senior Data Scientist*
