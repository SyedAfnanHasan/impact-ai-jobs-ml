"""_summary_
"""

import mlflow
import pandas as pd

# Tell MLflow where to store experiment information
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Create/select your experiment
mlflow.set_experiment("impact_ai_jobs_prediction")


# Later we will put model training here
with mlflow.start_run():

    print("MLflow experiment started")
