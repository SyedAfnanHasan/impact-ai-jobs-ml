from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import pandas as pd


# Connect to MLflow database
mlflow.set_tracking_uri(
    "sqlite:///notebooks/mlflow.db"
)


app = FastAPI(
    title="AI Job Impact Prediction API"
)


# Load champion model from MLflow Registry
model = mlflow.pyfunc.load_model(
    "models:/AI_Job_Impact_Model/4"
)


# Input schema
class JobImpactInput(BaseModel):
    layoff_probability: float
    automation_probability: float
    mental_stress_score: float
    job_satisfaction: float
    training_hours: float
    ai_literacy_score: float
    innovation_score: float
    digital_skill_score: float
    productivity_score: float
    salary_usd: float
    promotion_probability: float
    annual_performance_rating: float
    hours_worked_per_week: float
    ai_adoption_level: float
    age: float
    years_experience: float


# Health check endpoint
@app.get("/")
def home():
    return {
        "message": "AI Job Impact Model API running"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: JobImpactInput):

    # Convert input JSON to dataframe
    input_data = pd.DataFrame(
        [data.model_dump()]
    )

    # Make prediction
    prediction = model.predict(input_data)

    return {
        "prediction": prediction[0]
    }