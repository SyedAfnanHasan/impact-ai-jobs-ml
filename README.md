# AI Job Impact Prediction using Machine Learning and MLOps

## Project Overview

This project focuses on predicting employment outcomes after AI adoption using machine learning techniques.

The goal is to analyze how factors such as:

- AI adoption level
- Automation probability
- Employee skills
- Training and reskilling activities
- Productivity metrics
- Workplace characteristics

influence future employment outcomes.

The project implements an end-to-end machine learning workflow including:

- Data preprocessing and feature selection
- Machine learning model development
- Model comparison and evaluation
- MLflow experiment tracking
- MLflow Model Registry for model versioning
- Champion/Challenger model workflow
- Model deployment using FastAPI

---

## Problem Statement

Artificial Intelligence adoption is transforming industries and changing workforce requirements.

This project aims to predict an employee's potential employment outcome after AI adoption using machine learning techniques.

The model predicts one of the following employment outcomes:

- Employed
- Reskilled
- Career Change
- Unemployed

The predictions can help organizations:

- Understand potential workforce risks
- Identify employees who may require additional training
- Support workforce planning and reskilling strategies

---

## Dataset

The project uses a **synthetic cross-sectional labor-market dataset** that simulates the impact of AI adoption on global employment trends.

The dataset contains information related to:

- Employee demographics
- Education and professional experience
- Industry and job role
- AI adoption level
- Digital and AI literacy scores
- Productivity metrics
- Training and reskilling activities
- Job satisfaction
- Automation and layoff probabilities
- Employment outcomes after AI adoption

### Target Variable

The target variable for prediction is:

```text
employment_status_after_ai
```
---

## Machine Learning Workflow

The project follows a complete end-to-end machine learning lifecycle:

```text
Data Collection
        |
        ↓
Data Cleaning
        |
        ↓
Exploratory Data Analysis (EDA)
        |
        ↓
Feature Selection
        |
        ↓
Model Training
        |
        ↓
Model Evaluation
        |
        ↓
MLflow Experiment Tracking
        |
        ↓
MLflow Model Registry
        |
        ↓
FastAPI Deployment
```

---

## Feature Selection

Feature importance analysis was performed using the **Random Forest** algorithm to identify the most influential predictors.

The final model was trained using the following **top 16 selected features**:

- `layoff_probability`
- `automation_probability`
- `mental_stress_score`
- `job_satisfaction`
- `training_hours`
- `ai_literacy_score`
- `innovation_score`
- `digital_skill_score`
- `productivity_score`
- `salary_usd`
- `promotion_probability`
- `annual_performance_rating`
- `hours_worked_per_week`
- `ai_adoption_level`
- `age`
- `years_experience`

---

## Models Developed

The following machine learning models were developed and evaluated:

### 1. Random Forest

A baseline Random Forest model was trained using class balancing techniques to handle the imbalanced target classes.

### 2. Random Forest with GridSearchCV

Hyperparameter optimization was performed using **GridSearchCV** to improve model performance.

The following parameters were optimized:

| Parameter | Values Tested |
|---|---|
| `n_estimators` | `[200, 500]` |
| `max_depth` | `[10, 20, None]` |
| `min_samples_leaf` | `[1, 2]` |

---

## Model Evaluation

Since the dataset contains class imbalance, **accuracy alone was not considered sufficient** for evaluating model performance.

The primary evaluation metric used was:

- **Macro F1 Score**

Additional evaluation metrics included:

- Accuracy
- Precision
- Recall
- Classification Report

The final model selection was based on:

- Performance across minority classes
- Balanced prediction capability
- Overall model performance

---

## MLflow Implementation

MLflow was used for experiment tracking, model management, and version control throughout the machine learning lifecycle.

### Implemented MLflow Features

- Experiment tracking
- Parameter logging
- Metric logging
- Model artifact logging
- Model registration
- Model versioning

### MLflow Model Registry Workflow

```text
Random Forest Balanced
        |
        ↓
Random Forest GridSearchCV
        |
        ↓
Deployment-ready Pipeline Model
```

---

## FastAPI Deployment

The trained machine learning model is deployed using a **FastAPI** application for real-time inference.

### Deployment Architecture

```text
Client
  |
  ↓
FastAPI API
  |
  ↓
MLflow Registered Model
  |
  ↓
Prediction
```

### API Endpoint

```http
POST /predict
```
#### Input

The API accepts the 16 selected model features required for generating predictions.

#### Output

Example response:
```
{
    "prediction": "Employed"
}
```

# Project Structure

```text
impact-ai-jobs-ml/
│
├── app/
│   └── main.py
│
├── data/
│
├── notebooks/
│   ├── exploration.ipynb
│   └── mlflow.db
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

---

## Installation and Setup

### Clone the Repository

```bash
git clone https://github.com/SyedAfnanHasan/impact-ai-jobs-ml.git
```

### Clone the Repository

```bash
git clone https://github.com/SyedAfnanHasan/impact-ai-jobs-ml.git
```

### Navigate into the Project Directory

```
cd impact-ai-jobs-ml
```

### Create a Virtual Environment

```bash
python -m venv ml_venv
```

### Activate the Virtual Environment

```bash
source ml_venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```
