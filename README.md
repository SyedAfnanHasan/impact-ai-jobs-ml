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
