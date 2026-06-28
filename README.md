# Bank Customer Churn Risk Scoring

## Project Overview

This project develops a machine learning based churn risk scoring system for retail banking customers.
It predicts customer churn probability, assigns a risk category, and recommends retention actions.

## Objectives

- Predict customer churn
- Generate churn probability scores
- Identify important churn drivers
- Build a Streamlit dashboard
- Support targeted customer retention

## Dataset

The dataset contains 10,000 European bank customer records.

Main features include CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, and Exited.

Target variable:

- Exited = 1 means customer churned
- Exited = 0 means customer retained

## Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- Hist Gradient Boosting
- XGBoost

## Final Model

Final deployment model: Calibrated Tuned Gradient Boosting

Final calibrated test performance:

| Metric | Value |
|---|---:|
| Accuracy | 0.8400 |
| Precision | 0.5960 |
| Recall | 0.6634 |
| F1 Score | 0.6279 |
| ROC-AUC | 0.8667 |
| PR-AUC | 0.7095 |

## Dashboard Features

- Customer churn risk calculator
- Churn probability gauge
- Risk category output
- Retention recommendation
- What-if simulator
- Customer analytics
- Model performance comparison
- Feature importance dashboard

## How to Run

Install requirements:

pip install -r requirements.txt

Run the app:

streamlit run app.py

## Conclusion

This project reframes customer churn prediction as a proactive risk intelligence system for banking customer retention.