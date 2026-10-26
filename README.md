# Predicting Customer Churn Using TabPFN with Model Explainability

## Overview
This project predicts customer churn using TabPFN and compares its performance with a Logistic Regression baseline. Permutation Feature Importance is used to examine the model's global predictive reliance on input features.

## Dataset
IBM Telco Customer Churn dataset:  
https://github.com/IBM/telco-customer-churn-on-icp4d

The dataset was cleaned and split into training, validation, and test sets. See the code for the preprocessing steps.

## Project Structure
- `run_tabpfn.py` — TabPFN training and validation evaluation.
- `run_tabpfn_test.py` — TabPFN test evaluation.
- `run_tabpfn_explainability.py` — Permutation Feature Importance analysis.
- `logistic_regression.R` — Logistic Regression baseline.
- `data/` 
- `results/` — Evaluation outputs and charts.
- `requirements.txt` — Python dependencies.

## How to Run
1. Install the required Python packages.
2. Configure any required TabPFN authentication privately.
3. Run the Python scripts using the appropriate Python environment.
4. Run the R script in an R environment with the required packages installed.

Check each script for its expected input paths and execution order.

## Test Results

| Metric | Logistic Regression | TabPFN |
|---|---:|---:|
| Accuracy | 80.06% | 79.96% |
| Precision | 65.22% | 66.35% |
| Recall | 53.38% | 49.82% |
| F1-score | 58.71% | 56.91% |
| ROC-AUC | 0.8178 | 0.8253 |

## Explainability
Permutation Feature Importance ranked Contract as the most influential feature for validation ROC-AUC, followed by tenure, InternetService, and TotalCharges. These scores reflect model reliance, not causation.

## Limitations
The experiments use one dataset and a single train/validation/test split. Results may differ on other datasets or splits.
