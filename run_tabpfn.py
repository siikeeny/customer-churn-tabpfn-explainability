
# Customer Churn Prediction using TabPFN
# Step 13: Train and evaluate on the validation set

import pandas as pd
import numpy as np

from tabpfn import TabPFNClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from pathlib import Path

# --------------------------------------------------
# 1. Set the project folder
# --------------------------------------------------

project_dir = Path(r"C:\Users\srava\OneDrive\Documents")

train_path = project_dir / "churn_train.csv"
validation_path = project_dir / "churn_validation.csv"

# --------------------------------------------------
# 2. Load the saved datasets from R
# --------------------------------------------------

train_data = pd.read_csv(train_path)
validation_data = pd.read_csv(validation_path)

print("Training dataset shape:", train_data.shape)
print("Validation dataset shape:", validation_data.shape)

# Separate features (X) and target (y)
X_train = train_data.drop(columns=["Churn"])
y_train = train_data["Churn"].astype(str)

X_validation = validation_data.drop(columns=["Churn"])
y_validation = validation_data["Churn"].astype(str)

# --------------------------------------------------
# 3. Mark categorical columns
# --------------------------------------------------

categorical_columns = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

# SeniorCitizen is a 0/1 category, not a continuous measurement
X_train["SeniorCitizen"] = X_train["SeniorCitizen"].astype(str)
X_validation["SeniorCitizen"] = X_validation["SeniorCitizen"].astype(str)

if "SeniorCitizen" not in categorical_columns:
    categorical_columns.append("SeniorCitizen")

for col in categorical_columns:
    X_train[col] = X_train[col].astype("category")
    X_validation[col] = pd.Categorical(
        X_validation[col],
        categories=X_train[col].cat.categories
    )

print("\nCategorical columns:")
print(categorical_columns)

# --------------------------------------------------
# 4. Create and train the TabPFN classifier
# --------------------------------------------------

print("\nInitializing TabPFN on CPU...")

model = TabPFNClassifier(
    device="cpu",
    n_estimators=4,
    random_state=123
)

print("Training TabPFN. The first run may download model weights...")
model.fit(X_train, y_train)

print("Training completed.")

# --------------------------------------------------
# 5. Predict on the validation set
# --------------------------------------------------

print("\nPredicting validation results...")

validation_predictions = model.predict(X_validation)
validation_probabilities = model.predict_proba(X_validation)

# Find the probability column corresponding to churn = Yes
yes_index = list(model.classes_).index("Yes")
churn_probabilities = validation_probabilities[:, yes_index]

# --------------------------------------------------
# 6. Calculate validation metrics
# --------------------------------------------------

accuracy = accuracy_score(
    y_validation, validation_predictions
)

precision = precision_score(
    y_validation, validation_predictions,
    pos_label="Yes", zero_division=0
)

recall = recall_score(
    y_validation, validation_predictions,
    pos_label="Yes", zero_division=0
)

f1 = f1_score(
    y_validation, validation_predictions,
    pos_label="Yes", zero_division=0
)

roc_auc = roc_auc_score(
    (y_validation == "Yes").astype(int),
    churn_probabilities
)

cm = confusion_matrix(
    y_validation,
    validation_predictions,
    labels=["No", "Yes"]
)

print("\n========== TABPFN VALIDATION RESULTS ==========")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion matrix (rows=actual, columns=predicted):")
print(pd.DataFrame(
    cm,
    index=["Actual No", "Actual Yes"],
    columns=["Predicted No", "Predicted Yes"]
))

print("\nClassification report:")
print(classification_report(
    y_validation,
    validation_predictions,
    labels=["No", "Yes"],
    zero_division=0
))

# --------------------------------------------------
# 7. Save results for the project report
# --------------------------------------------------

results = pd.DataFrame([{
    "Model": "TabPFN",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "ROC_AUC": roc_auc
}])

results.to_csv(
    project_dir / "tabpfn_validation_results.csv",
    index=False
)

pd.DataFrame(
    cm,
    index=["Actual No", "Actual Yes"],
    columns=["Predicted No", "Predicted Yes"]
).to_csv(project_dir / "tabpfn_validation_confusion_matrix.csv")

print("\nResults saved to:")
print(project_dir / "tabpfn_validation_results.csv")
print(project_dir / "tabpfn_validation_confusion_matrix.csv")
