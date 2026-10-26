
# Step 18: Final TabPFN Test Evaluation

import pandas as pd
from pathlib import Path
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

# 1. Load training and untouched test data
project_dir = Path(r"C:\Users\srava\OneDrive\Documents")

train_data = pd.read_csv(project_dir / "churn_train.csv")
test_data = pd.read_csv(project_dir / "churn_test.csv")

X_train = train_data.drop(columns=["Churn"])
y_train = train_data["Churn"].astype(str)

X_test = test_data.drop(columns=["Churn"])
y_test = test_data["Churn"].astype(str)

print("Training dataset shape:", X_train.shape)
print("Test dataset shape:", X_test.shape)

# 2. Prepare categorical features consistently
categorical_columns = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

X_train["SeniorCitizen"] = X_train["SeniorCitizen"].astype(str)
X_test["SeniorCitizen"] = X_test["SeniorCitizen"].astype(str)

if "SeniorCitizen" not in categorical_columns:
    categorical_columns.append("SeniorCitizen")

for col in categorical_columns:
    X_train[col] = X_train[col].astype("category")
    X_test[col] = pd.Categorical(
        X_test[col],
        categories=X_train[col].cat.categories
    )

# 3. Train TabPFN on training data only
print("\nInitializing TabPFN on CPU...")

model = TabPFNClassifier(
    device="cpu",
    n_estimators=4,
    random_state=123
)

model.fit(X_train, y_train)

print("Training completed.")

# 4. Predict the test set
print("\nEvaluating on the untouched test set...")

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

yes_index = list(model.classes_).index("Yes")
churn_probabilities = probabilities[:, yes_index]

# 5. Calculate final test metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test, predictions, pos_label="Yes", zero_division=0
)
recall = recall_score(
    y_test, predictions, pos_label="Yes", zero_division=0
)
f1 = f1_score(
    y_test, predictions, pos_label="Yes", zero_division=0
)
roc_auc = roc_auc_score(
    (y_test == "Yes").astype(int),
    churn_probabilities
)

print("\n========== FINAL TABPFN TEST RESULTS ==========")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

# 6. Confusion matrix: rows = actual, columns = predicted
cm = confusion_matrix(
    y_test, predictions, labels=["No", "Yes"]
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual No", "Actual Yes"],
    columns=["Predicted No", "Predicted Yes"]
)

print("\nConfusion matrix:")
print(cm_df)

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=["No", "Yes"],
    zero_division=0
))

# 7. Save final test results
metrics_df = pd.DataFrame([{
    "Model": "TabPFN",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "ROC_AUC": roc_auc
}])

metrics_path = project_dir / "tabpfn_test_results.csv"
cm_path = project_dir / "tabpfn_test_confusion_matrix.csv"

metrics_df.to_csv(metrics_path, index=False)
cm_df.to_csv(cm_path)

print("\nResults saved to:")
print(metrics_path)
print(cm_path)
