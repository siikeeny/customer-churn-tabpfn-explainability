
# Step 17: TabPFN Permutation Feature Importance

import pandas as pd
import numpy as np
from pathlib import Path
from tabpfn import TabPFNClassifier
from sklearn.metrics import roc_auc_score

# 1. Locate the project files
project_dir = Path(r"C:\Users\srava\OneDrive\Documents")

train_data = pd.read_csv(project_dir / "churn_train.csv")
validation_data = pd.read_csv(project_dir / "churn_validation.csv")

X_train = train_data.drop(columns=["Churn"])
y_train = train_data["Churn"].astype(str)

X_validation = validation_data.drop(columns=["Churn"])
y_validation = validation_data["Churn"].astype(str)

print("Training dataset:", X_train.shape)
print("Validation dataset:", X_validation.shape)

# 2. Prepare categorical columns
categorical_columns = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

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

# 3. Train TabPFN
print("\nInitializing TabPFN on CPU...")

model = TabPFNClassifier(
    device="cpu",
    n_estimators=4,
    random_state=123
)

model.fit(X_train, y_train)

print("Training completed.")

# 4. Calculate baseline validation ROC-AUC
yes_index = list(model.classes_).index("Yes")

baseline_probabilities = model.predict_proba(X_validation)[:, yes_index]
baseline_auc = roc_auc_score(
    (y_validation == "Yes").astype(int),
    baseline_probabilities
)

print(f"\nBaseline validation ROC-AUC: {baseline_auc:.4f}")

# 5. Permute one feature at a time
# Three repetitions per feature to limit CPU runtime.
rng = np.random.default_rng(123)
importance_results = []

for col in X_validation.columns:
    print(f"Testing feature: {col}")

    auc_scores = []

    for repeat in range(3):
        X_permuted = X_validation.copy()

        # Shuffle this feature while keeping its data type consistent.
        shuffled_values = X_validation[col].to_numpy(copy=True)
        rng.shuffle(shuffled_values)

        if col in categorical_columns:
            X_permuted[col] = pd.Categorical(
                shuffled_values,
                categories=X_train[col].cat.categories
            )
        else:
            X_permuted[col] = shuffled_values

        probabilities = model.predict_proba(X_permuted)[:, yes_index]

        permuted_auc = roc_auc_score(
            (y_validation == "Yes").astype(int),
            probabilities
        )

        auc_scores.append(permuted_auc)

    mean_permuted_auc = float(np.mean(auc_scores))
    mean_drop = baseline_auc - mean_permuted_auc

    importance_results.append({
        "Feature": col,
        "Baseline_ROC_AUC": baseline_auc,
        "Mean_Permuted_ROC_AUC": mean_permuted_auc,
        "Importance_ROC_AUC_Drop": mean_drop,
        "Importance_Std": float(np.std(auc_scores, ddof=1))
    })

# 6. Rank the features and save results
importance_df = pd.DataFrame(importance_results)

importance_df = importance_df.sort_values(
    by="Importance_ROC_AUC_Drop",
    ascending=False
).reset_index(drop=True)

output_path = project_dir / "tabpfn_permutation_importance.csv"
importance_df.to_csv(output_path, index=False)

print("\n========== PERMUTATION FEATURE IMPORTANCE ==========")
print(importance_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\nResults saved to:")
print(output_path)
