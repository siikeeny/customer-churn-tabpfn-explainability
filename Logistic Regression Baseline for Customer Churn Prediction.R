# Logistic Regression Baseline for Customer Churn Prediction

# Step 1: Load the training and test datasets
churn_train <- read.csv(
  "churn_train.csv",
  stringsAsFactors = FALSE
)

churn_test <- read.csv(
  "churn_test.csv",
  stringsAsFactors = FALSE
)

# Step 2: Convert the target variable to a factor
churn_train$Churn <- factor(
  churn_train$Churn,
  levels = c("No", "Yes")
)

churn_test$Churn <- factor(
  churn_test$Churn,
  levels = c("No", "Yes")
)

# Step 3: Convert categorical predictors to factors
categorical_cols <- names(churn_train)[
  sapply(churn_train, is.character)
]

for (col in categorical_cols) {
  all_levels <- unique(c(
    churn_train[[col]],
    churn_test[[col]]
  ))

  churn_train[[col]] <- factor(
    churn_train[[col]],
    levels = all_levels
  )

  churn_test[[col]] <- factor(
    churn_test[[col]],
    levels = all_levels
  )
}

# Step 4: Train the Logistic Regression model
log_model <- glm(
  Churn ~ .,
  data = churn_train,
  family = binomial()
)

# Step 5: Predict churn probabilities on the test set
test_prob <- predict(
  log_model,
  newdata = churn_test,
  type = "response"
)

# Step 6: Convert probabilities into class predictions
test_pred <- factor(
  ifelse(test_prob >= 0.5, "Yes", "No"),
  levels = c("No", "Yes")
)

# Step 7: Generate the confusion matrix
conf_matrix <- table(
  Actual = churn_test$Churn,
  Predicted = test_pred
)

cat("Confusion Matrix:\n")
print(conf_matrix)

# Step 8: Calculate evaluation metrics
TN <- conf_matrix["No", "No"]
FP <- conf_matrix["No", "Yes"]
FN <- conf_matrix["Yes", "No"]
TP <- conf_matrix["Yes", "Yes"]

accuracy <- (TP + TN) / sum(conf_matrix)

precision <- ifelse(
  TP + FP == 0,
  0,
  TP / (TP + FP)
)

recall <- ifelse(
  TP + FN == 0,
  0,
  TP / (TP + FN)
)

f1 <- ifelse(
  precision + recall == 0,
  0,
  2 * precision * recall / (precision + recall)
)

# Step 9: Calculate ROC-AUC
if (!requireNamespace("pROC", quietly = TRUE)) {
  install.packages("pROC")
}

roc_result <- pROC::roc(
  churn_test$Churn,
  test_prob,
  levels = c("No", "Yes"),
  direction = "<",
  quiet = TRUE
)

auc_value <- as.numeric(pROC::auc(roc_result))

# Step 10: Display the results
cat("\nLOGISTIC REGRESSION TEST RESULTS\n")
cat("Accuracy:  ", round(accuracy, 4), "\n")
cat("Precision: ", round(precision, 4), "\n")
cat("Recall:    ", round(recall, 4), "\n")
cat("F1-score:  ", round(f1, 4), "\n")
cat("ROC-AUC:   ", round(auc_value, 4), "\n")

# Step 11: Save the evaluation metrics
results <- data.frame(
  Model = "Logistic Regression",
  Accuracy = accuracy,
  Precision = precision,
  Recall = recall,
  F1_Score = f1,
  ROC_AUC = auc_value
)

write.csv(
  results,
  "logistic_regression_test_results.csv",
  row.names = FALSE
)

cat("\nResults saved to logistic_regression_test_results.csv\n")
