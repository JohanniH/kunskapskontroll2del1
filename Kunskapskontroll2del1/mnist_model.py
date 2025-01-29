import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load MNIST-data
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
X = mnist["data"]
Y = mnist["target"].astype(np.uint8)

# SHare data ni training, validate and test
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Normalise data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Model 1: Logistic Regression
log_reg = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial', random_state=42)
log_reg.fit(X_train_scaled, Y_train)
log_reg_preds = log_reg.predict(X_val_scaled)
log_reg_acc = accuracy_score(Y_val, log_reg_preds)
print("Logistic Regression Accuracy:", log_reg_acc)

# Model 2: Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, Y_train)
rf_preds = rf_clf.predict(X_val)
rf_acc = accuracy_score(Y_val, rf_preds)
print("Random Forest Accuracy:", rf_acc)

# Validate best model on testdata
best_model = log_reg if log_reg_acc > rf_acc else rf_clf
best_model_name = "Logistic Regression" if log_reg_acc > rf_acc else "Random Forest"

print(f"Bästa modellen: {best_model_name}")
test_preds = best_model.predict(X_test_scaled if best_model_name == "Logistic Regression" else X_test)
print("Test Accuracy:", accuracy_score(Y_test, test_preds))
print("\nClassification Report:")
print(classification_report(Y_test, test_preds))

from joblib import dump
dump(best_model, 'mnist_model.joblib')