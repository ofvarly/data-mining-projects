
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine

data = load_wine()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

missing_values = X.isnull().sum()

print("Missing values in each column:")
print(missing_values)

random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

dt_model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 2, random_state=random_state)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)


dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
dt_grid_search = GridSearchCV(DecisionTreeClassifier(random_state=random_state), dt_param_grid, cv=5, n_jobs=-1, verbose=1)
dt_grid_search.fit(X_train, y_train)
y_pred_dt_tuned = dt_grid_search.best_estimator_.predict(X_test)

rf_model = RandomForestClassifier(n_estimators = 10, max_depth = 10, random_state=random_state)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

rf_param_grid = {
    'n_estimators': [10, 25, 50, 75, 100, 125, 150, 200, 500],
    'max_depth': [None, 2, 5, 10, 15, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=random_state), rf_param_grid, cv=5, n_jobs=-1, verbose=1)
rf_grid_search.fit(X_train, y_train)
y_pred_rf_tuned = rf_grid_search.best_estimator_.predict(X_test)


svm_model = SVC(kernel = 'linear', random_state=random_state)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

svm_param_grid = {
    'C': [0.1, 0.3, 0.5, 1, 3, 5, 7, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
svm_grid_search = GridSearchCV(SVC(random_state=random_state), svm_param_grid, cv=5, n_jobs=-1, verbose=1)
svm_grid_search.fit(X_train, y_train)
y_pred_svm_tuned = svm_grid_search.best_estimator_.predict(X_test)

print("\nBest parameters for each model after tuning:")
print(f"Decision Tree Best Params: {dt_grid_search.best_params_}")
print(f"Random Forest Best Params: {rf_grid_search.best_params_}")
print(f"SVM Best Params: {svm_grid_search.best_params_}")

def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

metrics = {
    'Decision Tree (Untuned)': evaluate_model(y_test, y_pred_dt),
    'Decision Tree (Tuned)': evaluate_model(y_test, y_pred_dt_tuned),
    'Random Forest (Untuned)': evaluate_model(y_test, y_pred_rf),
    'Random Forest (Tuned)': evaluate_model(y_test, y_pred_rf_tuned),
    'SVM (Untuned)': evaluate_model(y_test, y_pred_svm),
    'SVM (Tuned)': evaluate_model(y_test, y_pred_svm_tuned)
}

for model, (accuracy, precision, recall, f1) in metrics.items():
    print(f"{model} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

# Plot Confusion Matrices for comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
models = ['Decision Tree (Untuned)', 'Decision Tree (Tuned)', 'Random Forest (Untuned)', 'Random Forest (Tuned)', 'SVM (Untuned)', 'SVM (Tuned)']
predictions = [y_pred_dt, y_pred_dt_tuned, y_pred_rf, y_pred_rf_tuned, y_pred_svm, y_pred_svm_tuned]

for ax, model, y_pred in zip(axes.flatten(), models, predictions):
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'{model} Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.show()
