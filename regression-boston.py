
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv"
data = pd.read_csv(url, header=None)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
ridge_model = Ridge(alpha = 500, random_state=42)
lasso_model = Lasso(alpha = 5, random_state=42)
rf_model = RandomForestRegressor(n_estimators = 10, max_depth = 2, random_state=42)

linear_model.fit(X_train, y_train)
y_pred_linear_untuned = linear_model.predict(X_test)

ridge_model.fit(X_train, y_train)
y_pred_ridge_untuned = ridge_model.predict(X_test)

lasso_model.fit(X_train, y_train)
y_pred_lasso_untuned = lasso_model.predict(X_test)

rf_model.fit(X_train, y_train)
y_pred_rf_untuned = rf_model.predict(X_test)


def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2


mse_linear_untuned, r2_linear_untuned = evaluate_model(y_test, y_pred_linear_untuned)
mse_ridge_untuned, r2_ridge_untuned = evaluate_model(y_test, y_pred_ridge_untuned)
mse_lasso_untuned, r2_lasso_untuned = evaluate_model(y_test, y_pred_lasso_untuned)
mse_rf_untuned, r2_rf_untuned = evaluate_model(y_test, y_pred_rf_untuned)

ridge_param_grid = {'alpha': [0.1, 1, 10, 100, 500]}
lasso_param_grid = {'alpha': [0.1, 1, 10, 100, 500]}

rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

ridge_grid_search = GridSearchCV(estimator=ridge_model, param_grid=ridge_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
ridge_grid_search.fit(X_train, y_train)
y_pred_ridge_tuned = ridge_grid_search.predict(X_test)

lasso_grid_search = GridSearchCV(estimator=lasso_model, param_grid=lasso_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
lasso_grid_search.fit(X_train, y_train)
y_pred_lasso_tuned = lasso_grid_search.predict(X_test)

rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
rf_grid_search.fit(X_train, y_train)
y_pred_rf_tuned = rf_grid_search.predict(X_test)

mse_ridge_tuned, r2_ridge_tuned = evaluate_model(y_test, y_pred_ridge_tuned)
mse_lasso_tuned, r2_lasso_tuned = evaluate_model(y_test, y_pred_lasso_tuned)
mse_rf_tuned, r2_rf_tuned = evaluate_model(y_test, y_pred_rf_tuned)

print("\n=== Best Hyperparameters ===")
print("Best Parameters for Ridge Regression:", ridge_grid_search.best_params_)
print("Best Parameters for Lasso Regression:", lasso_grid_search.best_params_)
print("Best Parameters for Random Forest:", rf_grid_search.best_params_)

print("=== Untuned Models ===")
print("Linear Regression - MSE:", mse_linear_untuned, "R2:", r2_linear_untuned)
print("Ridge Regression - MSE:", mse_ridge_untuned, "R2:", r2_ridge_untuned)
print("Lasso Regression - MSE:", mse_lasso_untuned, "R2:", r2_lasso_untuned)
print("Random Forest - MSE:", mse_rf_untuned, "R2:", r2_rf_untuned)

print("\n=== Tuned Models ===")
print("Ridge Regression (Tuned) - MSE:", mse_ridge_tuned, "R2:", r2_ridge_tuned)
print("Lasso Regression (Tuned) - MSE:", mse_lasso_tuned, "R2:", r2_lasso_tuned)
print("Random Forest (Tuned) - MSE:", mse_rf_tuned, "R2:", r2_rf_tuned)


model_names = ['Linear Regression', 'Ridge Regression (Untuned)', 'Lasso Regression (Untuned)', 'Random Forest (Untuned)',
               'Ridge Regression (Tuned)', 'Lasso Regression (Tuned)', 'Random Forest (Tuned)']
mse_values = [mse_linear_untuned, mse_ridge_untuned, mse_lasso_untuned, mse_rf_untuned,
              mse_ridge_tuned, mse_lasso_tuned, mse_rf_tuned]
r2_values = [r2_linear_untuned, r2_ridge_untuned, r2_lasso_untuned, r2_rf_untuned,
             r2_ridge_tuned, r2_lasso_tuned, r2_rf_tuned]

results_df = pd.DataFrame({
    'Model': model_names,
    'MSE': mse_values,
    'R2': r2_values
})

plt.figure(figsize=(10, 6))
sns.barplot(x='MSE', y='Model', data=results_df, palette="viridis")
plt.title('Mean Squared Error (MSE) for Each Model')
plt.xlabel('Mean Squared Error')
plt.ylabel('Model')

plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='R2', y='Model', data=results_df, palette="coolwarm")
plt.title('R2 Score for Each Model')
plt.xlabel('R2 Score')
plt.ylabel('Model')

plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

def plot_predicted_vs_actual_subplot(y_true, y_pred, model_name, ax):
    ax.scatter(y_true, y_pred, alpha=0.7, label=model_name)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='black', lw=2, linestyle='--')
    ax.set_title(f'{model_name}')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.legend()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

plot_predicted_vs_actual_subplot(y_test, y_pred_linear_untuned, 'Linear Regression (Untuned)', axes[0])
plot_predicted_vs_actual_subplot(y_test, y_pred_ridge_untuned, 'Ridge Regression (Untuned)', axes[1])
plot_predicted_vs_actual_subplot(y_test, y_pred_lasso_untuned, 'Lasso Regression (Untuned)', axes[2])
plot_predicted_vs_actual_subplot(y_test, y_pred_rf_untuned, 'Random Forest (Untuned)', axes[3])

plt.suptitle('Untuned Models - Predicted vs Actual', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

plot_predicted_vs_actual_subplot(y_test, y_pred_ridge_tuned, 'Ridge Regression (Tuned)', axes[0])
plot_predicted_vs_actual_subplot(y_test, y_pred_lasso_tuned, 'Lasso Regression (Tuned)', axes[1])
plot_predicted_vs_actual_subplot(y_test, y_pred_rf_tuned, 'Random Forest (Tuned)', axes[2])

axes[3].axis('off')

plt.suptitle('Tuned Models - Predicted vs Actual', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()


