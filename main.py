# main.py
import yaml
import pandas as pd

from src.data_preparation import prepare_data

from src.model_training import (
    train_simple_model,
    train_multivariate_model,
    bias_variance_analysis,
    train_ridge_model,
    train_lasso_model,
    tune_ridge_model,
    tune_lasso_model,
    evaluate_model
)


# Prepare Data

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

file_path = config['data']['file_path']

X_train, X_test, y_train, y_test, preprocessor = prepare_data(
    file_path,
    config['data']['test_size'],
    config['data']['random_state']
)

#  Simple Linear Regression

simple_model, y_pred_simple = train_simple_model(
    X_train,
    X_test,
    y_train
)

mae_simple, mse_simple, rmse_simple, r2_simple = evaluate_model(
    y_test,
    y_pred_simple
)

print("\nSimple Linear Regression")
print("MAE:", mae_simple)
print("MSE:", mse_simple)
print("RMSE:", rmse_simple)
print("R²:", r2_simple)


#  Multivariate Linear Regression

multi_model, y_pred_multi = train_multivariate_model(
    X_train,
    X_test,
    y_train,
    preprocessor
)

mae_multi, mse_multi, rmse_multi, r2_multi = evaluate_model(
    y_test,
    y_pred_multi
)

print("\nMultivariate Linear Regression")
print("MAE:", mae_multi)
print("MSE:", mse_multi)
print("RMSE:", rmse_multi)
print("R²:", r2_multi)


# Bias-Variance Analysis

y_train_pred_multi, y_test_pred_multi = bias_variance_analysis(
    multi_model,
    X_train,
    X_test
)

train_mae, train_mse, train_rmse, train_r2 = evaluate_model(
    y_train,
    y_train_pred_multi
)

test_mae, test_mse, test_rmse, test_r2 = evaluate_model(
    y_test,
    y_test_pred_multi
)

bias_variance_results = pd.DataFrame({
    'Dataset': ['Training', 'Testing'],
    'MAE': [train_mae, test_mae],
    'MSE': [train_mse, test_mse],
    'RMSE': [train_rmse, test_rmse],
    'R²': [train_r2, test_r2]
})

print("\nBias-Variance Analysis")
print(bias_variance_results)


# Ridge Regression

ridge_model, y_pred_ridge = train_ridge_model(
    X_train,
    X_test,
    y_train,
    preprocessor,
    config['model']['default_alpha']
)

mae_ridge, mse_ridge, rmse_ridge, r2_ridge = evaluate_model(
    y_test,
    y_pred_ridge
)

print("\nRidge Regression")
print("MAE:", mae_ridge)
print("MSE:", mse_ridge)
print("RMSE:", rmse_ridge)
print("R²:", r2_ridge)


# Lasso Regression

lasso_model, y_pred_lasso = train_lasso_model(
    X_train,
    X_test,
    y_train,
    preprocessor,
    config['model']['default_alpha']
)

mae_lasso, mse_lasso, rmse_lasso, r2_lasso = evaluate_model(
    y_test,
    y_pred_lasso
)

print("\nLasso Regression")
print("MAE:", mae_lasso)
print("MSE:", mse_lasso)
print("RMSE:", rmse_lasso)
print("R²:", r2_lasso)


# Tune Ridge Regression

best_ridge_model, y_pred_ridge_tuned, ridge_params, ridge_cv_score = (
    tune_ridge_model(
        X_train,
        X_test,
        y_train,
        preprocessor,
        config['tuning']['alpha_values'],
        config['tuning']['cv_folds'],
        config['tuning']['scoring']
    )
)

mae_ridge_tuned, mse_ridge_tuned, rmse_ridge_tuned, r2_ridge_tuned = (
    evaluate_model(
        y_test,
        y_pred_ridge_tuned
    )
)

print("\nTuned Ridge Regression")
print("Best Parameters:", ridge_params)
print("Best CV Score:", ridge_cv_score)
print("MAE:", mae_ridge_tuned)
print("MSE:", mse_ridge_tuned)
print("RMSE:", rmse_ridge_tuned)
print("R²:", r2_ridge_tuned)


# Tune Lasso Regression

best_lasso_model, y_pred_lasso_tuned, lasso_params, lasso_cv_score = (
    tune_lasso_model(
        X_train,
        X_test,
        y_train,
        preprocessor,
        config['tuning']['alpha_values'],
        config['tuning']['cv_folds'],
        config['tuning']['scoring']
    )
)

mae_lasso_tuned, mse_lasso_tuned, rmse_lasso_tuned, r2_lasso_tuned = (
    evaluate_model(
        y_test,
        y_pred_lasso_tuned
    )
)

print("\nTuned Lasso Regression")
print("Best Parameters:", lasso_params)
print("Best CV Score:", lasso_cv_score)
print("MAE:", mae_lasso_tuned)
print("MSE:", mse_lasso_tuned)
print("RMSE:", rmse_lasso_tuned)
print("R²:", r2_lasso_tuned)


# Final Model Comparison

evaluation_results = pd.DataFrame({
    'Model': [
        'Simple Linear Regression',
        'Multivariate Linear Regression',
        'Tuned Ridge Regression',
        'Tuned Lasso Regression'
    ],
    'MAE': [
        mae_simple,
        mae_multi,
        mae_ridge_tuned,
        mae_lasso_tuned
    ],
    'MSE': [
        mse_simple,
        mse_multi,
        mse_ridge_tuned,
        mse_lasso_tuned
    ],
    'RMSE': [
        rmse_simple,
        rmse_multi,
        rmse_ridge_tuned,
        rmse_lasso_tuned
    ],
    'R²': [
        r2_simple,
        r2_multi,
        r2_ridge_tuned,
        r2_lasso_tuned
    ]
})

print("\nFinal Model Comparison")
print(evaluation_results)