# src/model_training.py

import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Simple Linear Regression
def train_simple_model(X_train, X_test, y_train):

    X_train_simple = X_train[['attendance_rate']]
    X_test_simple = X_test[['attendance_rate']]

    simple_model = LinearRegression()

    simple_model.fit(X_train_simple, y_train)

    y_pred_simple = simple_model.predict(X_test_simple)

    return simple_model, y_pred_simple


# Multivariate Linear Regression
def train_multivariate_model(X_train, X_test, y_train, preprocessor):

    multi_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    multi_model.fit(X_train, y_train)

    y_pred_multi = multi_model.predict(X_test)

    return multi_model, y_pred_multi

# Bias-Variance Analysis
def bias_variance_analysis(model, X_train, X_test):

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    return y_train_pred, y_test_pred


# Ridge Regression
def train_ridge_model(X_train, X_test, y_train, preprocessor, default_alpha):

    ridge_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=default_alpha))
    ])

    ridge_model.fit(X_train, y_train)

    y_pred_ridge = ridge_model.predict(X_test)

    return ridge_model, y_pred_ridge


# Lasso Regression
def train_lasso_model(X_train, X_test, y_train, preprocessor, default_alpha):

    lasso_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Lasso(alpha=default_alpha))
    ])

    lasso_model.fit(X_train, y_train)

    y_pred_lasso = lasso_model.predict(X_test)

    return lasso_model, y_pred_lasso

# Tune Ridge Regression
def tune_ridge_model(X_train, X_test, y_train, preprocessor, alpha_values, cv_folds, scoring):

    ridge_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge())
    ])

    ridge_param_grid = {
    'regressor__alpha': alpha_values
    }

    ridge_grid_search = GridSearchCV(
        estimator=ridge_model,
        param_grid=ridge_param_grid,
        cv=cv_folds,
        scoring=scoring
    )

    ridge_grid_search.fit(X_train, y_train)

    best_ridge_model = ridge_grid_search.best_estimator_

    y_pred_ridge_tuned = best_ridge_model.predict(X_test)

    return (
        best_ridge_model,
        y_pred_ridge_tuned,
        ridge_grid_search.best_params_,
        ridge_grid_search.best_score_
    )


# Tune Lasso Regression
def tune_lasso_model(X_train, X_test, y_train, preprocessor, alpha_values, cv_folds, scoring):

    lasso_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Lasso())
    ])

    lasso_param_grid = {
        'regressor__alpha': alpha_values
    }

    lasso_grid_search = GridSearchCV(
        estimator=lasso_model,
        param_grid=lasso_param_grid,
        cv=cv_folds,
        scoring=scoring
    )

    lasso_grid_search.fit(X_train, y_train)

    best_lasso_model = lasso_grid_search.best_estimator_

    y_pred_lasso_tuned = best_lasso_model.predict(X_test)

    return (
        best_lasso_model,
        y_pred_lasso_tuned,
        lasso_grid_search.best_params_,
        lasso_grid_search.best_score_
    )

# Model Evaluation
def evaluate_model(y_test, y_pred):

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return mae, mse, rmse, r2