1. Project Title: Students' Mathematics Score Prediction
    Project Description: The purpose of the project is to identify students who may be weaker in Mathematics before their final examination. This may allow additional support or intervention to be provided to these students.

2. 
The project requires Python and the following libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- PyYAML

3. 
Make sure the dataset is located inside the `data` folder.

The project structure should be:
```text
regression_self_prac/
│
├── data/
│   └── regression_bonus_practice_data.csv
│
├── src/
│   ├── data_preparation.py
│   └── model_training.py
│
├── config.yaml
├── eda.ipynb
├── main.py
├── README.md
└── requirements.txt
```

Execution of the code:
python main.py

4. Description of Logical Steps/Flow of the Pipeline

    The machine learning pipeline follows these steps:

    1. Load the dataset based on config.yaml
    2. Clean the dataset by handling inconsistent and missing values.
    3. Remove columns that are not useful for prediction.
    4. Separate the features from the target variable, `final_test`.
    5. Split the dataset into training and testing sets.
    6. Scale numerical features using `StandardScaler`.
    7. Encode categorical features using `OneHotEncoder`.
    8. Train Simple Linear Regression, Multivariate Linear Regression, Ridge Regression and Lasso Regression models.
    9. Evaluate the models using MAE, MSE, RMSE and R².
    10. Compare training and testing performance to analyse bias and variance.
    11. Use `GridSearchCV` to tune the alpha values of Ridge and Lasso Regression.
    12. Compare the final models and select the most suitable model.

5. Overview of Key Findings from EDA

Exploratory Data Analysis was performed in `eda.ipynb`.

The analysis included:

- Examining the distribution of the `final_test` scores using a histogram.
- Investigating the relationship between attendance rate and final test score using a scatter plot.
- Comparing final test scores for students with and without tuition using a box plot.
- Using a correlation heatmap to examine relationships between numerical variables.
- Checking the dataset for missing values.
- Checking categorical variables for inconsistent values.
- Investigating duplicate student IDs.

The EDA showed that attendance rate has a relationship with the final test score, but attendance alone is not sufficient to accurately predict students' results. This is also reflected in the weaker performance of the Simple Linear Regression model.

Multiple student characteristics were therefore used in the multivariate and regularised regression models.

6. Feature Handling Description

The target variable used in this project is:

```text
final_test
```

Rows with missing `final_test` values were removed because the target value is required for supervised model training.

The following columns were removed:

- `index` - an index value that does not provide useful information for prediction.
- `student_id` - an identifier for individual students rather than a predictive feature.
- `bag_color` - not considered relevant to predicting Mathematics performance.

Missing `attendance_rate` values were replaced using the median attendance rate.

Missing `CCA` values were replaced with `NIL`.

Inconsistent CCA categories were standardised. For example, `ARTS` was changed to `Arts`, `CLUBS` to `Clubs`, and `NONE` to `NIL`.

Tuition values were also standardised so that `Y` became `Yes` and `N` became `No`.

The numerical features used were:

- number_of_siblings
- n_male
- n_female
- age
- hours_per_week
- attendance_rate

These features were scaled using `StandardScaler`.

The categorical features used were:

- direct_admission
- CCA
- learning_style
- gender
- tuition
- mode_of_transport

These features were encoded using `OneHotEncoder`.


7. Explanation of Model Choices

Four regression approaches were investigated.

### Simple Linear Regression

Simple Linear Regression was used as a baseline model using `attendance_rate` as the predictor. This helps determine how well a single feature can predict students' final test scores.

### Multivariate Linear Regression

Multivariate Linear Regression was used to include multiple student characteristics in the prediction. This allows the model to consider several factors instead of relying only on attendance.

### Ridge Regression

Ridge Regression was used as a regularised regression model. Ridge applies L2 regularisation, which can help reduce overfitting by penalising large model coefficients.

### Lasso Regression

Lasso Regression was also tested. It uses L1 regularisation and can reduce some coefficients to zero, which can help with feature selection.

`GridSearchCV` was used to test different alpha values for Ridge and Lasso Regression and identify the best hyperparameter value using cross-validation.

8. Evaluation of Models

The models were evaluated using four regression metrics:

- **MAE** - average absolute difference between the predicted and actual scores.
- **MSE** - average squared prediction error.
- **RMSE** - square root of MSE, expressing prediction error in the same unit as the examination score.
- **R²** - measures how much variation in the final test scores is explained by the model.

The final results were:

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Simple Linear Regression | 10.8002 | 172.6926 | 13.1413 | 0.1103 |
| Multivariate Linear Regression | 7.2390 | 82.3601 | 9.0752 | 0.5757 |
| Tuned Ridge Regression | 7.2391 | 82.3589 | 9.0752 | 0.5757 |
| Tuned Lasso Regression | 7.2406 | 82.3232 | 9.0732 | 0.5759 |

The Simple Linear Regression model performed considerably worse than the other models. This indicates that attendance rate alone is not sufficient for predicting students' final test scores.

Multivariate Linear Regression, tuned Ridge Regression and tuned Lasso Regression produced very similar results.

For Multivariate Linear Regression, the training R² was approximately 0.5771 while the testing R² was approximately 0.5757. The similarity between the training and testing performance suggests that there is no significant overfitting.

Multivariate Linear Regression was selected as the final model because it achieved comparable performance to the regularised models while maintaining a simpler model structure.


