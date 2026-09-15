# src/data_preparation.py

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def prepare_data(file_path, test_size, random_state):


    # Load dataset

    df = pd.read_csv(file_path)
    # Data Cleaning


    df['CCA'] = df['CCA'].replace({
    'SPORTS': 'Sports',
    'ARTS': 'Arts',
    'CLUBS': 'Clubs',
    'NONE': 'NIL'
})
    
    columns_to_drop = [
        'index',
        'student_id',
        'bag_color'
    ]

    df = df.drop(
        columns=[
            col for col in columns_to_drop
            if col in df.columns
        ]
    )


    df = df.dropna(subset=['final_test'])

    df['tuition'] = df['tuition'].replace({
        'Y': 'Yes',
        'N': 'No'
    })

    df['attendance_rate'] = df['attendance_rate'].fillna(
    df['attendance_rate'].median()
)

    # Handle Remaining Missing Values

    df['CCA'] = df['CCA'].fillna('NIL')

    # Define Features and Target

    X = df.drop(columns=['final_test'])
    y = df['final_test']


    # Train-Test Split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )


    # Define Numerical Features

    numerical_features = [
        'number_of_siblings',
        'n_male',
        'n_female',
        'age',
        'hours_per_week',
        'attendance_rate'
    ]


    
    # Define Categorical Features

    categorical_features = [
        'direct_admission',
        'CCA',
        'learning_style',
        'gender',
        'tuition',
        'mode_of_transport'
    ]


    
    # Numerical Preprocessing

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])


    #  Categorical Preprocessing

    categorical_transformer = Pipeline(steps=[
        (
            'onehot',
            OneHotEncoder(handle_unknown='ignore')
        )
    ])


    # Combine Preprocessing

    preprocessor = ColumnTransformer(
        transformers=[
            (
                'num',
                numerical_transformer,
                numerical_features
            ),
            (
                'cat',
                categorical_transformer,
                categorical_features
            )
        ]
    )


    #  Return Prepared Data

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )