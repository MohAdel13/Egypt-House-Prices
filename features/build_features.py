import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import category_encoders as ce
from sklearn.preprocessing import StandardScaler


def log_transform(df):
    df["Log_Price"] = np.log1p(df["Price"])
    df = df.drop(columns=['Price'])
    return df


def split_data(df, test_size=0.2, random_state=42):
    X = df.drop(columns=['Log_Price'], axis=1)
    y = df['Log_Price']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def features_operations(df):
    df = log_transform(df)
    x_train, x_test, y_train, y_test = split_data(df)
    return preprocessor_make(x_train), x_train, x_test, y_train, y_test


def preprocessor_make(x_train):

    numeric_features = x_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = x_train.select_dtypes(include=['object']).columns

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())

    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
        , ("binary_encode", ce.BinaryEncoder())

    ])

    # Combine preprocessors in a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features)
            , ('num', numerical_transformer, numeric_features)
        ]
    )
    return preprocessor

