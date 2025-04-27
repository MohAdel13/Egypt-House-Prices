import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import category_encoders as ce
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    """
    Load data file into a DataFrame.

    Args:
        path (str): Path to CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """

    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicates and rows with unknown Price & Type.

    Args:
        df (pd.DataFrame): Raw DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame."""

    df_cleaned = df.drop_duplicates()
    df_cleaned = df_cleaned[(df_cleaned['Price'] != 'Unknown') & (df_cleaned['Type'] != 'Unknown')]
    return df_cleaned


def fix_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix wrong and inconsistent values.

    Args:
        df (pd.DataFrame): DataFrame to fix.

    Returns:
        pd.DataFrame: Updated DataFrame.
    """

    df_fixed = df.copy()

    df_fixed['Price'] = df_fixed['Price'].replace('Unknown', np.nan)

    df_fixed['Bedrooms'] = df_fixed['Bedrooms'].replace('10+', '10')

    df_fixed['Bathrooms'] = df_fixed['Bathrooms'].replace('10+', '10')

    df_fixed['Type'] = df_fixed['Type'].replace('Standalone Villa', 'Stand Alone Villa')
    df_fixed['Type'] = df_fixed['Type'].replace('Twin house', 'Twin House')

    df_fixed['City'] = df_fixed['City'].replace('(View phone number)', 'Unknown')

    df_fixed['Level'] = df_fixed['Level'].replace('Ground', '0')
    df_fixed['Level'] = df_fixed['Level'].replace('10+', 'Highest')
    max_levels = df_fixed[(df_fixed['Level'] != 'Highest') &
                          (df_fixed['Level'] != 'Unknown')].groupby('Type')['Level'].max()
    df_fixed['Level'] = df_fixed.apply(lambda row: max_levels[row['Type']] if row['Level'] == 'Highest' else row['Level'], axis=1)
    df_fixed['Level'] = df_fixed['Level'].replace('Unknown', np.nan)

    df_fixed['Delivery_Date'] = df_fixed['Delivery_Date'].replace({
        'Ready to move': '0',
        'soon': '3',
        'within 6 months': '6',
        '2022': '12',
        '2023': '24',
        '2024': '36',
        '2025': '48',
        '2026': '60',
        '2027': '72',
        'Unknown': np.nan
    })

    return df_fixed


def type_convert(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to numeric types.

    Args:
        df (pd.DataFrame): DataFrame to convert.

    Returns:
        pd.DataFrame: Converted DataFrame.
    """

    df_converted = df.copy()

    df_converted['Price'] = df_converted['Price'].astype(float)
    df_converted['Bedrooms'] = df_converted['Bedrooms'].astype(float)
    df_converted['Bathrooms'] = df_converted['Bathrooms'].astype(float)
    df_converted['Area'] = df_converted['Area'].astype(float)
    df_converted['Level'] = df_converted['Level'].astype(float)
    df_converted['Delivery_Date'] = df_converted['Delivery_Date'].astype(float)
    return df_converted


def range_fix(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Fix values slightly outside valid range and set outliers as NaN.

    Args:
        df (pd.DataFrame): DataFrame to fix.
        column (str): Column to process.

    Returns:
        pd.DataFrame: Updated DataFrame.
    """

    # Specifying valid range for each column values
    if column == 'Bedrooms':
        valid_ranges = {
            "Chalet": (1, 3),
            "Apartment": (1, 4),
            "Studio": (1, 2),
            "Penthouse": (1, 4),
            "Duplex": (2, 8),
            "Stand Alone Villa": (3, 6),
            "Twin House": (3, 6),
            "Town House": (3, 6),
        }
    elif column == 'Bathrooms':
        valid_ranges = {
            "Chalet": (1, 3),
            "Apartment": (1, 3),
            "Studio": (1, 2),
            "Penthouse": (1, 3),
            "Duplex": (2, 6),
            "Stand Alone Villa": (2, 5),
            "Twin House": (2, 5),
            "Town House": (2, 5)
        }
    else:
        valid_ranges = {
            "Chalet": (30, 180),
            "Apartment": (60, 250),
            "Studio": (30, 70),
            "Penthouse": (100, 240),
            "Duplex": (150, 500),
            "Stand Alone Villa": (180, 400),
            "Town House": (150, 500),
            "Twin House": (150, 500)
        }

    df_in_range = df.copy()

    for prop_type, (min_val, max_val) in valid_ranges.items():
        mask = df["Type"] == prop_type

        # Clip values within a 10% margin
        df_in_range.loc[mask & (df_in_range[column] < min_val) &
                        (df_in_range[column] >= min_val * 0.9), column] = min_val
        df_in_range.loc[mask & (df_in_range[column] > max_val) &
                        (df_in_range[column] <= max_val * 1.1), column] = max_val

        # Converting extreme outliers to nans
        df_in_range.loc[(mask & ((df_in_range[column] < min_val * 0.9) |
                                 (df_in_range[column] > max_val * 1.1))), column] = np.nan
    return df_in_range


def outlier_handling(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Handle outliers using IQR method.

    Args:
        df (pd.DataFrame): DataFrame to process.
        column (str): Column to check.

    Returns:
        pd.DataFrame: Updated DataFrame.
    """

    df_handled = df.copy()

    if column != 'Price':
        df_handled = range_fix(df_handled, column)

    property_types = df_handled["Type"].unique()

    for i, prop_type in enumerate(property_types):
        data = df_handled[(df_handled["Type"] == prop_type) & ~df_handled[column].isna()][column]

        if len(data) == 0:
            continue

        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        if column == 'Price':
            df_handled = df_handled[~((df_handled["Type"] == prop_type) &
                                      ((data < lower_bound) | (data > upper_bound)))]

        else:
            df_handled.loc[(df_handled["Type"] == prop_type) & ((data < lower_bound) | (data > upper_bound)), column] = np.nan

    return df_handled


def impute_missing_values(df: pd.DataFrame, columns_to_impute: list,
                          grouping_columns: list) -> pd.DataFrame:
    """
    Fill missing values with median grouped by columns.

    Args:
        df (pd.DataFrame): DataFrame to process.
        columns_to_impute (list): Columns to fill.
        grouping_columns (list): Columns to group by.

    Returns:
        pd.DataFrame: Updated DataFrame.
    """

    df_imputed = df.copy()

    df_imputed[columns_to_impute] = df_imputed.groupby(grouping_columns)[columns_to_impute].transform(
        lambda x: x.fillna(x[x.notnull()].median())
    )

    return df_imputed


def preprocess_cycle(path: str) -> pd.DataFrame:
    """
    Full data preprocessing cycle from loading to cleaning.

    Args:
        path (str): Path to CSV file.

    Returns:
        pd.DataFrame: Fully processed DataFrame.
    """

    loaded_data = load_data(path)

    cleaned_data = clean_data(loaded_data)

    fixed_data = fix_values(cleaned_data)

    converted_data = type_convert(fixed_data)

    columns_of_outliers = ['Price', 'Bedrooms', 'Bathrooms', 'Area']

    handled_data = converted_data.copy()
    for i in columns_of_outliers:
        handled_data = outlier_handling(handled_data, i)

    columns_to_impute = ['Price', 'Bedrooms', 'Bathrooms', 'Area', 'Level', 'Delivery_Date']
    grouping_columns = ['Type', 'Compound', 'City']

    imputed_data = handled_data.copy()

    for i in range(len(grouping_columns), 0, -1):
        imputed_data = impute_missing_values(imputed_data, columns_to_impute, grouping_columns[:i])

    ready_data = imputed_data.dropna()

    return ready_data


def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p transformation to Price.

    Args:
        df (pd.DataFrame): DataFrame with Price column.

    Returns:
        pd.DataFrame: Updated DataFrame.
    """

    df_transformed = df.copy()

    df_transformed["Log_Price"] = np.log1p(df_transformed["Price"])
    df_transformed = df_transformed.drop(columns=['Price'])
    return df_transformed


def split_data(df: pd.DataFrame) -> tuple:
    """
    Split features and target into train and test sets.

    Args:
        df (pd.DataFrame): DataFrame to split.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """

    X = df.drop(columns=['Log_Price'], axis=1)
    y = df['Log_Price']

    return train_test_split(X, y, test_size=0.2, random_state=42)


def features_operations(df: pd.DataFrame) -> tuple:
    """
        Run feature engineering operations and prepare preprocessing..

        Args:
            df (pd.DataFrame): DataFrame to process.

        Returns:
            tuple: (preprocessor, X_train, X_test, y_train, y_test)
        """

    df_transformed = log_transform(df)

    x_train, x_test, y_train, y_test = split_data(df_transformed)

    return preprocessor_make(x_train), x_train, x_test, y_train, y_test


def preprocessor_make(x_train):
    """
    Create a preprocessing pipeline.

    Args:
        x_train (pd.DataFrame): Training features.

    Returns:
        ColumnTransformer: Preprocessor pipeline.
    """

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
