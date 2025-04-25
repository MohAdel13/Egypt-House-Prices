import numpy as np
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df[(df['Price'] != 'Unknown') & (df['Type'] != 'Unknown')]
    return df


def fix_values(df: pd.DataFrame) -> pd.DataFrame:
    df['Price'] = df['Price'].replace('Unknown', np.nan)

    df['Bedrooms'] = df['Bedrooms'].replace('10+', '10')

    df['Bathrooms'] = df['Bathrooms'].replace('10+', '10')

    df['Type'] = df['Type'].replace('Standalone Villa', 'Stand Alone Villa')
    df['Type'] = df['Type'].replace('Twin house', 'Twin House')

    df['City'] = df['City'].replace('(View phone number)', 'Unknown')

    df['Level'] = df['Level'].replace('Ground', '0')
    df['Level'] = df['Level'].replace('10+', 'Highest')
    max_levels = df[(df['Level'] != 'Highest') & (df['Level'] != 'Unknown')].groupby('Type')['Level'].max()
    df['Level'] = df.apply(lambda row: max_levels[row['Type']] if row['Level'] == 'Highest' else row['Level'], axis=1)
    df['Level'] = df['Level'].replace('Unknown', np.nan)

    df['Delivery_Date'] = df['Delivery_Date'].replace({
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

    return df


def type_convert(df):
    df['Price'] = df['Price'].astype(float)
    df['Bedrooms'] = df['Bedrooms'].astype(float)
    df['Bathrooms'] = df['Bathrooms'].astype(float)
    df['Area'] = df['Area'].astype(float)
    df['Level'] = df['Level'].astype(float)
    df['Delivery_Date'] = df['Delivery_Date'].astype(float)
    return df


def range_fix(df, column):
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

    for prop_type, (min_val, max_val) in valid_ranges.items():
        mask = df["Type"] == prop_type

        # Clip values within a 10% margin
        df.loc[mask & (df[column] < min_val) & (df[column] >= min_val * 0.9), column] = min_val
        df.loc[mask & (df[column] > max_val) & (df[column] <= max_val * 1.1), column] = max_val

        # Converting extreme outliers to nans
        df.loc[(mask & ((df[column] < min_val * 0.9) | (df[column] > max_val * 1.1)))] = np.nan
    return df


def outlier_handling(df, column):
    if column != 'Price':
        df = range_fix(df, column)

    property_types = df["Type"].unique()

    for i, prop_type in enumerate(property_types):
        data = df[(df["Type"] == prop_type) & ~df[column].isna()][column]

        if len(data) == 0:
            continue

        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        if column == 'Price':
            df = df[~((df["Type"] == prop_type) & ((data < lower_bound) | (data > upper_bound)))]

        else:
            df[(df["Type"] == prop_type) & ((data < lower_bound) | (data > upper_bound))][
                column] = np.nan
    return df


def impute_missing_values(df, columns_to_impute, grouping_columns):
    # Apply median imputation
    df[columns_to_impute] = df.groupby(grouping_columns)[columns_to_impute].transform(
        lambda x: x.fillna(x[x.notnull()].median())
    )

    return df


def preprocess_cycle(path):
    df = load_data(path)

    df = clean_data(df)

    df = fix_values(df)

    df = type_convert(df)

    columns_of_outliers = ['Price', 'Bedrooms', 'Bathrooms', 'Area']
    for i in columns_of_outliers:
        outlier_handling(df, i)

    columns_to_impute = ['Price', 'Bedrooms', 'Bathrooms', 'Area', 'Level', 'Delivery_Date']
    grouping_columns = ['Type', 'Compound', 'City']
    for i in range(len(grouping_columns), 0, -1):
        df = impute_missing_values(df, columns_to_impute, grouping_columns[:i])

    df = df.dropna()
    return df
