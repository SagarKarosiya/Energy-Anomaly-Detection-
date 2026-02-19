import pandas as pd


def feature_engineering(df):

    print("⚙️ Feature Engineering...")

    energy_cols = [
        "electricity_cleaned",
        "hotwater_cleaned",
        "chilledwater"
    ]

    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month

    # Rolling features (24 hour window)
    for col in energy_cols:
        df[f'{col}_roll_mean_24'] = df[col].rolling(24).mean()
        df[f'{col}_roll_std_24'] = df[col].rolling(24).std()

    # Drop rows with NaN after rolling
    df = df.dropna()

    print("✅ Feature Engineering Completed")
    print("Total Features:", df.shape[1])

    return df
