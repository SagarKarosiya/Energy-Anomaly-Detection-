import pandas as pd


def preprocess_data(df):

    print("🧹 Preprocessing...")

    # Sort by timestamp
    df = df.sort_values("timestamp")

    # Handle missing values
    df = df.ffill().bfill()

    # Remove duplicate timestamps
    df = df.drop_duplicates(subset="timestamp")

    print("✅ Preprocessing Completed")

    return df
