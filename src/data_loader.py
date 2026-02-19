import os
import pandas as pd


def load_data(data_path):

    print("🚀 Loading Data...")

    files = [
        "electricity_cleaned.csv",
        "hotwater_cleaned.csv",
        "chilledwater.csv"
    ]

    dfs = []

    for file in files:
        print(f"Processing: {file}")

        df = pd.read_csv(os.path.join(data_path, file))

        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Get value column (besides timestamp)
        value_col = [col for col in df.columns if col != "timestamp"][0]

        # Rename to clean file name
        new_name = file.replace(".csv", "")
        df = df[['timestamp', value_col]]
        df = df.rename(columns={value_col: new_name})

        dfs.append(df)

    # Merge all dataframes on timestamp
    df_final = dfs[0]
    for df in dfs[1:]:
        df_final = pd.merge(df_final, df, on="timestamp", how="inner")

    print("✅ Final Shape:", df_final.shape)
    print("✅ Columns:", df_final.columns)

    return df_final
