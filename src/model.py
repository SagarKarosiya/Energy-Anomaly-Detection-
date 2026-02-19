from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def train_model(df):

    print("🤖 Training Isolation Forest Model...")

    # Remove timestamp column
    X = df.drop(columns=["timestamp"])

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Isolation Forest
    model = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42
    )

    df['anomaly'] = model.fit_predict(X_scaled)

    # Convert -1 to 1 (anomaly) and 1 to 0 (normal)
    df['anomaly'] = df['anomaly'].apply(lambda x: 1 if x == -1 else 0)

    print("✅ Model Training Completed")

    return df, model
