from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
import numpy as np


def train_model(df, model_choice="Isolation Forest"):

    print(f"🤖 Training {model_choice}...")

    X = df.select_dtypes(include=np.number).drop(
        columns=["final_anomaly"], errors="ignore"
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =========================
    # MODEL SELECTION
    # =========================
    if model_choice == "Isolation Forest":
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(X_scaled)
        preds = model.predict(X_scaled)
        scores = model.decision_function(X_scaled)

    elif model_choice == "Local Outlier Factor (LOF)":
        model = LocalOutlierFactor(contamination=0.05)
        preds = model.fit_predict(X_scaled)
        scores = model.negative_outlier_factor_

    elif model_choice == "One-Class SVM":
        model = OneClassSVM(nu=0.05)
        model.fit(X_scaled)
        preds = model.predict(X_scaled)
        scores = model.decision_function(X_scaled)

    elif model_choice == "Robust Covariance":
        model = EllipticEnvelope(contamination=0.05)
        model.fit(X_scaled)
        preds = model.predict(X_scaled)
        scores = model.decision_function(X_scaled)

    else:
        raise ValueError("Invalid model selected")

    # Convert (-1,1) → (1 anomaly, 0 normal)
    df["final_anomaly"] = np.where(preds == -1, 1, 0)

    # Add anomaly score (lower score = more abnormal)
    df["anomaly_score"] = scores

    anomaly_rate = df["final_anomaly"].mean() * 100

    print("✅ Model Training Completed")
    print(f"🔎 Anomaly Rate: {round(anomaly_rate, 2)}%")

    return df, model