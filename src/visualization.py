import matplotlib.pyplot as plt
import seaborn as sns


def plot_energy_with_anomalies(df):

    print("📊 Evoastra Group D  Visualizations...")

    energy_cols = [
        "electricity_cleaned",
        "hotwater_cleaned",
        "chilledwater"
    ]

    for col in energy_cols:

        plt.figure(figsize=(14, 6))

        # Normal points
        normal = df[df['anomaly'] == 0]
        anomalies = df[df['anomaly'] == 1]

        plt.plot(normal['timestamp'], normal[col], label='Normal', alpha=0.6)
        plt.scatter(anomalies['timestamp'], anomalies[col],
                    color='red', label='Anomaly', s=10)

        plt.title(f"{col} Consumption with Anomalies")
        plt.xlabel("Timestamp")
        plt.ylabel("Consumption")
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_anomaly_distribution(df):

    plt.figure(figsize=(6, 4))

    sns.countplot(x='anomaly', data=df)

    plt.title("Anomaly Distribution")
    plt.xticks([0, 1], ['Normal', 'Anomaly'])
    plt.tight_layout()
    plt.show()
