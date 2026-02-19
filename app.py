import streamlit as st
import plotly.express as px

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import feature_engineering
from src.model import train_model


DATA_PATH = "data/"

st.set_page_config(
    page_title="Energy Anomaly Detection",
    layout="wide"
)

st.title("⚡ Energy Anomaly Detection Dashboard")
st.markdown("Industrial ML Model using Isolation Forest")

# -------------------------------
# Load and Process Data
# -------------------------------

@st.cache_data
def load_pipeline():
    df = load_data(DATA_PATH)
    df = preprocess_data(df)
    df = feature_engineering(df)
    df, model = train_model(df)
    return df

with st.spinner("Running ML Pipeline..."):
    df = load_pipeline()

st.success("Model Loaded Successfully ✅")

# -------------------------------
# Anomaly Summary
# -------------------------------

col1, col2, col3 = st.columns(3)

total_records = len(df)
total_anomalies = df['anomaly'].sum()
anomaly_percent = (total_anomalies / total_records) * 100

col1.metric("Total Records", total_records)
col2.metric("Anomalies Detected", int(total_anomalies))
col3.metric("Anomaly %", f"{anomaly_percent:.2f}%")

# -------------------------------
# Energy Selection
# -------------------------------

energy_option = st.selectbox(
    "Select Energy Type",
    ["electricity_cleaned", "hotwater_cleaned", "chilledwater"]
)

# -------------------------------
# Plot Energy with Anomalies
# -------------------------------

normal = df[df['anomaly'] == 0]
anomaly = df[df['anomaly'] == 1]

fig = px.line(
    normal,
    x="timestamp",
    y=energy_option,
    title=f"{energy_option} Consumption"
)

fig.add_scatter(
    x=anomaly['timestamp'],
    y=anomaly[energy_option],
    mode='markers',
    name='Anomaly',
    marker=dict(color='red', size=6)
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Data Table
# -------------------------------

if st.checkbox("Show Raw Data"):
    st.dataframe(df.tail(200))

st.markdown("---")
st.markdown("Developed by Sagar Karosiya 🚀")
