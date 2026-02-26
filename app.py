import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.preprocessing import preprocess_data
from src.feature_engineering import feature_engineering
from src.model import train_model
from src.evaluation import evaluate_model
from src.business_insight import generate_business_insights
from src.pdf_report import generate_pdf_report

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Energy AI Dashboard",
    layout="wide",
    page_icon="⚡"
)

st.title("⚡ Energy Anomaly Detection System")
st.markdown("AI-Powered Commercial Energy Intelligence Platform")
st.markdown("Group-D")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("⚙ Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload Energy CSV File",
    type=["csv"]
)

cost_per_unit = st.sidebar.number_input(
    "Cost per Energy Unit ($)",
    value=0.12
)

run_button = st.sidebar.button(" Run Analysis")
# Model Picker
model_choice = st.sidebar.selectbox(
    "Select Anomaly Detection Model",
    [
        "Isolation Forest",
        "Local Outlier Factor (LOF)",
        "One-Class SVM",
        "Robust Covariance"
    ]
)
# =====================================================
# MAIN EXECUTION
# =====================================================
if uploaded_file is not None and run_button:

    with st.spinner("Running full ML pipeline..."):

        # -----------------------------
        # LOAD DATA
        # -----------------------------
        df = pd.read_csv(uploaded_file)

        # Auto-detect timestamp column
        timestamp_found = False
        for col in df.columns:
            if "time" in col.lower():
                df = df.rename(columns={col: "timestamp"})
                timestamp_found = True
                break

        if not timestamp_found:
            st.error("No timestamp column found.")
            st.stop()

        # -----------------------------
        # PREPROCESSING
        # -----------------------------
        df = preprocess_data(df)

        st.header("🧹 Preprocessing Results")

        col1, col2 = st.columns(2)

        col1.metric("Missing Values", df.isna().sum().sum())
        col1.metric("Timestamp Sorted",
                    df["timestamp"].is_monotonic_increasing)

        numeric_cols = df.select_dtypes(include=np.number).columns
        col2.metric("Min Scaled Value",
                    round(df[numeric_cols].min().min(), 4))
        col2.metric("Max Scaled Value",
                    round(df[numeric_cols].max().max(), 4))

        st.divider()

        # -----------------------------
        # FEATURE ENGINEERING
        # -----------------------------
        df = feature_engineering(df)

        st.header("⚙ Feature Engineering Results")
        st.success(f"Total Features After Engineering: {df.shape[1]}")
        st.success("✔ Rolling deviation metrics added")
        st.success("✔ Temporal seasonality features added")
        st.success("✔ Lag & momentum features added")

        st.divider()

        # -----------------------------
        # MODEL TRAINING
        # -----------------------------
        df,switched, model = train_model(df, model_choice)
        if switched:
          st.warning("Large dataset detected. Switched to Isolation Forest for performance.")
        st.header(" Model Results")

        anomaly_rate = round(df["final_anomaly"].mean() * 100, 2)

        st.metric("Model Used", model_choice)
        st.metric("Anomaly Rate (%)", anomaly_rate)
        st.success("✔ Anomaly labels generated")
        st.success("✔ Anomaly scores computed")

        st.divider()

        # -----------------------------
        # DATA OVERVIEW
        # -----------------------------
        st.header(" Data Overview")

        col_df = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": df.dtypes.values
        })

        st.dataframe(col_df, width="stretch")

        st.divider()

        # -----------------------------
        # EVALUATION
        # -----------------------------
        evaluation_stats = evaluate_model(df)

        st.header(" Evaluation Results")

        col1, col2 = st.columns(2)

        col1.metric("Total Samples",
                    evaluation_stats["Total Samples"])
        col1.metric("Total Anomalies",
                    evaluation_stats["Total Anomalies"])

        col2.metric("Anomaly Rate (%)",
                    evaluation_stats["Anomaly Rate (%)"])

        st.success("✔ Anomaly distribution statistics calculated")
        st.success("✔ Top anomalous samples identified")

        st.divider()

        # -----------------------------
        # TOP ANOMALIES
        # -----------------------------
        st.subheader(" Top 10 Most Anomalous Samples")

        if "anomaly_score" in df.columns:
            top_anomalies = df.sort_values("anomaly_score").head(10)
        else:
            top_anomalies = df[df["final_anomaly"] == 1].head(10)

        st.dataframe(top_anomalies, width="stretch")

        st.divider()

        # -----------------------------
        # BUSINESS INSIGHTS
        # -----------------------------
        insights, df = generate_business_insights(
            df,
            cost_per_unit=cost_per_unit
        )

        st.header(" Business Insight & Impact")

        col1, col2 = st.columns(2)

        col1.metric("Estimated Cost Impact ($)",
                    insights["estimated_cost_loss_$"])

        col2.metric("Anomaly Rate (%)",
                    insights["anomaly_rate_percent"])

        # Seasonal
        st.subheader(" Seasonal Pattern Analysis")
        monthly = df[df["final_anomaly"] == 1]["month"].value_counts().sort_index()
        st.bar_chart(monthly)

        # Peak hours
        st.subheader(" Peak Anomaly Hours")
        peak_hours = df[df["final_anomaly"] == 1]["hour"].value_counts().sort_index()
        st.bar_chart(peak_hours)

        # Recommendations
        st.subheader(" Business Recommendations")
        for rec in insights["recommendations"]:
            st.success(rec)

        st.divider()

# -----------------------------
# PDF DOWNLOAD
# -----------------------------
        st.subheader("📄 Download Executive Report")

        pdf_file = generate_pdf_report(df, insights, evaluation_stats)

        with open(pdf_file, "rb") as f:
         st.download_button(
        label=" Download PDF Report",
        data=f,
        file_name="Energy_AI_Report.pdf",
        mime="application/pdf"
    )
else:
    st.info("Upload CSV file and click 'Run Analysis' to begin.")
#  st.warning(SagarKarosiya)
# =====================================================
# FOOTER
# =====================================================
st.markdown(
    """
    <hr style="margin-top:50px;">
    <div style="text-align:center;font-size:14px;color:gray;">
        © 2026 Sagar Karosiya | Energy AI Dashboard <br>
        All Rights Reserved
    </div>
    """,
    unsafe_allow_html=True
)