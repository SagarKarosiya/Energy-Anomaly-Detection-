# ⚡ Energy Anomaly Detection System

An end-to-end Machine Learning project for detecting energy consumption anomalies in commercial buildings using Isolation Forest and deployed as an interactive Streamlit dashboard.

---

## 💻 Project Overview

Commercial buildings consume ~30% of global energy, generating billions in operational costs annually.

Unexpected energy spikes due to:
- Equipment failures
- Operational inefficiencies
- Occupancy mismatches
- System faults

This project builds a multivariate time-series anomaly detection system to automatically detect abnormal energy usage patterns.

---
Key Features
##🔹 Advanced Data Preprocessing

    Timestamp normalization

    Missing value handling (forward/backward fill)

    Outlier capping at 99th percentile

    Min-Max scaling to [0,1]

    Automatic energy column detection
    
## 🧠 ML Approach

- Multivariate Time-Series Data
- Feature Engineering (Rolling Statistics + Time Features)
- Isolation Forest (Unsupervised Anomaly Detection)
- Feature Scaling (StandardScaler)

---

## 📉 Features Engineered

- Hour of Day
- Day of Week
- Month
- 24-hour Rolling Mean
- 24-hour Rolling Standard Deviation
- Multi-energy correlation features

Total Engineered Features: 13+

---

## 🛠⚙️ Tech Stack 

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit

---

## 📁 Project Structure
MajorProject_Evoastra/
│
├── app.py
├── requirements.txt
├── README.md
│
└── src/
├── data_loader.py
├── preprocessing.py
├── feature_engineering.py
└── model.py


---

##  How It Works 🤔

1. Load energy datasets (Electricity, Hot Water, Chilled Water)
2. Clean & preprocess time-series data
3. Generate statistical and temporal features
4. Train Isolation Forest model
5. Detect anomalies
6. Visualize anomalies in interactive dashboard

---

## 📊 Dashboard Features

- Interactive energy type selection
- Anomaly visualization (red markers)
- KPI summary metrics
- Download anomaly data
- Real-time ML pipeline execution

---
## How to run ML Model In my PC 😁✌️💻
## ▶️ Run Locally 📡

### 1️⃣ Clone Repository
https://github.com/SagarKarosiya/Energy-Anomaly-Detection-.git

### 2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate

### 3️⃣ Install Requirements
pip install -r requirements.txt
Put your Data file (abc.csv) in data folder in the model.

### 4️⃣ Run Streamlit App
Paste the command in your terminal of VS Code :  <B> streamlit run app.py </B>


Open in browser:
http://localhost:8501


## 📱 Deployment

This project can be deployed on:

- Render.com
## Link : 🖥 (currently link not activate)
---

## 🏆 Key Highlights

✔ Industrial-scale time-series dataset  
✔ Multivariate anomaly detection  
✔ Modular ML pipeline architecture  
✔ Interactive web dashboard  
✔ Production-ready structure  

---

## 📈 Future Improvements

- Weather data integration
- SHAP explainability
- Model persistence
- Real-time anomaly detection
- Cloud deployment with CI/CD

---

## 👨‍💻 Author

## Sagar Karosiya  
AI & ML Engineer | Game Developer | Data Scientist  
## https://sagarkarosiya-portfolio.onrender.com/
---

## ⭐ If You Like This Project

Give it a star ⭐ on GitHub!






