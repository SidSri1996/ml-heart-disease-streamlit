import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.metrics import confusion_matrix

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Heart Disease ML App", layout="wide")

# ---------- UI STYLE ----------
st.markdown("""
<style>
.main {background-color: #f5f7fb;}
h1 {color: #0e4c92; text-align: center;}
h2, h3 {color: #1b6ca8;}
.stButton>button {background-color:#1b6ca8;color:white;border-radius:8px;}
.stDownloadButton>button {background-color:#0e4c92;color:white;border-radius:8px;}
[data-testid="stDataFrame"] {border-radius:10px;border:1px solid #e1e5ee;}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<h1>❤️ Heart Disease Prediction Dashboard</h1>", unsafe_allow_html=True)
st.caption("Machine Learning Classification & Model Comparison Web Application")
st.divider()

# ---------- LOAD MODELS ----------
models = joblib.load("model/models.pkl")
columns = joblib.load("model/columns.pkl")
scaler = joblib.load("model/scaler.pkl")
metrics = pd.read_csv("model/metrics.csv", index_col=0)

# ---------- METRICS ----------
st.subheader("📊 Model Performance Comparison")
col1, col2 = st.columns([3,1])
with col1:
    st.dataframe(metrics, use_container_width=True)
with col2:
    st.info("Compare performance of six classification algorithms trained on the dataset.")

# ---------- CSV FORMAT ----------
st.subheader("📄 Expected CSV Format")
required_columns = [
    "age","sex","dataset","cp","trestbps","chol","fbs","restecg",
    "thalch","exang","oldpeak","slope","ca","thal"
]
st.code(", ".join(required_columns))

# ---------- SAMPLE CSV ----------
sample = pd.DataFrame([
    [63,1,"Cleveland",3,145,233,1,0,150,0,2.3,0,0,"fixed defect"],
    [37,1,"Hungary",2,130,250,0,1,187,0,3.5,0,0,"normal"],
    [56,0,"Cleveland",1,120,236,0,1,178,0,0.8,2,0,"normal"]
], columns=required_columns)

st.download_button("Download Sample CSV", sample.to_csv(index=False),
                   "sample_input.csv", "text/csv")

# ---------- FULL DATASET DOWNLOAD ----------
st.subheader("📥 Download Full Dataset for Testing")
DATA_URL = "https://raw.githubusercontent.com/SidSri1996/ml-heart-disease-streamlit/main/data/heart_disease_uci.csv"
response = requests.get(DATA_URL)

if response.status_code == 200:
    st.download_button("Download Full Heart Disease Dataset",
                       response.content, "heart_disease_uci.csv", "text/csv")
else:
    st.warning("Dataset download unavailable")

# ---------- MODEL SELECT ----------
st.subheader("⚙️ Select Model")
model_name = st.selectbox("Choose classification algorithm", list(models.keys()))
model = models[model_name]

st.info("Select a model → Upload dataset → Click 'Run Prediction'")

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

run_prediction = st.button("▶ Run Prediction")

# ---------- PREDICTION PIPELINE ----------
if uploaded_file is not None and run_prediction:

    original_data = pd.read_csv(uploaded_file)
    st.dataframe(original_data.head(), use_container_width=True)

    # Handle target column
    y_true = None
    if 'num' in original_data.columns:
        y_true = (original_data['num'] > 0).astype(int)
        original_data = original_data.drop(columns=['num'])

    # PREPROCESS
    data = pd.get_dummies(original_data, drop_first=True)

    for col in columns:
        if col not in data:
            data[col] = 0

    data = data[columns]
    data = data.fillna(data.median())

    if model_name in ["Logistic Regression", "KNN"]:
        data = scaler.transform(data)

    # PREDICT
    predictions = model.predict(data)
    pred_df = pd.DataFrame({"Prediction": predictions})
    pred_df["Prediction"] = pred_df["Prediction"].map({0:"No Disease",1:"Heart Disease"})

    # Download predictions CSV
    output_df = original_data.copy()
    output_df["Prediction"] = pred_df["
::contentReference[oaicite:0]{index=0}
