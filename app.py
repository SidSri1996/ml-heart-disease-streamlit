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

st.download_button(
    label="Download Sample CSV",
    data=sample.to_csv(index=False),
    file_name="sample_input.csv",
    mime="text/csv"
)

# ---------- FULL DATASET DOWNLOAD ----------
st.subheader("📥 Download Full Dataset for Testing")
DATA_URL = "https://raw.githubusercontent.com/SidSri1996/ml-heart-disease-streamlit/main/data/heart_disease_uci.csv"
response = requests.get(DATA_URL)

if response.status_code == 200:
    st.download_button(
        label="Download Full Heart Disease Dataset",
        data=response.content,
        file_name="heart_disease_uci.csv",
        mime="text/csv"
    )
else:
    st.warning("Dataset download unavailable")

# ---------- MODEL SELECT ----------
st.subheader("⚙️ Select Model")
model_name = st.selectbox("Choose classification algorithm", list(models.keys()))
model = models[model_name]

# ---------- FILE UPLOAD ----------
st.subheader("📂 Upload Dataset")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    original_data = pd.read_csv(uploaded_file)
    st.dataframe(original_data.head(), use_container_width=True)

    # Handle target
    y_true = None
    if 'num' in original_data.columns:
        y_true = (original_data['num'] > 0).astype(int)
        original_data = original_data.drop(columns=['num'])

    # ---------- PREPROCESS ----------
    data = pd.get_dummies(original_data, drop_first=True)

    for col in columns:
        if col not in data:
            data[col] = 0

    data = data[columns]
    data = data.fillna(data.median())

    if model_name in ["Logistic Regression", "KNN"]:
        data = scaler.transform(data)

    # ---------- PREDICT ----------
    predictions = model.predict(data)
    pred_df = pd.DataFrame({"Prediction": predictions})
    pred_df["Prediction"] = pred_df["Prediction"].map({0:"No Disease",1:"Heart Disease"})

    # ================= DASHBOARD =================
    st.divider()
    st.subheader("📊 Prediction Dashboard")

    colA, colB = st.columns([1,1])

    # LEFT SIDE
    with colA:
        st.markdown("## 📋 Prediction Results")
        st.dataframe(pred_df, height=380, use_container_width=True)

        counts = pred_df["Prediction"].value_counts()

        fig1, ax1 = plt.subplots(figsize=(4,3))
        bars = ax1.bar(counts.index, counts.values)

        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x()+bar.get_width()/2, height+1,
                     f'{int(height)}', ha='center', fontsize=10)

        ax1.set_ylabel("Patients")
        ax1.set_title("Prediction Distribution")
        st.pyplot(fig1)

    # RIGHT SIDE
    with colB:
        st.markdown("## 🧠 Model Evaluation")
        if y_true is not None:
            cm = confusion_matrix(y_true, predictions)
            fig2, ax2 = plt.subplots(figsize=(4,3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax2)
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("Actual")
            st.pyplot(fig2)
        else:
            st.info("Upload dataset including 'num' column to view confusion matrix")

# ---------- FOOTER ----------
st.divider()
st.caption("Developed for BITS Pilani WILP - Machine Learning Assignment 2")
