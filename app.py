import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Heart Disease ML App", layout="centered")

st.title("❤️ Heart Disease Prediction Dashboard")

# Load models and metrics
models = pickle.load(open("model/models.pkl","rb"))
metrics = pd.read_csv("model/metrics.csv", index_col=0)

st.subheader("📊 Model Performance Comparison")
st.dataframe(metrics)

# Model selection
model_name = st.selectbox("Select Model", list(models.keys()))
model = models[model_name]

st.subheader("📂 Upload Test Dataset (only features, no target column)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("Preview of uploaded data:")
    st.dataframe(data.head())

    predictions = model.predict(data)

    st.subheader("Predictions")
    st.write(predictions)

    # Confusion matrix (dummy demonstration since test has no labels)
    st.subheader("Prediction Distribution")
    st.bar_chart(pd.Series(predictions).value_counts())
