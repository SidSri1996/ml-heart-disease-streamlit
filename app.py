import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease ML App", layout="centered")
st.title("❤️ Heart Disease Prediction Dashboard")

# Load saved objects
models = joblib.load("model/models.pkl")
columns = joblib.load("model/columns.pkl")
scaler = joblib.load("model/scaler.pkl")
metrics = pd.read_csv("model/metrics.csv", index_col=0)

st.subheader("📊 Model Performance Comparison")
st.dataframe(metrics)

st.subheader("📄 Expected CSV Format")

required_columns = [
    "age","sex","dataset","cp","trestbps","chol","fbs","restecg",
    "thalch","exang","oldpeak","slope","ca","thal"
]

st.write("Upload a CSV containing the following columns (no target column):")
st.code(", ".join(required_columns))

# Sample download
sample = pd.DataFrame(columns=required_columns)
st.download_button(
    label="Download Sample CSV",
    data=sample.to_csv(index=False),
    file_name="sample_input.csv",
    mime="text/csv"
)


# Model selection
model_name = st.selectbox("Select Model", list(models.keys()))
model = models[model_name]

st.subheader("📂 Upload Test Dataset")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    # same preprocessing as training
    data = pd.get_dummies(data, drop_first=True)

    for col in columns:
        if col not in data:
            data[col] = 0

    data = data[columns]
    data = data.fillna(data.median())

    if model_name in ["Logistic Regression", "KNN"]:
        data = scaler.transform(data)

    predictions = model.predict(data)

    st.subheader("Predictions")
    st.write(predictions)

    st.subheader("Prediction Distribution")
    st.bar_chart(pd.Series(predictions).value_counts())


