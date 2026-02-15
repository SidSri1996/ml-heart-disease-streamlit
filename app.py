import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Heart Disease ML App", layout="centered")

st.title("❤️ Heart Disease Prediction Dashboard")

# Load trained objects
models = joblib.load("model/models.pkl")
columns = joblib.load("model/columns.pkl")
scaler = joblib.load("model/scaler.pkl")
metrics = pd.read_csv("model/metrics.csv", index_col=0)

# Show metrics
st.subheader("📊 Model Performance Comparison")
st.dataframe(metrics)

# Show expected CSV format
st.subheader("📄 Expected CSV Format")

required_columns = [
    "age","sex","dataset","cp","trestbps","chol","fbs","restecg",
    "thalch","exang","oldpeak","slope","ca","thal"
]

st.write("Upload a CSV containing the following columns (target column 'num' optional):")
st.code(", ".join(required_columns))

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

# Upload file
st.subheader("📂 Upload Dataset")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    original_data = pd.read_csv(uploaded_file)

    st.write("Preview of uploaded data:")
    st.dataframe(original_data.head())

    # Extract target if exists
    y_true = None
    if 'num' in original_data.columns:
        y_true = original_data['num']
        original_data = original_data.drop(columns=['num'])

    # Preprocessing (same as training)
    data = pd.get_dummies(original_data, drop_first=True)

    for col in columns:
        if col not in data:
            data[col] = 0

    data = data[columns]
    data = data.fillna(data.median())

    if model_name in ["Logistic Regression", "KNN"]:
        data = scaler.transform(data)

    # Predictions
    predictions = model.predict(data)

    st.subheader("Predictions")
    st.write(predictions)

    st.subheader("Prediction Distribution")
    st.bar_chart(pd.Series(predictions).value_counts())

    # Confusion matrix if actual labels available
    if y_true is not None:
        cm = confusion_matrix(y_true, predictions)

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    else:
        st.info("Upload dataset including 'num' column to view confusion matrix")
