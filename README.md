Heart Disease Prediction using Machine Learning
1. Problem Statement

The goal of this project is to predict whether a patient has heart disease based on medical attributes using multiple machine learning classification algorithms.
The project demonstrates an end-to-end ML workflow including data preprocessing, model training, evaluation, and deployment using Streamlit.

2. Dataset Description

Dataset: UCI Heart Disease Dataset

Instances: 920 records

Features: 15 medical attributes (age, sex, chest pain type, cholesterol, etc.)

Target Variable: num

  0 → No heart disease

  1 → Heart disease

Preprocessing performed:

Converted multi-class target to binary classification

One-hot encoding of categorical variables

Missing value imputation using median

Feature scaling for distance-based models

| Model               | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
| ------------------- | -------- | ----- | --------- | ------ | ----- | ----- |
| Logistic Regression | 0.864    | 0.934 | 0.853     | 0.912  | 0.882 | 0.725 |
| Decision Tree       | 0.864    | 0.882 | 0.860     | 0.902  | 0.880 | 0.725 |
| KNN                 | 0.859    | 0.912 | 0.852     | 0.902  | 0.876 | 0.714 |
| Naive Bayes         | 0.848    | 0.910 | 0.856     | 0.873  | 0.864 | 0.691 |
| Random Forest       | 0.880    | 0.953 | 0.864     | 0.931  | 0.896 | 0.759 |
| XGBoost             | 0.880    | 0.932 | 0.877     | 0.912  | 0.894 | 0.758 |

4. Observations
   
| Model               | Observation                                                       |
| ------------------- | ----------------------------------------------------------------- |
| Logistic Regression | Performs well due to near linear separability of medical features |
| Decision Tree       | Slight overfitting but interpretable                              |
| KNN                 | Sensitive to feature scaling and noise                            |
| Naive Bayes         | Fast but assumes feature independence                             |
| Random Forest       | Best overall performance due to ensemble averaging                |
| XGBoost             | High accuracy with strong generalization capability               |

5. Streamlit App Features

  Upload custom test dataset (CSV)

  Select model from dropdown

  View model performance comparison

  Generate predictions interactively

  6. Live Application

Streamlit App:
https://ml-heart-disease-app-cwz8usnvhxvcbcwdw9l5ml.streamlit.app/

GitHub Repository:
https://github.com/SidSri1996/ml-heart-disease-streamlit
