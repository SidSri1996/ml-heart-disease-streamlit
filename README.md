Heart Disease Prediction using Machine Learning
1. Problem Statement
   
The objective of this project is to develop a machine learning system capable of predicting the presence of heart disease in a patient using clinical and physiological attributes.
Multiple classification algorithms are implemented and compared to evaluate their effectiveness.
The project also demonstrates a complete machine learning pipeline, including preprocessing, model evaluation, and deployment using a Streamlit web application.

2. Dataset Description

     Dataset: UCI Heart Disease Dataset
     Instances: 920 records
     Features: 15 medical attributes (age, sex, chest pain type, cholesterol, etc.)
     Target Variable: num

Target interpretation:
  0 → No heart disease
  1 → Heart disease

Preprocessing performed:
    Converted multi-class target to binary classification
    Applied one-hot encoding to categorical attributes
    Handled missing values using median imputation
    Performed feature scaling for distance-based algorithms


3. Model Performance Comparison
   
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
     Select a classification model from dropdown
     View comparison of model performance metrics
     Generate predictions interactively
     Display confusion matrix when actual labels are provided

6. Live Application

Streamlit App:
https://ml-heart-disease-app-cwz8usnvhxvcbcwdw9l5ml.streamlit.app/

GitHub Repository:
https://github.com/SidSri1996/ml-heart-disease-streamlit
