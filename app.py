import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import joblib # will see how it pans out, if not will use joblib for model loading and saving.
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, matthews_corrcoef, 
                             confusion_matrix, classification_report, ConfusionMatrixDisplay)

# create Streamlit app must include at least the following features : -
# a. Dataset upload option (CSV) [As streamlit free tier has limited capacity, upload only test data] [ 1 mark ]
# b. Model selection dropdown (if multiple models) [ 1 mark ] i.e. this will be static, I'll provide it, will create template for this first.
# c. Display of evaluation metrics [ 1 mark ]
# d. Confusion matrix or classification report [ 1 mark ]

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.impute import SimpleImputer
from pandas import DataFrame

st.set_page_config(page_title="ML Model Evaluation", layout="wide")
st.title("Machine Learning Model Evaluation Dashboard")

# a. Dataset upload
st.header("1. Upload Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

def validate_test_set(test_df: DataFrame) -> DataFrame:
    df_columns = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension']
    # Check for missing required columns
    missing_cols = set(df_columns) - set(test_df.columns)
    if missing_cols:
        st.error(f"Missing these columns: {df_columns}")
    else:
        test_df = test_df[df_columns]
        st.success('Column Validation Successful... now features are aligned')
        imputer: SimpleImputer = joblib.load("./model/imputer.pkl")
        scaler: StandardScaler = joblib.load("./model/scaler.pkl")
        test_imputed = imputer.transform(test_df)
        X_scaled = DataFrame(scaler.transform(test_imputed), columns=df_columns)
        return X_scaled


# evaluate the models using y_pred, y_true and y_proba
def evaluate_model(y_true, y_pred, y_proba):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba) 
    mcc = matthews_corrcoef(y_true, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    
    return accuracy, precision, recall, f1, roc_auc, mcc

# plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig: Figure =plt.figure(figsize=(6, 4))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    st.pyplot(fig, use_container_width=True)


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    y_test = df['target']

    X_test = df
    X_test = validate_test_set(X_test)
    
    # b. Model selection
    st.header("2. Select Model")

    models = {"Logistic Regression":"model/lr_model.pkl", 
              "Decision Tree Classifier":"model/dcf_model.pkl",
              "K-Nearest Neighbor Classifier":"model/knn_model.pkl",
              "Naive Bayes Classifier - Gaussian or Multinomial": "model/nb_model.pkl",
              "Ensemble Model - Random Forest": "model/rfc_model.pkl", 
              "Ensemble Model - XGBoost": "model/xgb_model.pkl"}
    selected_model = st.selectbox("Choose a model:", 
                                  options=list(key for key in models.keys()))
    

    # c. Evaluation metrics
    st.header("3. Evaluation Metrics")

    if st.button("Evaluate Model"):
        model_name = selected_model
        model_file = models[model_name]
        trained_model = joblib.load(model_file)
        y_pred = trained_model.predict(X_test)
        y_proba = trained_model.predict_proba(X_test)[:, 1] # probability of the positive class
        accuracy, precision, recall, f1, roc_auc, mcc = evaluate_model(y_true=y_test, y_pred=y_pred, y_proba=y_proba)
        
        col1, col2, col3, col4, col5, col6= st.columns(6)
        col1.metric("Accuracy", f"{accuracy:.2f}")
        col2.metric("Precision", f"{precision:.2f}")
        col3.metric("Recall", f"{recall:.2f}")
        col4.metric("F1-Score", f"{f1:.2f}")
        col5.metric("AUC", f"{roc_auc:.2f}")
        col6.metric("MCC", f"{mcc:.2f}")

        # d. Confusion matrix
        st.header("4. Confusion Matrix & Classification Report")
        
        plot_confusion_matrix(y_true=y_test, y_pred=y_pred)
        
        report_dict = classification_report(y_test, y_pred, output_dict=True)

        report_df = DataFrame(report_dict).transpose()
        # 3. Style it: highlight scores above 0.95 in green
        def highlight_high_scores(s):
            return ['background-color: #90EE90' if (isinstance(v, float) and v > 0.95) else '' for v in s]

        st.subheader("Interactive Classification Report")
        st.dataframe(report_df.style.apply(highlight_high_scores, axis=1).format(precision=2))

# streamlit run app.py