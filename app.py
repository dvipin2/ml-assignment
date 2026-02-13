import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import pickle # will see how it pans out, if not will use joblib for model loading and saving.

# create Streamlit app must include at least the following features : -
# a. Dataset upload option (CSV) [As streamlit free tier has limited capacity, upload only test data] [ 1 mark ]
# b. Model selection dropdown (if multiple models) [ 1 mark ] i.e. this will be static, I'll provide it, will create template for this first.
# c. Display of evaluation metrics [ 1 mark ]
# d. Confusion matrix or classifi cation report [ 1 mark ]

# below is just test code, I built in a hurry, will evolve it on weekend, will add more features.
# will add model loading and evaluation on the uploaded test data, 
# for now it's just a template to show how the app will look like and work.

import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Model Evaluation", layout="wide")
st.title("Machine Learning Model Evaluation Dashboard")

# a. Dataset upload
st.header("1. Upload Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(f"Dataset shape: {df.shape}")
    st.dataframe(df.head())
    
    # b. Model selection
    st.header("2. Select Model")
    models = ["Logistic Regression", "Random Forest", "SVM"]
    selected_model = st.selectbox("Choose a model:", models)
    
    # c. Evaluation metrics
    st.header("3. Evaluation Metrics")
    if st.button("Evaluate Model"):
        # Placeholder for actual model evaluation
        accuracy = 0.85
        precision = 0.83
        recall = 0.87
        f1 = 0.85
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy:.2f}")
        col2.metric("Precision", f"{precision:.2f}")
        col3.metric("Recall", f"{recall:.2f}")
        col4.metric("F1-Score", f"{f1:.2f}")
        
        # d. Confusion matrix
        st.header("4. Confusion Matrix & Classification Report")
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)
        
        st.text(classification_report(y_true, y_pred))

# streamlit run app.py