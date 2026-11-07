import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Classification Model Evaluator", layout="wide")

st.title("Machine Learning Classification Evaluation App")
st.write("Upload the test dataset to evaluate pre-trained models without retraining.")

# Dataset upload option
uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("1. Test Dataset Preview")
    st.dataframe(data.head(10))
    
    if 'target' not in data.columns:
        st.error("Uploaded CSV must contain the 'target' column.")
    else:
        X_test = data.drop(columns=['target'])
        y_test = data['target']
        
        # Model selection dropdown
        st.subheader("2. Model Selection")
        model_name = st.selectbox(
            "Select Classifier:", 
            ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]
        )
        
        model_paths = {
            "Logistic Regression": "model/lr_model.joblib",
            "Decision Tree": "model/dt_model.joblib",
            "KNN": "model/knn_model.joblib",
            "Naive Bayes": "model/nb_model.joblib",
            "Support Vector Machine": "model/svm_model.joblib"
        }
        
        try:
            model = joblib.load(model_paths[model_name])
            st.success(f"{model_name} loaded successfully!")
            
            # Predict
            y_pred = model.predict(X_test)
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                auc_val = roc_auc_score(y_test, y_proba)
            except Exception:
                auc_val = None

            # Display evaluation metrics
            st.subheader("3. Evaluation Metrics")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            c2.metric("AUC", f"{auc_val:.4f}" if auc_val is not None else "N/A")
            c3.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
            c4.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
            c5.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
            c6.metric("MCC Score", f"{matthews_corrcoef(y_test, y_pred):.4f}")
            
            # Confusion matrix & classification report
            st.subheader("4. Diagnostics & Reporting")
            col_diag1, col_diag2 = st.columns(2)
            
            with col_diag1:
                st.write("**Confusion Matrix**")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
                plt.ylabel('Actual Label')
                plt.xlabel('Predicted Label')
                st.pyplot(fig)
                
            with col_diag2:
                st.write("**Classification Report**")
                report_dict = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report_dict).transpose())
                
        except FileNotFoundError:
            st.error(f"Could not find model file at {model_paths[model_name]}. Ensure the 'model/' folder exists in the repository.")
else:
    st.info("Please upload `test_data.csv` to proceed.")
