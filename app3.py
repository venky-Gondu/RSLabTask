import sklearn
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

warnings.filterwarnings('ignore')

def main():
    st.title("🔍 Supervised Classification with Machine Learning Algorithms")
    
    st.markdown("""
    This application allows you to upload a dataset, apply various machine learning algorithms for classification, 
    and visualize the results. You can also test the models on a separate test dataset.
    """)
    
    # File upload
    uploaded_file = st.file_uploader("📂 Upload your training Excel file", type=["xlsx"])
    test_file = st.file_uploader("📂 Upload your test Excel file", type=["xlsx"])
    
    if uploaded_file is not None and test_file is not None:
        # Read the Excel files
        df = pd.read_excel(uploaded_file)
        df_test = pd.read_excel(test_file)
        
        st.markdown("### 🔍 Data Preview")
        st.dataframe(df.head())
        
        x = df.drop('target', axis=1)
        y = df['target']
        
        # Scaling, normalization, and standardization
        scale = MinMaxScaler(feature_range=(-1, 1))
        x1 = scale.fit_transform(x)
        x2 = sklearn.preprocessing.normalize(x1)
        scaler = StandardScaler()
        x3 = scaler.fit_transform(x1)
        
        x1 = pd.DataFrame(x1)
        x2 = pd.DataFrame(x2)
        x3 = pd.DataFrame(x3)
        
        st.markdown("### 📊 Scaled Data Description")
        st.write("#### Min-Max Scaled Data")
        st.dataframe(x1.describe().T)
        
        st.write("#### Normalized Data")
        st.dataframe(x2.describe().T)
        
        st.write("#### Standardized Data")
        st.dataframe(x3.describe().T)
        
        # Defining machine learning algorithms for classification
        models = {
            'KNeighborsClassifier': KNeighborsClassifier(),
            'SVC': SVC(),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'RandomForestClassifier': RandomForestClassifier(),
            'XGBoost': xgb.XGBClassifier()
        }
        
        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(y)
        
        # Train-test split
        x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size=0.2, random_state=42)
        
        # Evaluate the model
        def evaluate_model(model, X_train, y_train, X_test, y_test):
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='micro')  # Set average to 'micro' for multiclass
            recall = recall_score(y_test, y_pred, average='micro')  # Set average to 'micro' for multiclass
            f1 = f1_score(y_test, y_pred, average='micro')  # Set average to 'micro' for multiclass
            return accuracy, precision, recall, f1
        
        model_results = {}
        for name, model in models.items():
            model.fit(x_train, y_train)
            accuracy, precision, recall, f1 = evaluate_model(model, x_train, y_train, x_test, y_test)
            model_results[name] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
        
        model_results = pd.DataFrame(model_results).T
        model_results.columns = ['accuracy', 'precision', 'recall', 'f1']
        st.markdown("### 📊 Model Evaluation Results")
        st.dataframe(model_results)
        
        
        # Classify the test data and evaluate them with right reason
        df_test_scaled = scale.fit_transform(df_test)
        df_test = pd.DataFrame(df_test_scaled)
        df_test.columns = df_test.columns.astype(str)  # Convert column names to strings
        
        for name, model in models.items():
            # Select the same number of features as used during training
            df_test_subset = df_test.iloc[:, :model.n_features_in_]
            y_pred = model.predict(df_test_subset)  # Predict using the subset
            df_test[name] = y_pred
        
        st.markdown("### 📝 Test Data Predictions")
        st.dataframe(df_test.head())
        
        # Show some classification graphs
        st.markdown("### 📊 Classification Graphs")
        model_name = st.selectbox("Select a model to visualize", list(models.keys()))
        
        if model_name:
            model = models[model_name]
            y_pred = model.predict(x_test)
    
            fig, ax = plt.subplots()
            sns.scatterplot(x=x_test.iloc[:, 0], y=x_test.iloc[:, 1], hue=y_pred, palette='viridis', ax=ax)
            ax.set_title(f'{model_name} Classification Results')
            st.pyplot(fig)

    
    else:
        st.info("Please upload both training and test Excel files to proceed.")

if __name__ == "__main__":
    main()


