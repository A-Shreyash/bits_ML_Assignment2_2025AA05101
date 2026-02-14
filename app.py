import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Page Configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    layout="wide",
    page_icon="🏦",
    initial_sidebar_state="collapsed",
    menu_items=None  # Hide the deploy and settings menu
)

# Custom CSS for Theme
st.markdown("""
    <style>
    body, .main, .block-container, .stApp {
        background-color: #f5f5dc !important;
        color: #8b4513 !important;
    }
    .main-header {
        font-size: 2.5em;
        color: #daa520 !important;
        text-align: center;
        margin-bottom: 20px;
        background-color: #8b4513 !important;  /* Darker background for header */
        padding: 10px;
        border-radius: 10px;
    }
    .metric-card {
        background-color: #fff8dc !important;
        color: #8b4513 !important;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        margin: 2px;
        border: 1px solid #daa520;
        font-size: 0.9em;
    }
    .metric-card h3 {
        color: #8b4513 !important;
        margin: 0;
        text-align: center !important;
    }
    .metric-card p {
        color: #8b4513 !important;
        margin: 0;
        font-size: 1.5em !important;  /* Larger numbers */
        font-weight: bold;
        text-align: center !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f5f5dc !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #fff8dc !important;
        color: #8b4513 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #daa520 !important;
        color: white !important;
    }
    .stDataFrame, .stDataFrame table {
        background-color: #fff8dc !important;
        color: #8b4513 !important;
    }
    .stButton button {
        background-color: #daa520 !important;
        color: white !important;
        border-radius: 5px;
    }
    .stSelectbox, .stNumberInput, .stTextInput {
        background-color: #fff8dc !important;
        color: #8b4513 !important;
    }
    .stMarkdown, .stText, .stSubheader {
        color: #8b4513 !important;
    }
    [data-testid="stAppViewContainer"], [data-testid="stVerticalBlock"] {
        background-color: #f5f5dc !important;
    }
    .footer {
        text-align: center;
        color: #8b4513 !important;
        font-size: 0.8em;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Main Header
st.markdown('<h1 class="main-header">🏦 Loan Approval Prediction – ML Models Comparison</h1>', unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    paths = {
        "Logistic Regression": "model/logistic_regression.pkl",
        "Decision Tree": "model/decision_tree.pkl",
        "KNN": "model/knn.pkl",
        "Naive Bayes": "model/naive_bayes.pkl",
        "Random Forest": "model/random_forest.pkl",
        "XGBoost": "model/xgboost.pkl",
    }
    models = {}
    for name, path in paths.items():
        if not os.path.exists(path):
            st.error(f"Missing model file: {path}")
            st.stop()
        models[name] = joblib.load(path)
    return models

models = load_models()

# Load Scaler
if not os.path.exists("model/scaler.pkl"):
    st.error("Missing scaler.pkl")
    st.stop()
scaler = joblib.load("model/scaler.pkl")

# Load Dataset
@st.cache_data
def load_dataset():
    return pd.read_csv("Loan_approval_data_2025.csv")

df = load_dataset()
TARGET_COL = "loan_status"

# Preprocessing
df.drop(columns=["customer_id"], inplace=True, errors="ignore")
categorical_cols = ["occupation_status", "product_type", "loan_intent"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Create processed test data for download
test_df = pd.concat([X_test, y_test.rename(TARGET_COL)], axis=1)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Model Evaluation", "🔍 Model Comparison", "📈 Data Exploration"])

# Tab 1: Model Evaluation
with tab1:
    st.subheader("Select a Model and View Its Performance")
    selected_model_name = st.selectbox("Choose a Model", list(models.keys()), key="eval_model")
    model = models[selected_model_name]
    
    if selected_model_name in ["Logistic Regression", "KNN"]:
        X_test_final = scaler.transform(X_test)
    else:
        X_test_final = X_test
    
    y_pred = model.predict(X_test_final)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    st.markdown("### Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><h3>Accuracy</h3><p>{acc:.4f}</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>Precision</h3><p>{precision:.4f}</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h3>Recall</h3><p>{recall:.4f}</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><h3>F1 Score</h3><p>{f1:.4f}</p></div>', unsafe_allow_html=True)
    
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(3, 2.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {selected_model_name}")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig)

# Tab 2: Model Comparison
with tab2:
    st.subheader("Compare All Models Side-by-Side")
    results = []
    for name, mdl in models.items():
        if name in ["Logistic Regression", "KNN"]:
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        y_pred = mdl.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })
    results_df = pd.DataFrame(results)
    st.dataframe(results_df.style.highlight_max(axis=0, color='#daa520'))
    
    metric_options = ["Accuracy", "Precision", "Recall", "F1 Score"]
    selected_metric = st.selectbox("Select Metric for Comparison", metric_options, key="metric_select")
    
    st.markdown(f"### {selected_metric} Comparison")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(data=results_df, x="Model", y=selected_metric, ax=ax, palette="Oranges")
    ax.set_title(f"Model {selected_metric} Comparison")
    ax.set_ylabel(selected_metric)
    plt.xticks(rotation=45)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig)

# Tab 3: Data Exploration
with tab3:
    st.subheader("Explore the Dataset")
    
    st.markdown("### Dataset Overview")
    st.write(f"Shape: {df.shape}")
    st.write("First 5 rows:")
    st.dataframe(df.head())
    
    st.markdown("### Summary Statistics")
    st.dataframe(df.describe())
    
    st.markdown("### Target Distribution")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.countplot(data=df, x=TARGET_COL, ax=ax, palette="Oranges")
    ax.set_title("Loan Status Distribution")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig)
    
    st.markdown("### Correlation Heatmap (Numerical Features)")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COL in num_cols:
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=False, cmap="Oranges", ax=ax)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.pyplot(fig)
    else:
        st.write("No numerical columns found for correlation.")
    
    st.markdown("### Download Processed Test Data")
    st.download_button(
        label="Download processed_test_data.csv",
        data=test_df.to_csv(index=False),
        file_name="processed_test_data.csv",
        mime="text/csv",
        key="download_test_data"
    )

# Footer
st.markdown('<div class="footer">© 2026 Created by Shreyash Sompurkar - 2025AA05101</div>', unsafe_allow_html=True)