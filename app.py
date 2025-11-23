"""
Low Birth Weight Risk Prediction - Interactive Web Application

Streamlit web app for visualizing model results and making predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Low Birth Weight Risk Prediction",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-high {
        color: #d62728;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .prediction-low {
        color: #2ca02c;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data - Optimized for production
@st.cache_data(ttl=3600, show_spinner=False)
def load_models():
    """Load all trained models with error handling"""
    models = {}
    model_files = {
        'Logistic Regression': 'outputs/models/Logistic_Regression.pkl',
        'Logistic Regression (SMOTE)': 'outputs/models/Logistic_Regression_SMOTE.pkl',
        'Random Forest': 'outputs/models/Random_Forest.pkl',
        'XGBoost': 'outputs/models/XGBoost.pkl',
        'XGBoost (Calibrated)': 'outputs/models/XGBoost_Calibrated.pkl'
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                # Try memory mapping first, fallback to normal load
                try:
                    models[name] = joblib.load(path, mmap_mode='r')
                except:
                    models[name] = joblib.load(path)
            except Exception:
                # Silently skip failed models in production
                continue
    
    return models

@st.cache_data(ttl=3600, show_spinner=False)
def load_results():
    """Load model evaluation results"""
    try:
        if os.path.exists('outputs/results_summary.csv'):
            return pd.read_csv('outputs/results_summary.csv', index_col=0)
    except Exception:
        pass
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_image(path):
    """Load a single image on demand"""
    try:
        if os.path.exists(path):
            return Image.open(path)
    except Exception:
        pass
    return None

def get_image_paths():
    """Return image file paths without loading"""
    return {
        'EDA Overview': 'outputs/01_eda_overview.png',
        'Feature Distributions': 'outputs/02_eda_feature_distributions.png',
        'ROC Curves': 'outputs/03_roc_curves.png',
        'PR Curves': 'outputs/04_pr_curves.png',
        'Calibration Plots': 'outputs/05_calibration_plots.png',
        'Confusion Matrices': 'outputs/06_confusion_matrices.png',
        'SHAP Summary': 'outputs/07_shap_summary.png',
        'SHAP Importance': 'outputs/08_shap_importance.png',
        'Risk Heatmap': 'outputs/10_risk_heatmap.png',
        'Model Comparison': 'outputs/11_model_comparison.png'
    }

# Initialize session state - Lazy loading for production
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.models = {}
    st.session_state.results = None

# Lazy load data only when needed (not at startup)
def ensure_models_loaded():
    """Load models only when needed"""
    if not st.session_state.models_loaded:
        try:
            st.session_state.models = load_models()
            st.session_state.results = load_results()
            st.session_state.models_loaded = True
        except Exception as e:
            st.error(f"Error loading models: {e}")
            st.session_state.models_loaded = True  # Prevent retry loop

# Main App - Wrap in try-catch for production stability
try:
    # Lazy load models only when needed
    ensure_models_loaded()
    
    st.markdown('<div class="main-header">Low Birth Weight Risk Prediction System</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Home", "Model Dashboard", "Make Prediction", "Visualizations", "About"]
    )

    # Home Page
    if page == "Home":
        st.header("Welcome to the Low Birth Weight Risk Prediction System")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Models Trained", len(st.session_state.models) if st.session_state.models else 0)
        
        with col2:
            if st.session_state.results is not None:
                try:
                    best_model = st.session_state.results['ROC-AUC'].idxmax()
                    best_score = st.session_state.results.loc[best_model, 'ROC-AUC']
                    st.metric("Best Model (ROC-AUC)", f"{best_score:.4f}", best_model)
                except:
                    st.metric("Best Model", "N/A")
            else:
                st.metric("Best Model", "N/A")
        
        with col3:
            image_paths = get_image_paths()
            available_images = sum(1 for p in image_paths.values() if os.path.exists(p))
            st.metric("Visualizations", available_images)
        
        st.markdown("---")
        
        st.subheader("Overview")
        st.write("""
        This application provides an interactive interface for predicting low birth weight risk 
        using machine learning models. Low birth weight (LBW) is defined as an infant weight 
        below 2.5 kg and is a major indicator of neonatal health.
        
        **Features:**
        - **Interactive Predictions**: Input patient data and get instant risk predictions
        - **Model Comparison**: Compare performance of 5 different ML models
        - **Comprehensive Visualizations**: Explore EDA, model performance, and SHAP explanations
        - **Explainable AI**: Understand which factors contribute to risk predictions
        
        **Use the navigation menu to explore different sections.**
        """)
        
        if st.session_state.results is not None:
            st.subheader("Model Performance Summary")
            st.dataframe(st.session_state.results.style.highlight_max(axis=0, subset=['ROC-AUC', 'PR-AUC']))

    elif page == "Model Dashboard":
        st.header("Model Performance Dashboard")
        
        if st.session_state.results is None:
            st.error("Results not available. Please run main.py first to generate models.")
        else:
            # Metrics comparison
            st.subheader("Performance Metrics Comparison")
            
            metrics = ['ROC-AUC', 'PR-AUC', 'F1 Score', 'Recall']
            selected_metric = st.selectbox("Select Metric", metrics)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=st.session_state.results.index,
                y=st.session_state.results[selected_metric].astype(float),
                marker_color='steelblue',
                text=st.session_state.results[selected_metric].astype(float).round(4),
                textposition='outside'
            ))
            fig.update_layout(
                title=f"{selected_metric} by Model",
                xaxis_title="Model",
                yaxis_title=selected_metric,
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig, width='stretch')
            
            # Detailed metrics table
            st.subheader("Detailed Metrics")
            st.dataframe(st.session_state.results)
            
            # Best model highlight
            try:
                best_model = st.session_state.results['ROC-AUC'].astype(float).idxmax()
                best_score = st.session_state.results.loc[best_model, 'ROC-AUC']
                st.success(f"**Best Performing Model**: {best_model} with ROC-AUC of {best_score}")
            except:
                st.info("Unable to determine best model.")

    # Make Prediction
    elif page == "Make Prediction":
        st.header("Make a Prediction")
        
        if len(st.session_state.models) == 0:
            st.error("No models available. Please run main.py first to train models.")
        else:
            st.write("Enter patient information to predict low birth weight risk:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Maternal Age (years)", min_value=14, max_value=50, value=25, step=1)
                lwt = st.number_input("Mother's Weight at Last Menstrual Period (lbs)", min_value=80, max_value=250, value=130, step=1)
                race = st.selectbox("Race", options=[1, 2, 3], format_func=lambda x: {1: "White", 2: "Black", 3: "Other"}[x])
                smoke = st.selectbox("Smoking Status", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            
            with col2:
                ptl = st.number_input("Number of Previous Premature Labors", min_value=0, max_value=3, value=0, step=1)
                ht = st.selectbox("History of Hypertension", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                ui = st.selectbox("Uterine Irritability", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                ftv = st.number_input("Number of Physician Visits (First Trimester)", min_value=0, max_value=6, value=1, step=1)
            
            # Model selection
            selected_model = st.selectbox("Select Model for Prediction", list(st.session_state.models.keys()))
            
            if st.button("Predict Risk", type="primary"):
                # Prepare input data
                input_data = pd.DataFrame({
                    'age': [age],
                    'lwt': [lwt],
                    'race': [race],
                    'smoke': [smoke],
                    'ptl': [ptl],
                    'ht': [ht],
                    'ui': [ui],
                    'ftv': [ftv]
                })
                
                # Make prediction
                model = st.session_state.models[selected_model]
                
                try:
                    # Get prediction probability
                    prob = model.predict_proba(input_data)[0, 1]
                    prediction = model.predict(input_data)[0]
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        risk_percent = prob * 100
                        if risk_percent >= 50:
                            st.markdown(f'<div class="prediction-high">Risk: {risk_percent:.1f}%</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="prediction-low">Risk: {risk_percent:.1f}%</div>', unsafe_allow_html=True)
                    
                    with col2:
                        if prediction == 1:
                            st.markdown('<div class="prediction-high">Prediction: Low Birth Weight Risk</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="prediction-low">Prediction: Normal Birth Weight</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.metric("Model Used", selected_model)
                    
                    # Risk interpretation
                    st.markdown("---")
                    if risk_percent >= 70:
                        st.warning("**High Risk**: This patient has a high risk of low birth weight. Consider additional monitoring and interventions.")
                    elif risk_percent >= 40:
                        st.info("**Moderate Risk**: This patient has a moderate risk. Regular monitoring is recommended.")
                    else:
                        st.success("**Low Risk**: This patient has a low risk of low birth weight.")
                    
                    # Visualize probability
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = risk_percent,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Risk Probability (%)"},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 40], 'color': "lightgreen"},
                                {'range': [40, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, width='stretch')
                    
                    # Feature importance (if available)
                    st.markdown("---")
                    st.subheader("Key Risk Factors")
                    st.write("""
                    Based on clinical research, the following factors are known to influence low birth weight risk:
                    - **Smoking**: Significantly increases risk
                    - **Hypertension**: Major risk factor
                    - **Maternal Age**: Very young or older mothers have higher risk
                    - **Maternal Weight**: Low pre-pregnancy weight increases risk
                    - **Uterine Irritability**: May indicate complications
                    - **Previous Premature Labors**: History increases risk
                    """)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
                    st.write("Please check that all input fields are filled correctly.")

    # Visualizations
    elif page == "Visualizations":
        st.header("Model Visualizations")
        
        image_paths = get_image_paths()
        available_viz = {name: path for name, path in image_paths.items() if os.path.exists(path)}
        
        if len(available_viz) == 0:
            st.error("No visualizations available. Please run main.py first to generate plots.")
        else:
            # Visualization selector
            viz_options = list(available_viz.keys())
            selected_viz = st.selectbox("Select Visualization", viz_options)
            
            if selected_viz:
                st.subheader(selected_viz)
                img = load_image(available_viz[selected_viz])
                if img:
                    st.image(img, width='stretch')
                else:
                    st.error(f"Could not load {selected_viz}")
                
                # Add descriptions
                descriptions = {
                    'EDA Overview': "Exploratory data analysis showing target distribution, birth weight distributions, and feature correlations.",
                    'Feature Distributions': "Distribution of numeric features by birth weight class (normal vs low birth weight).",
                    'ROC Curves': "Receiver Operating Characteristic curves comparing all models. Higher AUC indicates better performance.",
                    'PR Curves': "Precision-Recall curves showing the trade-off between precision and recall for each model.",
                    'Calibration Plots': "Reliability curves showing how well-calibrated each model's probability predictions are.",
                    'Confusion Matrices': "Confusion matrices showing true positives, false positives, true negatives, and false negatives.",
                    'SHAP Summary': "SHAP summary plot showing the impact of each feature on model predictions.",
                    'SHAP Importance': "Feature importance based on SHAP values, showing which factors most influence predictions.",
                    'Risk Heatmap': "Risk heatmap showing how maternal age and weight combinations affect risk.",
                    'Model Comparison': "Bar chart comparing all models across different performance metrics."
                }
                
                if selected_viz in descriptions:
                    st.info(descriptions[selected_viz])
            
            # Show all visualizations in expanders
            st.markdown("---")
            st.subheader("All Visualizations")
            
            for viz_name, path in available_viz.items():
                with st.expander(viz_name):
                    img = load_image(path)
                    if img:
                        st.image(img, width='stretch')
                    else:
                        st.warning(f"Could not load {viz_name}")

    # About Page
    elif page == "About":
        st.header("About This Application")
        
        st.write("""
    ## Low Birth Weight Risk Prediction System
    
    This application uses machine learning models to predict the risk of low birth weight 
    (defined as birth weight < 2.5 kg) based on maternal health and demographic factors.
    
    ### Dataset
    - **Source**: MASS::birthwt dataset (Hosmer & Lemeshow, 1989)
    - **Size**: 189 samples from Baystate Medical Center (Springfield, MA, 1986)
    - **Features**: Maternal age, weight, race, smoking, hypertension, uterine irritability, medical visits
    
    ### Models
    The system includes 5 trained machine learning models:
    1. **Logistic Regression** - Baseline linear model
    2. **Logistic Regression (SMOTE)** - With synthetic oversampling for class imbalance
    3. **Random Forest** - Ensemble tree-based model
    4. **XGBoost** - Gradient boosting model
    5. **XGBoost (Calibrated)** - XGBoost with probability calibration
    
    ### Evaluation Metrics
    - **ROC-AUC**: Area under the ROC curve (higher is better, max 1.0)
    - **PR-AUC**: Area under the Precision-Recall curve
    - **Brier Score**: Calibration measure (lower is better, min 0.0)
    - **F1 Score**: Harmonic mean of precision and recall
    - **Recall**: Sensitivity - ability to detect low birth weight cases
    
    ### Clinical Interpretation
    The models identify key risk factors including:
    - Smoking during pregnancy
    - Hypertension
    - Maternal age and weight
    - Uterine irritability
    - Previous premature labors
    
    ### References
    - Hosmer, D.W. & Lemeshow, S. (1989). Applied Logistic Regression. New York: Wiley.
    - Venables, W. N. & Ripley, B. D. (2002). Modern Applied Statistics with S (4th ed.). Springer.
    - Lundberg, S.M. & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP).
    
    ### Disclaimer
    This tool is for research and educational purposes. It should not replace clinical judgment 
    or professional medical advice. Always consult with healthcare professionals for medical decisions.
    """)
    
    st.markdown("---")
    st.subheader("Technical Details")
    st.code("""
    Technologies Used:
    - Python 3.13
    - Streamlit (Web Framework)
    - scikit-learn (Machine Learning)
    - XGBoost (Gradient Boosting)
    - SHAP (Explainability)
    - Plotly (Interactive Visualizations)
    - Pandas, NumPy (Data Processing)
    """, language="text")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>Low Birth Weight Risk Prediction System | Built with Streamlit</div>",
        unsafe_allow_html=True
    )

except Exception as e:
    st.error("An error occurred while loading the application.")
    st.exception(e)
    st.info("Please refresh the page or contact support if the issue persists.")

