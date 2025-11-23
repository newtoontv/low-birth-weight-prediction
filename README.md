# Low Birth Weight Risk Prediction Using Machine Learning

## Project Overview

This project develops and evaluates machine learning models to predict the likelihood of low birth weight (LBW) - defined as infant weight below 2.5 kg - based on maternal health and demographic factors.

## Dataset

- **Source**: MASS::birthwt dataset (Hosmer & Lemeshow, 1989)
- **Size**: 189 samples × 10 features
- **Location**: Baystate Medical Center (Springfield, MA, 1986)
- **Target Variable**: `low` (1 = birth weight < 2500g)

## Features

- Maternal age
- Maternal weight
- Race
- Smoking status
- Hypertension
- Uterine irritability
- Medical visit frequency
- Other risk factors

## Models Implemented

1. **Logistic Regression** - Baseline model
2. **Random Forest** - Ensemble method with hyperparameter tuning
3. **XGBoost** - Gradient boosting with hyperparameter tuning

## Evaluation Metrics

- ROC-AUC score
- PR-AUC score
- Brier score
- Calibration plots
- Confusion matrices
- SHAP explainability

## Installation

### Option 1: Local Development

1. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install OpenMP (macOS - required for XGBoost):**
```bash
brew install libomp
```

### Option 2: Docker

```bash
docker build -t lbw-prediction-app .
docker run -p 8501:8501 lbw-prediction-app
```

Or use docker-compose:
```bash
docker-compose up -d
```

## Usage

### Step 1: Train Models

First, train the machine learning models:

```bash
python main.py
```

This will:
- Load and preprocess the dataset
- Train 5 different models
- Generate evaluation metrics
- Create visualizations
- Save models to `outputs/models/`

### Step 2: Run Web Application

Start the interactive web application:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Web Application Features

The Streamlit web application provides:

-  **Home Dashboard**: Overview of models and performance
-  **Model Dashboard**: Interactive comparison of all models
-  **Make Prediction**: Input patient data and get instant risk predictions
-  **Visualizations**: Browse all generated plots and charts
-  **About**: Project information and documentation

### Prediction Interface

Enter patient information:
- Maternal age, weight, race
- Smoking status, hypertension history
- Uterine irritability, physician visits
- Previous premature labors

Get instant predictions with:
- Risk probability percentage
- Visual risk gauge
- Clinical interpretation
- Model selection

## Output

All results, models, and visualizations are saved in the `outputs/` directory:
- `outputs/models/` - Trained model files (.pkl)
- `outputs/*.png` - All visualization plots
- `outputs/results_summary.csv` - Evaluation metrics

## Production Deployment

### Simple Deployment (Recommended)

**For Streamlit Cloud (Free & Easy - No Docker):**
See [SIMPLE_DEPLOY.md](SIMPLE_DEPLOY.md) for step-by-step instructions.

1. Push code to GitHub
2. Deploy on [share.streamlit.io](https://share.streamlit.io)
3. Done! Your app is live

### Advanced Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment options:
- Docker deployment
- Cloud deployment (AWS, Azure, GCP)
- Heroku
- Production best practices

## References

- Hosmer, D.W. & Lemeshow, S. (1989). Applied Logistic Regression. New York: Wiley.
- Venables, W. N. & Ripley, B. D. (2002). Modern Applied Statistics with S (4th ed.). Springer.
- Lundberg, S.M. & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP).

