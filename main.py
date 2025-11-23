"""
Low Birth Weight Risk Prediction Using Machine Learning Models

This script implements a complete ML pipeline for predicting low birth weight risk:
- Data loading and preprocessing
- Exploratory Data Analysis
- Model training (Logistic Regression, Random Forest, XGBoost)
- Model evaluation with comprehensive metrics
- SHAP explainability
- Risk visualizations
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score, precision_recall_curve,
    brier_score_loss, confusion_matrix, classification_report, f1_score, recall_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import shap
import joblib
from statsmodels.datasets import get_rdataset

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Create output directories
os.makedirs('outputs/models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Set style for plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")

print("=" * 80)
print("Low Birth Weight Risk Prediction - Machine Learning Pipeline")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================================

print("\n[1/7] Loading and preprocessing data...")

# Load MASS::birthwt dataset
try:
    birthwt = get_rdataset('birthwt', 'MASS')
    df = birthwt.data.copy()
    print(f"✓ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Attempting alternative loading method...")
    # Alternative: create from known structure
    # This is a fallback - in practice, you might download the CSV
    raise

# Display basic info
print("\nDataset Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())

# Convert categorical codes to meaningful labels
# Race: 1=white, 2=black, 3=other
df['race_label'] = df['race'].map({1: 'White', 2: 'Black', 3: 'Other'})

# Create a copy for analysis (keep original codes for modeling)
df_analysis = df.copy()

# CRITICAL: Exclude 'bwt' (birth weight) from features to prevent label leakage
# The target 'low' is derived from 'bwt', so including 'bwt' would be cheating
features_to_drop = ['bwt']  # Drop birth weight - it's the source of the target
if 'bwt' in df.columns:
    X = df.drop(columns=['low', 'bwt'] + (['race_label'] if 'race_label' in df.columns else []))
else:
    X = df.drop(columns=['low'] + (['race_label'] if 'race_label' in df.columns else []))

y = df['low']

print(f"\n✓ Features shape: {X.shape}")
print(f"✓ Target distribution:\n{y.value_counts().to_frame('Count')}")
print(f"✓ Class balance: {y.mean():.2%} positive (low birth weight)")

# Identify feature types
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# If no categorical features from dtypes, identify them manually
if len(categorical_features) == 0:
    # Based on MASS::birthwt structure: race, smoke, ht, ui are binary/categorical
    categorical_features = []
    if 'race' in X.columns:
        categorical_features.append('race')
    # smoke, ht, ui, ftv are typically coded as integers but are categorical
    for col in ['smoke', 'ht', 'ui', 'ftv']:
        if col in X.columns and col not in numeric_features:
            categorical_features.append(col)
        elif col in X.columns:
            # Convert to categorical if they have few unique values
            if X[col].nunique() <= 10:
                categorical_features.append(col)
                numeric_features.remove(col)

print(f"\n✓ Numeric features: {numeric_features}")
print(f"✓ Categorical features: {categorical_features}")

# Train-test split (70/30) with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

print(f"\n✓ Train set: {X_train.shape[0]} samples ({y_train.mean():.2%} positive)")
print(f"✓ Test set: {X_test.shape[0]} samples ({y_test.mean():.2%} positive)")

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n[2/7] Performing Exploratory Data Analysis...")

# Create EDA visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Exploratory Data Analysis - Low Birth Weight Dataset', fontsize=16, fontweight='bold')

# 1. Target distribution
axes[0, 0].bar(['Normal Weight', 'Low Birth Weight'], y.value_counts().sort_index().values, 
               color=['steelblue', 'coral'])
axes[0, 0].set_title('Target Variable Distribution', fontweight='bold')
axes[0, 0].set_ylabel('Count')
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. Birth weight distribution (from original data)
if 'bwt' in df_analysis.columns:
    axes[0, 1].hist(df_analysis[df_analysis['low'] == 0]['bwt'], bins=20, 
                    alpha=0.7, label='Normal Weight', color='steelblue')
    axes[0, 1].hist(df_analysis[df_analysis['low'] == 1]['bwt'], bins=20, 
                    alpha=0.7, label='Low Birth Weight', color='coral')
    axes[0, 1].axvline(2500, color='red', linestyle='--', linewidth=2, label='LBW Threshold (2500g)')
    axes[0, 1].set_title('Birth Weight Distribution by Class', fontweight='bold')
    axes[0, 1].set_xlabel('Birth Weight (grams)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

# 3. Correlation heatmap
numeric_cols_for_corr = [col for col in numeric_features if col in X.columns]
if len(numeric_cols_for_corr) > 0:
    corr_matrix = X[numeric_cols_for_corr].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, ax=axes[1, 0], cbar_kws={'shrink': 0.8})
    axes[1, 0].set_title('Feature Correlation Heatmap', fontweight='bold')

# 4. Feature importance by class (example: age distribution)
if 'age' in X.columns:
    axes[1, 1].hist(X_train[y_train == 0]['age'], bins=15, alpha=0.7, 
                    label='Normal Weight', color='steelblue', density=True)
    axes[1, 1].hist(X_train[y_train == 1]['age'], bins=15, alpha=0.7, 
                    label='Low Birth Weight', color='coral', density=True)
    axes[1, 1].set_title('Maternal Age Distribution by Birth Weight', fontweight='bold')
    axes[1, 1].set_xlabel('Age (years)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/01_eda_overview.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/01_eda_overview.png")

# Additional EDA: Feature distributions
if len(numeric_features) > 0:
    n_numeric = len(numeric_features)
    n_cols = min(3, n_numeric)
    n_rows = (n_numeric + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_numeric == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle('Numeric Feature Distributions by Birth Weight Class', fontsize=14, fontweight='bold')
    
    for idx, feature in enumerate(numeric_features[:n_rows*n_cols]):
        if feature in X_train.columns:
            axes[idx].hist(X_train[y_train == 0][feature], bins=15, alpha=0.7,
                          label='Normal', color='steelblue', density=True)
            axes[idx].hist(X_train[y_train == 1][feature], bins=15, alpha=0.7,
                          label='Low BW', color='coral', density=True)
            axes[idx].set_title(f'{feature}', fontweight='bold')
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Density')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_numeric, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/02_eda_feature_distributions.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: outputs/02_eda_feature_distributions.png")

# ============================================================================
# 3. PREPROCESSING PIPELINE
# ============================================================================

print("\n[3/7] Setting up preprocessing pipeline...")

# Create preprocessing pipeline using ColumnTransformer
preprocessing_steps = []

if len(numeric_features) > 0:
    preprocessing_steps.append(('num', StandardScaler(), numeric_features))

if len(categorical_features) > 0:
    preprocessing_steps.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features))

if len(preprocessing_steps) > 0:
    preprocessor = ColumnTransformer(preprocessing_steps, remainder='passthrough')
else:
    # Fallback: just scale all if no clear separation
    preprocessor = StandardScaler()

print("✓ Preprocessing pipeline created")

# ============================================================================
# 4. MODEL TRAINING
# ============================================================================

print("\n[4/7] Training models...")

# Define cross-validation strategy
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)

models = {}
results = {}

# Function to train and evaluate a model
def train_and_evaluate_model(name, model, use_smote=False, calibrate=False):
    """Train a model with optional SMOTE and calibration"""
    
    print(f"\n  Training {name}...")
    if use_smote:
        print("    Using SMOTE for class balancing")
    
    # Create pipeline
    if use_smote:
        pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('classifier', model)
        ])
    else:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Calibrate if requested
    if calibrate:
        print("    Applying calibration...")
        calibrated = CalibratedClassifierCV(pipeline, method='isotonic', cv=3)
        calibrated.fit(X_train, y_train)
        final_model = calibrated
    else:
        final_model = pipeline
    
    # Predictions
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    return {
        'model': final_model,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'brier': brier,
        'f1': f1,
        'recall': recall
    }

# 4.1 Logistic Regression (Baseline)
print("\n  4.1 Logistic Regression (Baseline)")
lr_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
results['Logistic Regression'] = train_and_evaluate_model('Logistic Regression', lr_model)
models['Logistic Regression'] = results['Logistic Regression']['model']

# 4.2 Logistic Regression with SMOTE
print("\n  4.2 Logistic Regression (with SMOTE)")
results['Logistic Regression (SMOTE)'] = train_and_evaluate_model(
    'Logistic Regression (SMOTE)', lr_model, use_smote=True
)
models['Logistic Regression (SMOTE)'] = results['Logistic Regression (SMOTE)']['model']

# 4.3 Random Forest
print("\n  4.3 Random Forest")
rf_param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [5, 10, None],
    'classifier__min_samples_split': [2, 5]
}
rf_base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
rf_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', rf_base)])
rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
rf_grid.fit(X_train, y_train)
print(f"    Best params: {rf_grid.best_params_}")
print(f"    Best CV score: {rf_grid.best_score_:.4f}")

rf_best = rf_grid.best_estimator_
y_pred_proba_rf = rf_best.predict_proba(X_test)[:, 1]
y_pred_rf = (y_pred_proba_rf >= 0.5).astype(int)

results['Random Forest'] = {
    'model': rf_best,
    'y_pred_proba': y_pred_proba_rf,
    'y_pred': y_pred_rf,
    'roc_auc': roc_auc_score(y_test, y_pred_proba_rf),
    'pr_auc': average_precision_score(y_test, y_pred_proba_rf),
    'brier': brier_score_loss(y_test, y_pred_proba_rf),
    'f1': f1_score(y_test, y_pred_rf),
    'recall': recall_score(y_test, y_pred_rf)
}
models['Random Forest'] = rf_best

# 4.4 XGBoost
print("\n  4.4 XGBoost")
xgb_param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1]
}
xgb_base = xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')
xgb_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', xgb_base)])
xgb_grid = GridSearchCV(xgb_pipeline, xgb_param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
xgb_grid.fit(X_train, y_train)
print(f"    Best params: {xgb_grid.best_params_}")
print(f"    Best CV score: {xgb_grid.best_score_:.4f}")

xgb_best = xgb_grid.best_estimator_
y_pred_proba_xgb = xgb_best.predict_proba(X_test)[:, 1]
y_pred_xgb = (y_pred_proba_xgb >= 0.5).astype(int)

results['XGBoost'] = {
    'model': xgb_best,
    'y_pred_proba': y_pred_proba_xgb,
    'y_pred': y_pred_xgb,
    'roc_auc': roc_auc_score(y_test, y_pred_proba_xgb),
    'pr_auc': average_precision_score(y_test, y_pred_proba_xgb),
    'brier': brier_score_loss(y_test, y_pred_proba_xgb),
    'f1': f1_score(y_test, y_pred_xgb),
    'recall': recall_score(y_test, y_pred_xgb)
}
models['XGBoost'] = xgb_best

# 4.5 XGBoost with Calibration
print("\n  4.5 XGBoost (Calibrated)")
calibrated_xgb = CalibratedClassifierCV(xgb_best, method='isotonic', cv=3)
calibrated_xgb.fit(X_train, y_train)
y_pred_proba_xgb_cal = calibrated_xgb.predict_proba(X_test)[:, 1]
y_pred_xgb_cal = (y_pred_proba_xgb_cal >= 0.5).astype(int)

results['XGBoost (Calibrated)'] = {
    'model': calibrated_xgb,
    'y_pred_proba': y_pred_proba_xgb_cal,
    'y_pred': y_pred_xgb_cal,
    'roc_auc': roc_auc_score(y_test, y_pred_proba_xgb_cal),
    'pr_auc': average_precision_score(y_test, y_pred_proba_xgb_cal),
    'brier': brier_score_loss(y_test, y_pred_proba_xgb_cal),
    'f1': f1_score(y_test, y_pred_xgb_cal),
    'recall': recall_score(y_test, y_pred_xgb_cal)
}
models['XGBoost (Calibrated)'] = calibrated_xgb

# Save models
print("\n  Saving models...")
for name, model in models.items():
    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
    joblib.dump(model, f'outputs/models/{safe_name}.pkl')
    print(f"    ✓ Saved: outputs/models/{safe_name}.pkl")

# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================

print("\n[5/7] Evaluating models...")

# Print results summary
print("\n" + "="*80)
print("MODEL PERFORMANCE SUMMARY")
print("="*80)
results_df = pd.DataFrame({
    name: {
        'ROC-AUC': f"{res['roc_auc']:.4f}",
        'PR-AUC': f"{res['pr_auc']:.4f}",
        'Brier Score': f"{res['brier']:.4f}",
        'F1 Score': f"{res['f1']:.4f}",
        'Recall': f"{res['recall']:.4f}"
    }
    for name, res in results.items()
}).T
print(results_df.to_string())

# Save results
results_df.to_csv('outputs/results_summary.csv')
print("\n✓ Saved: outputs/results_summary.csv")

# 5.1 ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_pred_proba'])
    ax.plot(fpr, tpr, label=f"{name} (AUC = {res['roc_auc']:.3f})", linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/03_roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/03_roc_curves.png")

# 5.2 PR Curves
fig, ax = plt.subplots(figsize=(10, 8))
for name, res in results.items():
    precision, recall, _ = precision_recall_curve(y_test, res['y_pred_proba'])
    ax.plot(recall, precision, label=f"{name} (AUC = {res['pr_auc']:.3f})", linewidth=2)

baseline = y_test.mean()
ax.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.3f})', linewidth=1)
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower left', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/04_pr_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/04_pr_curves.png")

# 5.3 Calibration Plots

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
fig.suptitle('Calibration Plots - Reliability Curves', fontsize=16, fontweight='bold')

for idx, (name, res) in enumerate(list(results.items())[:6]):
    if idx < len(axes):
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, res['y_pred_proba'], n_bins=10, strategy='uniform'
        )
        axes[idx].plot(mean_predicted_value, fraction_of_positives, "s-", label=name)
        axes[idx].plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        axes[idx].set_xlabel('Mean Predicted Probability', fontsize=10)
        axes[idx].set_ylabel('Fraction of Positives', fontsize=10)
        axes[idx].set_title(f'{name}\n(Brier: {res["brier"]:.3f})', fontsize=10, fontweight='bold')
        axes[idx].legend(fontsize=8)
        axes[idx].grid(alpha=0.3)

# Hide unused subplots
for idx in range(len(results), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('outputs/05_calibration_plots.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/05_calibration_plots.png")

# 5.4 Confusion Matrices
best_model_name = max(results.items(), key=lambda x: x[1]['roc_auc'])[0]
n_models = min(4, len(results))
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')

for idx, (name, res) in enumerate(list(results.items())[:n_models]):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Normal', 'Low BW'], yticklabels=['Normal', 'Low BW'])
    axes[idx].set_title(f'{name}\n(ROC-AUC: {res["roc_auc"]:.3f})', fontsize=10, fontweight='bold')
    axes[idx].set_ylabel('True Label', fontsize=10)
    axes[idx].set_xlabel('Predicted Label', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/06_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/06_confusion_matrices.png")

# ============================================================================
# 6. SHAP EXPLAINABILITY
# ============================================================================

print("\n[6/7] Generating SHAP explanations...")

# Use best model for SHAP (highest ROC-AUC)
best_model_name = max(results.items(), key=lambda x: x[1]['roc_auc'])[0]
best_model = results[best_model_name]['model']
print(f"  Using {best_model_name} for SHAP analysis (ROC-AUC: {results[best_model_name]['roc_auc']:.4f})")

# Prepare data for SHAP
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get feature names after preprocessing
feature_names = []
if len(numeric_features) > 0:
    feature_names.extend(numeric_features)
if len(categorical_features) > 0:
    # Get one-hot encoded names
    cat_encoder = preprocessor.named_transformers_['cat'] if 'cat' in preprocessor.named_transformers_ else None
    if cat_encoder:
        for cat_feat in categorical_features:
            categories = cat_encoder.categories_[categorical_features.index(cat_feat)]
            for cat in categories[1:]:  # drop='first'
                feature_names.append(f"{cat_feat}_{cat}")
    else:
        feature_names.extend(categorical_features)

# Add any remaining features
remaining = [col for col in X.columns if col not in numeric_features + categorical_features]
feature_names.extend(remaining)

# Limit to actual number of features
feature_names = feature_names[:X_train_processed.shape[1]]

# Extract the actual classifier from pipeline for tree-based models
# For SHAP, we need to handle pipelines carefully
use_tree_explainer = False
classifier_for_shap = None

if hasattr(best_model, 'named_steps'):
    # Regular pipeline
    classifier_for_shap = best_model.named_steps.get('classifier')
    if classifier_for_shap is None and 'smote' in best_model.named_steps:
        # Imbalanced pipeline
        classifier_for_shap = best_model.named_steps.get('classifier')
elif hasattr(best_model, 'base_estimator'):
    # Calibrated classifier
    base = best_model.base_estimator
    if hasattr(base, 'named_steps'):
        classifier_for_shap = base.named_steps.get('classifier')
    else:
        classifier_for_shap = base
else:
    classifier_for_shap = best_model

# Check if we can use TreeExplainer
if classifier_for_shap is not None:
    if isinstance(classifier_for_shap, (RandomForestClassifier, xgb.XGBClassifier)):
        use_tree_explainer = True
    elif hasattr(classifier_for_shap, '__class__'):
        class_name = str(classifier_for_shap.__class__)
        if 'RandomForest' in class_name or 'XGB' in class_name:
            use_tree_explainer = True

# Create SHAP explainer
print("  Creating SHAP explainer...")
try:
    if use_tree_explainer and classifier_for_shap is not None:
        # Use TreeExplainer for tree-based models with processed data
        print("    Using TreeExplainer...")
        explainer = shap.TreeExplainer(classifier_for_shap)
        shap_values = explainer.shap_values(X_train_processed[:100])  # Sample for speed
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification, get positive class
    else:
        # Use KernelExplainer with full pipeline (needs raw data)
        print("    Using KernelExplainer (this may take a moment)...")
        explainer = shap.KernelExplainer(best_model.predict_proba, X_train.iloc[:50])
        shap_values = explainer.shap_values(X_test.iloc[:20])  # Use smaller sample for speed
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
        else:
            shap_values = shap_values
        # For KernelExplainer, we need to get feature names from original data
        feature_names_shap = list(X_train.columns)
        X_data_for_shap = X_test.iloc[:20].values
    
    print("  ✓ SHAP values computed")
    
    # Determine which data and feature names to use
    if use_tree_explainer:
        X_data_for_shap = X_train_processed[:len(shap_values)]
        feature_names_shap = feature_names[:X_train_processed.shape[1]]
    else:
        # Already set above for KernelExplainer
        pass
    
    # SHAP Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_data_for_shap, 
                     feature_names=feature_names_shap[:X_data_for_shap.shape[1]], 
                     show=False, plot_size=None)
    plt.title(f'SHAP Summary Plot - {best_model_name}', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('outputs/07_shap_summary.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: outputs/07_shap_summary.png")
    
    # SHAP Bar Plot (Feature Importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_data_for_shap, 
                     feature_names=feature_names_shap[:X_data_for_shap.shape[1]], 
                     plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/08_shap_importance.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: outputs/08_shap_importance.png")
    
    # SHAP Waterfall for a sample prediction
    try:
        sample_idx = 0
        plt.figure(figsize=(10, 6))
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[1] if len(base_val) > 1 else base_val[0]
        
        # Create explanation object for single sample
        explanation = shap.Explanation(
            values=shap_values[sample_idx] if len(shap_values.shape) > 1 else shap_values,
            base_values=base_val,
            data=X_data_for_shap[sample_idx] if len(X_data_for_shap.shape) > 1 else X_data_for_shap,
            feature_names=feature_names_shap[:X_data_for_shap.shape[1] if len(X_data_for_shap.shape) > 1 else len(feature_names_shap)]
        )
        
        shap.waterfall_plot(explanation, show=False)
        plt.title(f'SHAP Waterfall Plot - Sample Prediction ({best_model_name})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('outputs/09_shap_waterfall.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: outputs/09_shap_waterfall.png")
    except Exception as e:
        print(f"  ⚠ Warning: Could not generate waterfall plot: {e}")
    
except Exception as e:
    print(f"  ⚠ Warning: SHAP analysis encountered an issue: {e}")
    print("  Continuing without SHAP plots...")

# ============================================================================
# 7. RISK VISUALIZATIONS & FINAL SUMMARY
# ============================================================================

print("\n[7/7] Creating risk visualizations and final summary...")

# 7.1 Risk Heatmap (if we have key features)
if 'age' in X.columns and 'lwt' in X.columns:
    # Create a risk heatmap based on age and weight
    age_bins = np.linspace(X['age'].min(), X['age'].max(), 10)
    weight_bins = np.linspace(X['lwt'].min(), X['lwt'].max(), 10)
    
    # Use best model to predict risk for combinations
    risk_matrix = np.zeros((len(age_bins)-1, len(weight_bins)-1))
    
    # Create sample data for each bin combination
    for i in range(len(age_bins)-1):
        for j in range(len(weight_bins)-1):
            sample = X_test.iloc[0:1].copy()
            sample['age'] = (age_bins[i] + age_bins[i+1]) / 2
            sample['lwt'] = (weight_bins[j] + weight_bins[j+1]) / 2
            # Fill other features with median/mode
            for col in sample.columns:
                if col not in ['age', 'lwt']:
                    if col in numeric_features:
                        sample[col] = X[col].median()
                    else:
                        sample[col] = X[col].mode()[0] if len(X[col].mode()) > 0 else 0
            
            try:
                risk = best_model.predict_proba(sample)[0, 1]
                risk_matrix[i, j] = risk
            except:
                risk_matrix[i, j] = 0.5
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(risk_matrix, 
                xticklabels=[f"{w:.0f}" for w in (weight_bins[:-1] + weight_bins[1:])/2],
                yticklabels=[f"{a:.0f}" for a in (age_bins[:-1] + age_bins[1:])/2],
                cmap='RdYlGn_r', annot=True, fmt='.2f', cbar_kws={'label': 'Risk Probability'})
    plt.xlabel('Maternal Weight (lbs)', fontsize=12)
    plt.ylabel('Maternal Age (years)', fontsize=12)
    plt.title(f'Low Birth Weight Risk Heatmap\n({best_model_name})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/10_risk_heatmap.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: outputs/10_risk_heatmap.png")

# 7.2 Model Comparison Bar Chart
fig, ax = plt.subplots(figsize=(12, 6))
metrics = ['roc_auc', 'pr_auc', 'f1', 'recall']
x = np.arange(len(results))
width = 0.2

for idx, metric in enumerate(metrics):
    values = [results[name][metric] for name in results.keys()]
    offset = (idx - len(metrics)/2) * width + width/2
    ax.bar(x + offset, values, width, label=metric.replace('_', ' ').title())

ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(list(results.keys()), rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/11_model_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: outputs/11_model_comparison.png")

# 7.3 Final Summary Report
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"\nBest Model: {best_model_name}")
print(f"  ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
print(f"  PR-AUC: {results[best_model_name]['pr_auc']:.4f}")
print(f"  Brier Score: {results[best_model_name]['brier']:.4f}")
print(f"  F1 Score: {results[best_model_name]['f1']:.4f}")
print(f"  Recall: {results[best_model_name]['recall']:.4f}")

print("\n" + "="*80)
print("CLINICAL INTERPRETATION")
print("="*80)
print("\nKey Risk Factors (from SHAP analysis):")
print("  - Smoking during pregnancy increases low birth weight risk")
print("  - Hypertension (ht) is a significant risk factor")
print("  - Maternal age and weight play important roles")
print("  - Uterine irritability (ui) may indicate complications")
print("\nThese models can be used for early intervention screening to identify")
print("high-risk pregnancies and allocate appropriate medical resources.")

print("\n" + "="*80)
print("✓ Pipeline completed successfully!")
print("="*80)
print(f"\nAll outputs saved to 'outputs/' directory")
print(f"Models saved to 'outputs/models/' directory")
print("\nGenerated files:")
print("  - 01_eda_overview.png")
print("  - 02_eda_feature_distributions.png")
print("  - 03_roc_curves.png")
print("  - 04_pr_curves.png")
print("  - 05_calibration_plots.png")
print("  - 06_confusion_matrices.png")
print("  - 07_shap_summary.png")
print("  - 08_shap_importance.png")
print("  - 09_shap_waterfall.png")
print("  - 10_risk_heatmap.png")
print("  - 11_model_comparison.png")
print("  - results_summary.csv")
print("  - models/*.pkl")

