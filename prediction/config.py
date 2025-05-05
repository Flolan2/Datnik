# -*- coding: utf-8 -*-
"""
Configuration settings for the DatScan prediction experiments.
Paths are defined relative to the project structure.
"""

import os
import numpy as np
from scipy.stats import randint, loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# --- Path Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # -> .../Datnik/Online/prediction
PREDICTION_DIR = SCRIPT_DIR                           # -> .../Datnik/Online/prediction
ONLINE_DIR = os.path.dirname(PREDICTION_DIR)           # -> .../Datnik/Online
DATNIK_DIR = os.path.dirname(ONLINE_DIR)               # -> .../Datnik
print(f"[Config] Prediction script dir: {PREDICTION_DIR}")
print(f"[Config] Detected parent 'Online' dir: {ONLINE_DIR}")
print(f"[Config] Assuming 'Datnik' directory for Input/Output: {DATNIK_DIR}")

INPUT_FOLDER = os.path.join(DATNIK_DIR, "Input")
OUTPUT_FOLDER_BASE = os.path.join(DATNIK_DIR, "Output")

# --- Unique Output Subfolder Name ---
# Changed to reflect the comparison being done
PREDICTION_SUBFOLDER_NAME = "prediction_leakage_comparison"
DATA_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER_BASE, "Data", PREDICTION_SUBFOLDER_NAME)
PLOT_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER_BASE, "Plots", PREDICTION_SUBFOLDER_NAME)
# --- End Output Subfolder ---

# --- Input Data ---
INPUT_CSV_NAME = "merged_summary_with_medon.csv"

# --- Prediction Target ---
TARGET_IMAGING_BASE = "Contralateral_Striatum"
TARGET_Z_SCORE_COL = f"{TARGET_IMAGING_BASE}_Z"
ABNORMALITY_THRESHOLD = -1.96
TARGET_COLUMN_NAME = 'DatScan_Status'

# --- Grouping Variable ---
GROUP_ID_COL = "Patient ID"

# --- Features ---
BASE_KINEMATIC_COLS = [
    "meanamplitude", "stdamplitude", "meanspeed", "stdspeed", "meanrmsvelocity",
    "stdrmsvelocity", "meanopeningspeed", "stdopeningspeed", "meanclosingspeed",
    "stdclosingspeed", "meancycleduration", "stdcycleduration", "rangecycleduration",
    "rate", "amplitudedecay", "velocitydecay", "ratedecay", "cvamplitude",
    "cvcycleduration", "cvspeed", "cvrmsvelocity", "cvopeningspeed", "cvclosingspeed"
]
TASKS_TO_RUN_SEPARATELY = ['ft', 'hm']

# --- ADDED: Splitting Mode Control ---
# Define which splitting strategies to compare
# 'group': StratifiedGroupKFold based on Patient ID (prevents leakage)
# 'standard': Standard StratifiedKFold ignoring Patient ID (simulates leakage)
SPLITTING_MODES_TO_RUN = ['group', 'standard']
# --- End Splitting Mode Control ---

# --- Experiment Settings ---
BASE_RANDOM_STATE = 42
N_REPETITIONS = 50 # Keeping it reasonably high for stable comparison
TEST_SET_SIZE = 0.25
N_SPLITS_CV = 5
ENABLE_TUNING = True
N_ITER_RANDOM_SEARCH = 30 # Keep as is for now
TUNING_SCORING_METRIC = 'roc_auc'

# --- Experiment Configurations ---
# Keeping the expanded set from previous step
CONFIGURATIONS_TO_RUN = [
    {
        'config_name': 'LR_Standard_Median',
        'model_name': 'logistic',
        'scaler': 'standard', 'imputer': 'median', 'resampler': None,
        'feature_selector': None, 'selector_k': None,
    },
    {
        'config_name': 'RF_Standard_Median',
        'model_name': 'random_forest',
        'scaler': 'standard', 'imputer': 'median', 'resampler': None,
        'feature_selector': None, 'selector_k': None,
    },
    {
        'config_name': 'SVM_Standard_Median',
        'model_name': 'svm',
        'scaler': 'standard', 'imputer': 'median', 'resampler': None,
        'feature_selector': None, 'selector_k': None,
    },
     {
        'config_name': 'GBM_Standard_Median',
        'model_name': 'gradient_boosting',
        'scaler': 'standard', 'imputer': 'median', 'resampler': None,
        'feature_selector': None, 'selector_k': None,
    },
    {
        'config_name': 'KNN_Standard_Median',
        'model_name': 'knn',
        'scaler': 'standard', 'imputer': 'median', 'resampler': None,
        'feature_selector': None, 'selector_k': None,
    },
]

# --- Model Definitions and Hyperparameter Spaces ---
MODEL_PIPELINE_STEPS = {
    'logistic': {
        'estimator': LogisticRegression(random_state=None, class_weight='balanced', max_iter=2000, solver='liblinear'),
        'param_dist': { 'classifier__C': loguniform(1e-3, 1e3), 'classifier__penalty': ['l1', 'l2'] }
    },
    'random_forest': {
        'estimator': RandomForestClassifier(random_state=None, class_weight='balanced', n_jobs=-1),
        'param_dist': { 'classifier__n_estimators': randint(50, 400), 'classifier__max_depth': [None, 10, 20, 30, 40],
                        'classifier__min_samples_split': randint(2, 15), 'classifier__min_samples_leaf': randint(1, 15),
                        'classifier__max_features': ['sqrt', 'log2', 0.5, 0.7] }
    },
    'svm': {
        'estimator': SVC(random_state=None, class_weight='balanced', probability=True),
        'param_dist': { 'classifier__C': loguniform(1e-2, 1e3), 'classifier__gamma': loguniform(1e-4, 1e-1), 'classifier__kernel': ['rbf', 'linear'] }
    },
    'gradient_boosting': {
        'estimator': GradientBoostingClassifier(random_state=None, validation_fraction=0.1, n_iter_no_change=5, tol=0.01),
        'param_dist': { 'classifier__n_estimators': randint(50, 300), 'classifier__learning_rate': loguniform(0.01, 0.3),
                        'classifier__max_depth': randint(3, 10), 'classifier__subsample': loguniform(0.6, 0.4),
                        'classifier__min_samples_split': randint(2, 15), 'classifier__min_samples_leaf': randint(1, 15),
                        'classifier__max_features': ['sqrt', 'log2', None] }
    },
    'knn': {
         'estimator': KNeighborsClassifier(n_jobs=-1),
         'param_dist': { 'classifier__n_neighbors': randint(3, 25), 'classifier__weights': ['uniform', 'distance'],
                         'classifier__metric': ['minkowski', 'euclidean', 'manhattan'] }
    }
}

# --- Stacking Configuration ---
ENABLE_STACKING = True
# Use a potentially stable base config for stacking comparison across modes
STACKING_BASE_CONFIG_NAME = 'LR_Standard_Median'
META_CLASSIFIER_CONFIG = {
    'model_name': 'logistic_meta',
    'estimator': LogisticRegression(random_state=None, class_weight='balanced', max_iter=1000),
}
STACKING_TASKS_TO_COMBINE = ['ft', 'hm']

# --- Output Options ---
SAVE_AGGREGATED_SUMMARY = True
SAVE_AGGREGATED_IMPORTANCES = True
SAVE_META_MODEL_COEFFICIENTS = True
GENERATE_PLOTS = True
PLOT_TOP_N_FEATURES = 15

# --- Imblearn Check --- (Keep as is)
try:
    # ... (imblearn import logic) ...
    IMBLEARN_AVAILABLE = True
except ImportError:
    # ... (set flags) ...
    IMBLEARN_AVAILABLE = False
    print("Warning: 'imbalanced-learn' not found. Resampling options unavailable.")