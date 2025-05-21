# -*- coding: utf-8 -*-
"""
Configuration settings for the DatScan prediction experiments.
Paths are defined relative to the project structure.

** FINAL Version - Focused on BINARY classification, group splitting,
** comparing best LR models (RFE vs All Feats) + RF baseline.
"""

import os
import numpy as np
from scipy.stats import randint, loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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
# <<< MODIFIED: Set a specific name for this binary run >>>
PREDICTION_SUBFOLDER_NAME = "prediction_binary_final"
DATA_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER_BASE, "Data", PREDICTION_SUBFOLDER_NAME)
PLOT_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER_BASE, "Plots", PREDICTION_SUBFOLDER_NAME)
# --- End Output Subfolder ---

# --- Input Data ---
INPUT_CSV_NAME = "merged_summary_with_medon.csv"

# --- Prediction Target (BINARY) ---
TARGET_IMAGING_BASE = "Contralateral_Striatum"
TARGET_Z_SCORE_COL = f"{TARGET_IMAGING_BASE}_Z"
ABNORMALITY_THRESHOLD = -1.96  # Standard threshold for abnormality
TARGET_COLUMN_NAME = 'DatScan_Status' # Binary target column name

# --- PROBLEM TYPE (Implicitly Binary) ---
# PROBLEM_TYPE = 'binary' # <<< REMOVED: No longer needed for switching logic

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

# --- Splitting Mode Control ---
SPLITTING_MODES_TO_RUN = ['group'] # Focus on valid methodology only

# --- Experiment Settings ---
BASE_RANDOM_STATE = 42
N_REPETITIONS = 50
TEST_SET_SIZE = 0.25
N_SPLITS_CV = 4
ENABLE_TUNING = True
N_ITER_RANDOM_SEARCH = 30
TUNING_SCORING_METRIC = 'roc_auc' # <<< Ensure this is set for binary

# --- Experiment Configurations ---
# Final set comparing best LR approaches and RF baseline
CONFIGURATIONS_TO_RUN = [
    {
        'config_name': 'LR_Standard_Median_Random',
        'model_name': 'logistic',
        'scaler': 'standard', 'imputer': 'median', 'resampler': None,
        'feature_selector': None, 'selector_k': None,
        'search_type': 'random'
    },
    {
        'config_name': 'LR_Standard_Median_RFE15_Random',
        'model_name': 'logistic',
        'scaler': 'standard', 'imputer': 'median', 'resampler': None,
        'feature_selector': 'rfe',
        'selector_k': 15,
        'search_type': 'random'
    },
    {
        'config_name': 'RF_Standard_Median_Random',
        'model_name': 'random_forest',
        'scaler': 'standard', 'imputer': 'median', 'resampler': None,
        'feature_selector': None, 'selector_k': None,
        'search_type': 'random'
    },
]

# --- Model Definitions and Hyperparameter Spaces ---
MODEL_PIPELINE_STEPS = {
    'logistic': {
        'estimator': LogisticRegression(random_state=None, class_weight='balanced', max_iter=2000, solver='liblinear'),
        'param_dist': {
            'classifier__C': loguniform(1e-3, 1e3),
            'classifier__penalty': ['l1', 'l2']
        },
        'param_grid': None # Not used for random search
    },
    'random_forest': {
        'estimator': RandomForestClassifier(random_state=None, class_weight='balanced', n_jobs=-1),
        'param_dist': {
             'classifier__n_estimators': randint(50, 400),
             'classifier__max_depth': [None, 10, 20, 30, 40],
             'classifier__min_samples_split': randint(2, 15),
             'classifier__min_samples_leaf': randint(1, 15),
             'classifier__max_features': ['sqrt', 'log2', 0.5, 0.7]
         },
         'param_grid': None
    },
}

# --- Stacking Configuration ---
ENABLE_STACKING = False # Stacking disabled
# Stacking settings below are ignored if ENABLE_STACKING is False
STACKING_BASE_CONFIG_NAME = 'LR_Standard_Median_RFE15_Random'
META_CLASSIFIER_CONFIG = {
    'model_name': 'logistic_meta',
    'estimator': LogisticRegression(random_state=None, class_weight='balanced', max_iter=1000),
}
STACKING_TASKS_TO_COMBINE = ['ft', 'hm']

# --- Output Options ---
SAVE_AGGREGATED_SUMMARY = True
SAVE_AGGREGATED_IMPORTANCES = True
SAVE_META_MODEL_COEFFICIENTS = False # Stacking disabled
GENERATE_PLOTS = True
PLOT_TOP_N_FEATURES = 15

# --- Imblearn Check ---
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    ImbPipeline = None; SMOTE = None; IMBLEARN_AVAILABLE = False
    print("Warning: 'imbalanced-learn' not found. Resampling options unavailable.")