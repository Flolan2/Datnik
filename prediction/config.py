# -*- coding: utf-8 -*-
"""
Configuration settings for the DatScan prediction experiments.
Paths are defined relative to the project structure.

** FINAL Version - Focused on BINARY classification, group splitting,
** comparing best LR models (RFE vs All Feats) + RF baseline.
** UPDATED: Feature Engineering with polynomial (sq) and log transforms of top original features.
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
PREDICTION_SUBFOLDER_NAME = "prediction_binary_FE_poly_log_v1" # New name for this experiment
DATA_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER_BASE, "Data", PREDICTION_SUBFOLDER_NAME)
PLOT_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER_BASE, "Plots", PREDICTION_SUBFOLDER_NAME)

# --- Input Data ---
INPUT_CSV_NAME = "merged_summary_with_medon.csv"

# --- Prediction Target (BINARY) ---
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

# --- Splitting Mode Control ---
SPLITTING_MODES_TO_RUN = ['group']

# --- Experiment Settings ---
BASE_RANDOM_STATE = 42
N_REPETITIONS = 50 # Or your desired number for a full run (e.g., 50)
TEST_SET_SIZE = 0.25
N_SPLITS_CV = 4    # For inner CV
ENABLE_TUNING = True
N_ITER_RANDOM_SEARCH = 30 # Or your desired number (e.g., 30)
TUNING_SCORING_METRIC = 'roc_auc'

# --- Feature Engineering Configuration ---
ENABLE_FEATURE_ENGINEERING = True # Master switch

# In prediction/config.py


ENABLE_FEATURE_ENGINEERING = True

# In prediction/config.py

ENABLE_FEATURE_ENGINEERING = True


# In prediction/config.py

# --- Unique Output Subfolder Name ---
PREDICTION_SUBFOLDER_NAME = "prediction_binary_FE_MegaRun_v1" # NEW NAME FOR THIS BIG RUN

# ... (other parts of config.py like paths, model definitions, etc., remain the same) ...

# --- Feature Engineering Configuration ---
ENABLE_FEATURE_ENGINEERING = True # Master switch


# --- Feature Engineering Configuration ---
ENABLE_FEATURE_ENGINEERING = True

FEATURE_ENGINEERING_SETS = [
    # === FT: Highly Selected Engineered Features ===
    {
        'name': 'ft_meancycleduration_poly', # Will create _deg2 and _deg3
        'function': 'create_polynomial_features',
        'tasks': ['ft'],
        'params': {'col': 'ft_meancycleduration', 'degree': 3, 'new_col_prefix': 'ft_meancycleduration_poly'}
    },
    {
        'name': 'ft_log_meancycleduration',
        'function': 'create_log_transform',
        'tasks': ['ft'],
        'params': {'col': 'ft_meancycleduration', 'new_col_name': 'ft_log_meancycleduration'}
    },
    {
        'name': 'ft_cvamplitude_sq',
        'function': 'create_polynomial_features',
        'tasks': ['ft'],
        'params': {'col': 'ft_cvamplitude', 'degree': 2, 'new_col_prefix': 'ft_cvamplitude_poly'}
    },
    {
        'name': 'ft_rate_poly', # Will create _deg2 and _deg3
        'function': 'create_polynomial_features',
        'tasks': ['ft'],
        'params': {'col': 'ft_rate', 'degree': 3, 'new_col_prefix': 'ft_rate_poly'}
    },
    {
        'name': 'ft_log_rate',
        'function': 'create_log_transform',
        'tasks': ['ft'],
        'params': {'col': 'ft_rate', 'new_col_name': 'ft_log_rate'}
    },
    {
        'name': 'ft_cvspeed_sq',
        'function': 'create_polynomial_features',
        'tasks': ['ft'],
        'params': {'col': 'ft_cvspeed', 'degree': 2, 'new_col_prefix': 'ft_cvspeed_poly'}
    },
    # The ft_stdopeningspeed_poly_deg2 was NOT highly selected by RFE in MegaRun for FT, but original was.
    # Let's try squaring ft_stdopeningspeed (original was good for RFE)
    {
        'name': 'ft_stdopeningspeed_sq',
        'function': 'create_polynomial_features',
        'tasks': ['ft'],
        'params': {'col': 'ft_stdopeningspeed', 'degree': 2, 'new_col_prefix': 'ft_stdopeningspeed_poly'}
    },
    # Interaction Term for FT (highly selected)
    {
        'name': 'ft_interaction_lograte_cvspeedsq',
        'function': 'create_interaction_terms',
        'tasks': ['ft'],
        'params': {'col1': 'ft_log_rate', 'col2': 'ft_cvspeed_poly_deg2', 'new_col_name': 'ft_inter_lograte_cvspeedsq'}
    },

    # === HM: Highly Selected Engineered Features ===
    {
        'name': 'hm_stdopeningspeed_poly', # Will create _deg2 and _deg3
        'function': 'create_polynomial_features',
        'tasks': ['hm'],
        'params': {'col': 'hm_stdopeningspeed', 'degree': 3, 'new_col_prefix': 'hm_stdopeningspeed_poly'}
    },
    {
        'name': 'hm_log_stdopeningspeed', # Selected moderately by RFE, but less than poly terms
        'function': 'create_log_transform',
        'tasks': ['hm'],
        'params': {'col': 'hm_stdopeningspeed', 'new_col_name': 'hm_log_stdopeningspeed'}
    },
    {
        'name': 'hm_stdspeed_poly', # Will create _deg2 and _deg3
        'function': 'create_polynomial_features',
        'tasks': ['hm'],
        'params': {'col': 'hm_stdspeed', 'degree': 3, 'new_col_prefix': 'hm_stdspeed_poly'}
    },
    {
        'name': 'hm_log_stdspeed',
        'function': 'create_log_transform',
        'tasks': ['hm'],
        'params': {'col': 'hm_stdspeed', 'new_col_name': 'hm_log_stdspeed'}
    },
    {
        'name': 'hm_stdrmsvelocity_sq',
        'function': 'create_polynomial_features',
        'tasks': ['hm'],
        'params': {'col': 'hm_stdrmsvelocity', 'degree': 2, 'new_col_prefix': 'hm_stdrmsvelocity_poly'}
    },
    # Interaction Term for HM (highly selected)
    {
        'name': 'hm_interaction_stdopenpoly3_mcd',
        'function': 'create_interaction_terms',
        'tasks': ['hm'],
        'params': {'col1': 'hm_stdopeningspeed_poly_deg3', 'col2': 'hm_meancycleduration', 'new_col_name': 'hm_inter_stdopenp3_mcd'}
    },
    # Normalized Decay Features for HM (highly selected)
    {
        'name': 'hm_norm_veldecay',
        'function': 'create_ratios',
        'tasks': ['hm'],
        'params': {'num_col': 'hm_velocitydecay', 'den_col': 'hm_meanspeed', 'new_col_name': 'hm_norm_veldecay'}
    },
    {
        'name': 'hm_norm_ampdecay',
        'function': 'create_ratios',
        'tasks': ['hm'],
        'params': {'num_col': 'hm_amplitudedecay', 'den_col': 'hm_meanamplitude', 'new_col_name': 'hm_norm_ampdecay'}
    },
    # Let's add squaring for hm_rate and hm_meancycleduration as their originals were top RFE picks
    {
        'name': 'hm_rate_sq',
        'function': 'create_polynomial_features',
        'tasks': ['hm'],
        'params': {'col': 'hm_rate', 'degree': 2, 'new_col_prefix': 'hm_rate_poly'}
    },
    {
        'name': 'hm_meancycleduration_sq',
        'function': 'create_polynomial_features',
        'tasks': ['hm'],
        'params': {'col': 'hm_meancycleduration', 'degree': 2, 'new_col_prefix': 'hm_meancycleduration_poly'}
    },
]

# --- Experiment Configurations ---
# Same as before, these models will now pick from original + new poly/log features
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
# (Keep these the same as your last successful run)
MODEL_PIPELINE_STEPS = {
    'logistic': {
        'estimator': LogisticRegression(random_state=None, class_weight='balanced', max_iter=2000, solver='liblinear'),
        'param_dist': {
            'classifier__C': loguniform(1e-3, 1e3),
            'classifier__penalty': ['l1', 'l2']
        },
        'param_grid': None
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
ENABLE_STACKING = False # Keeping this disabled
# ... (rest of stacking config can remain, it won't be used) ...

# --- Output Options ---
SAVE_AGGREGATED_SUMMARY = True
SAVE_AGGREGATED_IMPORTANCES = True
SAVE_META_MODEL_COEFFICIENTS = False
GENERATE_PLOTS = True # Set to True for the full run
PLOT_TOP_N_FEATURES = 15

# --- Imblearn Check ---
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    ImbPipeline = None; SMOTE = None; IMBLEARN_AVAILABLE = False
    print("Warning: 'imbalanced-learn' not found. Resampling options unavailable.")