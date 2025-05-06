# -*- coding: utf-8 -*-
"""
Configuration settings for the DatScan prediction experiments.
Paths are defined relative to the project structure.

** FINAL Version - Focused on group splitting and comparing best LR models (RFE vs All Feats) + RF baseline.
** Stacking is disabled as it didn't improve results.
"""

import os
import numpy as np
from scipy.stats import randint, loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# --- Imports for commented-out models (kept for reference) ---
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# --- End imports for commented-out models ---


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
PREDICTION_SUBFOLDER_NAME = "prediction_final_LR_RF_RFE" # Final comparison run name
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

# --- Splitting Mode Control ---
SPLITTING_MODES_TO_RUN = ['group'] # Focus on valid methodology only
# --- End Splitting Mode Control ---

# --- Experiment Settings ---
BASE_RANDOM_STATE = 42
N_REPETITIONS = 50 # Keep high for final results
TEST_SET_SIZE = 0.25
N_SPLITS_CV = 4      # Using 4 for stability
ENABLE_TUNING = True
N_ITER_RANDOM_SEARCH = 30 # Keep reasonable number for RandomizedSearch
TUNING_SCORING_METRIC = 'roc_auc'

# --- Experiment Configurations ---
# Final set comparing best LR approaches and RF baseline
CONFIGURATIONS_TO_RUN = [
    # --- Logistic Regression (All Features) - Tuned with RandomizedSearchCV ---
    {
        'config_name': 'LR_Standard_Median_Random',
        'model_name': 'logistic',
        'scaler': 'standard', 'imputer': 'median', 'resampler': None,
        'feature_selector': None, 'selector_k': None,
        'search_type': 'random'
    },
    # --- Logistic Regression (RFE Features, k=15) - Tuned with RandomizedSearchCV ---
    {
        'config_name': 'LR_Standard_Median_RFE15_Random', # Best performer so far
        'model_name': 'logistic',
        'scaler': 'standard', 'imputer': 'median', 'resampler': None,
        'feature_selector': 'rfe',
        'selector_k': 15, # Fixed K based on previous tests
        'search_type': 'random'
    },
    # --- Random Forest (All Features) - Tuned with RandomizedSearchCV ---
    {
        'config_name': 'RF_Standard_Median_Random',
        'model_name': 'random_forest',
        'scaler': 'standard', 'imputer': 'median', 'resampler': None,
        'feature_selector': None, 'selector_k': None,
        'search_type': 'random'
    },
]

# --- Model Definitions and Hyperparameter Spaces ---
# Using only param_dist for RandomizedSearch
MODEL_PIPELINE_STEPS = {
    'logistic': {
        'estimator': LogisticRegression(random_state=None, class_weight='balanced', max_iter=2000, solver='liblinear'),
        'param_dist': {
            'classifier__C': loguniform(1e-3, 1e3),
            'classifier__penalty': ['l1', 'l2']
        },
        'param_grid': None # Not used
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
ENABLE_STACKING = False # <<< DISABLED for final run, as it didn't outperform single best model
STACKING_BASE_CONFIG_NAME = 'LR_Standard_Median_RFE15_Random' # Base on best model if re-enabled
META_CLASSIFIER_CONFIG = {
    'model_name': 'logistic_meta',
    'estimator': LogisticRegression(random_state=None, class_weight='balanced', max_iter=1000),
}
STACKING_TASKS_TO_COMBINE = ['ft', 'hm']

# --- Output Options ---
SAVE_AGGREGATED_SUMMARY = True
SAVE_AGGREGATED_IMPORTANCES = True
SAVE_META_MODEL_COEFFICIENTS = False # Stacking disabled, so no meta coefficients
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


# =============================================================================
# === Summary of Models & Variants Tried During Development ===
# =============================================================================
#
# 1. Initial Leakage Comparison:
#    - Compared 'group' (StratifiedGroupKFold) vs 'standard' (StratifiedKFold) splitting.
#    - Result: 'standard' mode showed massively inflated performance (AUC >0.8-0.9),
#              confirming severe data leakage when ignoring Patient IDs.
#    - Conclusion: Only 'group' mode results are scientifically valid.
#
# 2. Initial Model Screening (Group Mode):
#    - LR_Robust_Median: Showed numerical instability (very large coefficients). Removed.
#    - LR_Standard_Median_SMOTE: SMOTE inactive. Equivalent to LR_Standard_Median. Kept LR+StdScaler.
#    - RF_Standard_Median: Performed modestly, kept as baseline.
#    - SVM_Standard_Median: Performed poorly (esp. HM task), low valid runs. Removed.
#    - GBM_Standard_Median: Failed completely (N_Valid_Runs=0). Removed.
#    - KNN_Standard_Median: Performed poorly (AUC < 0.6). Removed.
#
# 3. Stability Tuning:
#    - Observed frequent "ValueError: Only one class present in y_true" during inner CV with N_SPLITS_CV=5.
#    - Action: Reduced N_SPLITS_CV to 4, which significantly improved N_Valid_Runs (to 46/50).
#
# 4. Stacking Experiments:
#    - Stacking based on LR_Robust_Median: Performed poorly (AUC ~0.55), overweighted weak FT model.
#    - Stacking based on RF_Standard_Median: Performed *worse* than chance (AUC ~0.37), RF probabilities likely not suitable.
#    - Stacking based on LR_Standard_Median_Random: Performed okay (AUC ~0.63) but did *not* outperform the best single base model (LR_Standard_Median_Random on FT, AUC ~0.67).
#    - Stacking based on LR_Standard_Median_RFE15_Random: Performed okay (AUC ~0.63) but did *not* outperform the best single base model (LR_Standard_Median_RFE15_Random on FT, AUC ~0.68).
#    - Conclusion: Simple logistic stacking doesn't add value here. Disabled for final run.
#
# 5. Feature Selection Experiments:
#    - SelectKBest (k=15): Did not improve (or slightly worsened) AUC compared to using all features for LR and RF. Discarded.
#    - RFE (k=15): Showed a slight improvement for LR on FT features (AUC ~0.68 vs ~0.67). Kept this configuration.
#    - GridSearchCV + RFE (tuning K): Failed consistently due to internal errors in nested CV. Discarded.
#
# 6. Final Comparison Set:
#    - LR_Standard_Median_Random (All features, RandomizedSearch)
#    - LR_Standard_Median_RFE15_Random (Top 15 features via RFE, RandomizedSearch) - Current best performer
#    - RF_Standard_Median_Random (All features, RandomizedSearch) - Comparison model
#
# =============================================================================