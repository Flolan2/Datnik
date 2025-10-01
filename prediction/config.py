# --- START OF FILE prediction/config.py (FINAL, WITH NEW FEATURES) ---
# -*- coding: utf-8 -*-
"""
Configuration settings for the DatScan prediction experiments.
PUBLICATION RUN: Focusing on best LR+RFE models with and without
final engineered features for FT and HM tasks. All analyses are age-controlled.
"""

import os
import numpy as np
from scipy.stats import randint, loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier # Keep for RFE estimator if not directly used

# --- Path Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is /Users/Lange_L/Documents/Kinematik/Datnik/Online/prediction
PREDICTION_DIR = SCRIPT_DIR
ONLINE_DIR = os.path.dirname(PREDICTION_DIR)            # This is /Users/Lange_L/Documents/Kinematik/Datnik/Online
PROJECT_ROOT_DIR = os.path.dirname(ONLINE_DIR)          # <<< RENAMED: This is /Users/Lange_L/Documents/Kinematik/Datnik
print(f"[Config] Prediction script dir: {PREDICTION_DIR}")
print(f"[Config] Detected parent 'Online' dir: {ONLINE_DIR}")
print(f"[Config] Assuming project root directory: {PROJECT_ROOT_DIR}")

# --- Input Data ---
INPUT_FOLDER = os.path.join(PROJECT_ROOT_DIR, "Output", "Data_Processed")
INPUT_CSV_NAME = "final_merged_data.csv"

# --- Output Folders for THIS prediction experiment ---
PREDICTION_SUBFOLDER_NAME = "prediction_binary_PublicationRun_V2_AgeControlled_FE_Test" # Updated folder name for new test
OUTPUT_FOLDER_BASE = os.path.join(PROJECT_ROOT_DIR, "Output")
DATA_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER_BASE, "Prediction_Results", PREDICTION_SUBFOLDER_NAME)
PLOT_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER_BASE, "Prediction_Plots", PREDICTION_SUBFOLDER_NAME)

# --- Prediction Target (BINARY) ---
TARGET_IMAGING_BASE = "Contralateral_Putamen" # IMPORTANT: Matching the successful run
TARGET_Z_SCORE_COL = f"{TARGET_IMAGING_BASE}_Z"
ABNORMALITY_THRESHOLD = -1.96
TARGET_COLUMN_NAME = 'DatScan_Status_AgeControlled'

# --- Grouping and Covariate Variables ---
GROUP_ID_COL = "Patient ID"
AGE_COL = 'Age' # --- MODIFIED: Define the Age column for mandatory control ---

# --- Features ---
BASE_KINEMATIC_COLS = [
    "meanamplitude", "stdamplitude", "meanspeed", "stdspeed", "meanrmsvelocity",
    "stdrmsvelocity", "meanopeningspeed", "stdopeningspeed", "meanclosingspeed",
    "stdclosingspeed", "meancycleduration", "stdcycleduration", "rangecycleduration",
    "rate", "amplitudedecay", "velocitydecay", "ratedecay", "cvamplitude",
    "cvcycleduration", "cvspeed", "cvrmsvelocity", "cvopeningspeed", "cvclosingspeed"
]

# --- Splitting Mode Control ---
SPLITTING_MODES_TO_RUN = ['group']

# --- Experiment Settings ---
BASE_RANDOM_STATE = 42
N_REPETITIONS = 1100
TEST_SET_SIZE = 0.25
N_SPLITS_CV = 4
ENABLE_TUNING = True
N_ITER_RANDOM_SEARCH = 30
TUNING_SCORING_METRIC = 'roc_auc'

# --- Feature Engineering Configuration ---
FEATURE_ENGINEERING_SETS_OPTIMIZED = [
    # === NEW HYPOTHESES TO TEST (BASED ON PREVIOUS RFE RESULTS) ===
    {
        'name': 'ft_inter_composite_variability',
        'function': 'create_interaction_terms',
        'params': {
            'col1': 'ft_cvamplitude',
            'col2': 'ft_cvrmsvelocity',
            'new_col_name': 'ft_inter_composite_variability'
        }
    },
    {
        'name': 'ft_ratio_slowness_adjusted_decay',
        'function': 'create_ratios',
        'params': {
            'num_col': 'ft_ratedecay',
            'den_col': 'ft_meancycleduration',
            'new_col_name': 'ft_ratio_slowness_adjusted_decay'
        }
    },
    {
        'name': 'ft_log_meancycleduration_from_suggestion', # Renamed to avoid clash
        'function': 'create_log_transform',
        'params': {
            'col': 'ft_meancycleduration',
            'new_col_name': 'ft_log_meancycleduration'
        }
    },

    # === FT: Previously Selected Engineered Features ===
    {
        'name': 'ft_meancycleduration_poly',
        'function': 'create_polynomial_features',
        'params': {'col': 'ft_meancycleduration', 'degree': 3, 'new_col_prefix': 'ft_meancycleduration_poly'}
    },
    # {'name': 'ft_log_meancycleduration', 'function': 'create_log_transform', 'params': {'col': 'ft_meancycleduration', 'new_col_name': 'ft_log_meancycleduration'}}, # This is a duplicate of our new one
    {
        'name': 'ft_cvamplitude_sq',
        'function': 'create_polynomial_features',
        'params': {'col': 'ft_cvamplitude', 'degree': 2, 'new_col_prefix': 'ft_cvamplitude_poly'}
    },
    {
        'name': 'ft_rate_poly',
        'function': 'create_polynomial_features',
        'params': {'col': 'ft_rate', 'degree': 3, 'new_col_prefix': 'ft_rate_poly'}
    },
    {
        'name': 'ft_log_rate',
        'function': 'create_log_transform',
        'params': {'col': 'ft_rate', 'new_col_name': 'ft_log_rate'}
    },
    {
        'name': 'ft_cvspeed_sq',
        'function': 'create_polynomial_features',
        'params': {'col': 'ft_cvspeed', 'degree': 2, 'new_col_prefix': 'ft_cvspeed_poly'}
    },
    {
        'name': 'ft_stdopeningspeed_sq',
        'function': 'create_polynomial_features',
        'params': {'col': 'ft_stdopeningspeed', 'degree': 2, 'new_col_prefix': 'ft_stdopeningspeed_poly'}
    },
    {
        'name': 'ft_interaction_lograte_cvspeedsq',
        'function': 'create_interaction_terms',
        'params': {'col1': 'ft_log_rate', 'col2': 'ft_cvspeed_poly_deg2', 'new_col_name': 'ft_inter_lograte_cvspeedsq'}
    },

    # === HM: Highly Selected Engineered Features ===
    # (These can remain, they won't affect the FT-only run)
    {
        'name': 'hm_stdopeningspeed_poly',
        'function': 'create_polynomial_features',
        'params': {'col': 'hm_stdopeningspeed', 'degree': 3, 'new_col_prefix': 'hm_stdopeningspeed_poly'}
    },
    {
        'name': 'hm_log_stdopeningspeed',
        'function': 'create_log_transform',
        'params': {'col': 'hm_stdopeningspeed', 'new_col_name': 'hm_log_stdopeningspeed'}
    },
    {
        'name': 'hm_stdspeed_poly',
        'function': 'create_polynomial_features',
        'params': {'col': 'hm_stdspeed', 'degree': 3, 'new_col_prefix': 'hm_stdspeed_poly'}
    },
    {
        'name': 'hm_log_stdspeed',
        'function': 'create_log_transform',
        'params': {'col': 'hm_stdspeed', 'new_col_name': 'hm_log_stdspeed'}
    },
    {
        'name': 'hm_stdrmsvelocity_sq',
        'function': 'create_polynomial_features',
        'params': {'col': 'hm_stdrmsvelocity', 'degree': 2, 'new_col_prefix': 'hm_stdrmsvelocity_poly'}
    },
    {
        'name': 'hm_interaction_stdopenpoly3_mcd',
        'function': 'create_interaction_terms',
        'params': {'col1': 'hm_stdopeningspeed_poly_deg3', 'col2': 'hm_meancycleduration', 'new_col_name': 'hm_inter_stdopenp3_mcd'}
    },
    {
        'name': 'hm_norm_veldecay',
        'function': 'create_ratios',
        'params': {'num_col': 'hm_velocitydecay', 'den_col': 'hm_meanspeed', 'new_col_name': 'hm_norm_veldecay'}
    },
    {
        'name': 'hm_norm_ampdecay',
        'function': 'create_ratios',
        'params': {'num_col': 'hm_amplitudedecay', 'den_col': 'hm_meanamplitude', 'new_col_name': 'hm_norm_ampdecay'}
    },
    {
        'name': 'hm_rate_sq',
        'function': 'create_polynomial_features',
        'params': {'col': 'hm_rate', 'degree': 2, 'new_col_prefix': 'hm_rate_poly'}
    },
    {
        'name': 'hm_meancycleduration_sq',
        'function': 'create_polynomial_features',
        'params': {'col': 'hm_meancycleduration', 'degree': 2, 'new_col_prefix': 'hm_meancycleduration_poly'}
    },
]

# --- Experiment Configurations (Model Pipelines) ---
CONFIGURATIONS_TO_RUN = [
    # --- FT Task ---
    {
        'config_name': 'LR_RFE15_FT_OriginalFeats', # Baseline for FT
        'model_name': 'logistic',
        'apply_feature_engineering': False, # This is our control group
        'task_prefix_for_features': 'ft',
        'scaler': 'standard', 'imputer': 'median', 'resampler': None,
        'feature_selector': 'rfe', 'selector_k': 15,
        'search_type': 'random'
    },
    {
        'config_name': 'LR_RFE15_FT_EngFeats_OptimizedV2', # This is our experimental group
        'model_name': 'logistic',
        'apply_feature_engineering': True,
        'feature_engineering_definitions': FEATURE_ENGINEERING_SETS_OPTIMIZED,
        'task_prefix_for_features': 'ft',
        'scaler': 'standard', 'imputer': 'median', 'resampler': None,
        'feature_selector': 'rfe', 'selector_k': 15,
        'search_type': 'random'
    },
    
    # --- HM Task ---
    {
        'config_name': 'LR_RFE15_HM_OriginalFeats', # Baseline for HM
        'model_name': 'logistic',
        'apply_feature_engineering': False, # Control group using original features
        'task_prefix_for_features': 'hm',
        'scaler': 'standard', 'imputer': 'median', 'resampler': None,
        'feature_selector': 'rfe', 'selector_k': 15,
        'search_type': 'random'
    },
# ----------------------------------------
]

# --- Model Definitions and Hyperparameter Spaces ---
MODEL_PIPELINE_STEPS = {
    'logistic': {
        'estimator': LogisticRegression(random_state=None, class_weight='balanced', max_iter=2000, solver='liblinear'),
        'param_dist': {
            'classifier__C': loguniform(1e-3, 1e3),
            'classifier__penalty': ['l1', 'l2']
        },
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
    },
}

# --- Stacking Configuration ---
ENABLE_STACKING = False

# --- Output Options ---
SAVE_AGGREGATED_SUMMARY = True
SAVE_AGGREGATED_IMPORTANCES = True
SAVE_META_MODEL_COEFFICIENTS = False
GENERATE_PLOTS = True
PLOT_TOP_N_FEATURES = 15

# --- Imblearn Check ---
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
    RESAMPLER_OPTIONS = {'smote': SMOTE(random_state=None)}
except ImportError:
    ImbPipeline = None; SMOTE = None; IMBLEARN_AVAILABLE = False
    RESAMPLER_OPTIONS = {}
    print("[Config] Warning: 'imbalanced-learn' not found. Resampling options unavailable.")

print("[Config] PUBLICATION RUN (AGE-CONTROLLED) Configuration loaded.")
# --- END OF FILE prediction/config.py (FINAL, WITH NEW FEATURES) ---