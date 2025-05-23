# --- START OF FILE datnik_predict_datscan.py ---

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone script to predict DatScan status based on kinematic features,
running Logistic Regression models SEPARATELY for Finger Tapping (ft) and
Hand Movement (hm) tasks, INCLUDING MODEL STACKING to combine task predictions.

Includes robustness checks (using StandardScaler or RobustScaler), tuning,
optional resampling, evaluates models, reports aggregated performance
(mean +/- std dev) per task and for stacking, and generates plots.

*** Uses Group-Based Splitting by Patient ID to prevent data leakage. ***
"""

import os
import sys # <<< ADDED IMPORT SYS >>>
import pandas as pd
import numpy as np
import json
from time import time
from scipy.stats import randint, loguniform
import collections # For defaultdict
import warnings

# Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_predict,
    StratifiedGroupKFold # <<< ADDED FOR GROUP SPLITTING
)
# <<< ROBUST SCALER: Import RobustScaler >>>
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score,
    f1_score, precision_score, recall_score, roc_curve
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# Imbalanced-learn imports (optional)
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    ImbPipeline = Pipeline; SMOTE = None; IMBLEARN_AVAILABLE = False
    # print("Warning: 'imbalanced-learn' not found. Resampling options unavailable.")

# Suppress specific warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=FutureWarning) # Suppress future warnings too

# -----------------------------------------------------
# Configuration Parameters - MODIFY AS NEEDED
# -----------------------------------------------------
# --- Input Data ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# script_parent_dir will correctly be /Users/Lange_L/Documents/Kinematik/Datnik
script_parent_dir = os.path.dirname(SCRIPT_DIR)

# <<< CORRECTED PATH DEFINITIONS >>>
# Directly use script_parent_dir to define Input and Output locations
INPUT_FOLDER = os.path.join(script_parent_dir, "Input")
OUTPUT_FOLDER_BASE = os.path.join(script_parent_dir, "Output")
# DATA_OUTPUT_FOLDER and PLOT_OUTPUT_FOLDER depend on OUTPUT_FOLDER_BASE, so they will be correct now
DATA_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER_BASE, "Data")
PLOT_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER_BASE, "Plots")

# <<< NOTE: Make sure this matches the output of the summarize script >>>
INPUT_CSV_NAME = "merged_summary_with_medon.csv" # Or "merged_summary.csv" if that's intended
# --- Prediction Target ---
TARGET_IMAGING_BASE = "Contralateral_Striatum"
TARGET_Z_SCORE_COL = f"{TARGET_IMAGING_BASE}_Z"
ABNORMALITY_THRESHOLD = -1.96 # Standard threshold for Z-score

# --- Grouping Variable ---
GROUP_ID_COL = "Patient ID" # <<< IMPORTANT FOR GROUP SPLIT

# --- Features ---
BASE_KINEMATIC_COLS = [
    "meanamplitude", "stdamplitude", "meanspeed", "stdspeed", "meanrmsvelocity",
    "stdrmsvelocity", "meanopeningspeed", "stdopeningspeed", "meanclosingspeed",
    "stdclosingspeed", "meancycleduration", "stdcycleduration", "rangecycleduration",
    "rate", "amplitudedecay", "velocitydecay", "ratedecay", "cvamplitude",
    "cvcycleduration", "cvspeed", "cvrmsvelocity", "cvopeningspeed", "cvclosingspeed"
]
TASKS_TO_RUN_SEPARATELY = ['ft', 'hm']

# --- Model to Test ---
MODEL_NAME = 'logistic'

# --- Common Evaluation Settings ---
TEST_SET_SIZE = 0.25 # Fraction of PATIENTS in the test set
N_SPLITS_CV = 5      # Number of folds for inner cross-validation (tuning, OOF)
IMPUTATION_STRATEGY = 'median'
BASE_RANDOM_STATE = 42

# <<< ROBUST SCALER: Configuration Flag >>>
USE_ROBUST_SCALER = True # Set to True for RobustScaler, False for StandardScaler

# --- Stacking Configuration ---
ENABLE_STACKING = True
META_MODEL_NAME = 'logistic_meta'
META_CLASSIFIER = LogisticRegression(random_state=BASE_RANDOM_STATE + 100,
                                       class_weight='balanced',
                                       max_iter=1000)
STACKING_TASKS = ['ft', 'hm'] # Base models to feed into stacking

# --- Robustness Check ---
N_REPETITIONS = 100 # Reduce for faster testing, increase (e.g., 50-100) for stable results

# --- Resampling Strategy ---
RESAMPLING_STRATEGY = None # 'smote' or None
if RESAMPLING_STRATEGY == 'smote' and not IMBLEARN_AVAILABLE:
    print(f"Warning: RESAMPLING_STRATEGY='smote' but imblearn not available. Disabling.")
    RESAMPLING_STRATEGY = None

# --- Hyperparameter Tuning (for base models) ---
ENABLE_TUNING = True
N_ITER_RANDOM_SEARCH = 50
TUNING_SCORING_METRIC = 'roc_auc'

# --- Output Options ---
SAVE_INDIVIDUAL_RUN_RESULTS = False # Can create many files if N_REPETITIONS is large
SAVE_AGGREGATED_SUMMARY = True
SAVE_AGGREGATED_IMPORTANCES = True
SAVE_META_MODEL_COEFFICIENTS = True
GENERATE_PLOTS = True
PLOT_TOP_N_COEFFICIENTS = 15

# -----------------------------------------------------
# Helper Function for JSON Serialization (Unchanged)
# -----------------------------------------------------
def convert_numpy_types(obj):
    """Recursively converts NumPy types in a dictionary or list to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        if np.isnan(obj): return None
        elif np.isinf(obj): return None
        else: return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.void)):
        return None
    return obj

# -----------------------------------------------------
# Plotting Functions (Unchanged)
# -----------------------------------------------------
def plot_metric_distributions(metrics_dict, tasks_to_plot, metric_key, title, filename):
    """Generates box plots comparing a metric across tasks (can include 'stacked')."""
    plot_data = []
    present_models = list(metrics_dict.keys())
    valid_tasks_to_plot = [t for t in tasks_to_plot if t in present_models]
    if not valid_tasks_to_plot:
        print(f"Warning: No data found for specified tasks in plot_metric_distributions for '{metric_key}'. Skipping plot.")
        return

    for task_or_model_key in valid_tasks_to_plot:
        model_name_key = list(metrics_dict[task_or_model_key].keys())[0] if metrics_dict[task_or_model_key] else None
        if not model_name_key: continue

        task_metrics = metrics_dict.get(task_or_model_key, {}).get(model_name_key, [])
        task_label = task_or_model_key.upper()
        if task_or_model_key == 'stacked': task_label = "Stacked Model"

        for run_metric_dict in task_metrics:
            value = run_metric_dict.get(metric_key, np.nan)
            if pd.notna(value):
                 plot_data.append({'Task': task_label, metric_key: value})

    if not plot_data:
        print(f"Warning: No data to plot for metric '{metric_key}'. Skipping plot '{filename}'.")
        return

    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(max(6, len(valid_tasks_to_plot)*1.5), 5))
    order = [t.upper() if t != 'stacked' else "Stacked Model" for t in valid_tasks_to_plot]
    sns.boxplot(x='Task', y=metric_key, data=df_plot, palette='viridis', width=0.5, order=order)
    if len(df_plot) <= N_REPETITIONS * len(valid_tasks_to_plot) * 1.5: # Avoid overplotting if many runs
        sns.stripplot(x='Task', y=metric_key, data=df_plot, color=".25", size=4, alpha=0.5, order=order)

    plt.title(title, fontsize=14, weight='bold')
    plt.xlabel("Model / Task", fontsize=12)
    plt.ylabel(metric_key.replace('_', ' ').title(), fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    sns.despine()
    plt.tight_layout()
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Metric distribution plot saved to: {filename}")
    except Exception as e:
        print(f"Error saving metric distribution plot {filename}: {e}")
    finally:
        plt.close()

def plot_aggregated_roc_curves(roc_data_dict, auc_dict, tasks_to_plot, title, filename):
    """Plots average ROC curves with variability (can include 'stacked')."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(7, 7))
    base_fpr = np.linspace(0, 1, 101)

    present_models = list(roc_data_dict.keys())
    valid_tasks_to_plot = [t for t in tasks_to_plot if t in present_models]
    if not valid_tasks_to_plot:
        print(f"Warning: No data found for specified tasks in plot_aggregated_roc_curves. Skipping plot.")
        return

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(valid_tasks_to_plot)))
    plot_successful = False

    for i, task_or_model_key in enumerate(valid_tasks_to_plot):
        model_name_key = list(roc_data_dict[task_or_model_key].keys())[0] if roc_data_dict[task_or_model_key] else None
        if not model_name_key: continue

        task_roc_runs = roc_data_dict.get(task_or_model_key, {}).get(model_name_key, [])
        task_aucs_list = [m.get('roc_auc') for m in auc_dict.get(task_or_model_key, {}).get(model_name_key, []) if pd.notna(m.get('roc_auc'))]

        task_label = task_or_model_key.upper()
        if task_or_model_key == 'stacked': task_label = "Stacked Model"

        if not task_roc_runs or not task_aucs_list:
            # print(f"Warning: Insufficient ROC data or AUCs for '{task_label}'. Skipping its curve.") # Verbose
            continue

        tprs_interp = []
        for run_data in task_roc_runs:
             if run_data and len(run_data) == 2 and run_data[0] is not None and run_data[1] is not None:
                 fpr, tpr = run_data; fpr, tpr = np.array(fpr), np.array(tpr)
                 if len(fpr) < 2 or len(tpr) < 2: continue # Need at least 2 points to interpolate
                 tpr_interp = np.interp(base_fpr, fpr, tpr); tpr_interp[0] = 0.0 # Ensure start at 0
                 tprs_interp.append(tpr_interp)

        if not tprs_interp: # Check if any valid interpolations were made
            # print(f"Warning: No valid interpolated TPRs for '{task_label}'. Skipping curve.") # Verbose
            continue

        # Recalculate task_aucs_list to match the number of successful interpolations
        if len(tprs_interp) != len(task_aucs_list):
             # print(f"Warning: Mismatch interpolated ROC curves ({len(tprs_interp)}) vs initial AUCs ({len(task_aucs_list)}) for {task_label}. Using corresponding AUCs.") # Verbose
             # This implies some ROC curves failed but the metrics were calculated. More robust is to re-filter AUCs.
             # A simpler approach is just use the AUCs we have, assuming they correspond to runs where ROC *could* be calculated
             task_aucs_list = [m.get('roc_auc') for m in auc_dict.get(task_or_model_key, {}).get(model_name_key, []) if pd.notna(m.get('roc_auc'))][:len(tprs_interp)]
             if not task_aucs_list: continue # If filtering leaves no AUCs, skip

        mean_tprs = np.mean(tprs_interp, axis=0); std_tprs = np.std(tprs_interp, axis=0)
        tprs_upper = np.minimum(mean_tprs + std_tprs, 1); tprs_lower = np.maximum(mean_tprs - std_tprs, 0)
        mean_auc = np.mean(task_aucs_list); std_auc = np.std(task_aucs_list)

        label = f'{task_label} (AUC = {mean_auc:.2f} ± {std_auc:.2f})'
        plt.plot(base_fpr, mean_tprs, label=label, color=colors[i], lw=2.5)
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, color=colors[i], alpha=0.2)
        plot_successful = True

    if not plot_successful:
         print("Warning: No ROC curves were successfully plotted.")
         plt.close(); return

    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
    plt.xlim([-0.01, 1.01]); plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title(title, fontsize=14, weight='bold')
    plt.legend(loc='lower right', fontsize=10, frameon=True, facecolor='white', framealpha=0.8)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Aggregated ROC curve plot saved to: {filename}")
    except Exception as e: print(f"Error saving ROC curve plot {filename}: {e}")
    finally: plt.close()

def plot_aggregated_coefficients(coeffs_df, model_label, top_n, title, filename):
    """Plots aggregated coefficients with error bars for a single model/task."""
    plt.style.use('seaborn-v0_8-whitegrid')
    if coeffs_df is None or coeffs_df.empty or top_n <= 0:
        # print(f"Warning: No coefficient data or non-positive top_n for '{model_label}'. Skipping plot '{filename}'.") # Verbose
        return
    if not {'Mean_Coefficient', 'Std_Coefficient'}.issubset(coeffs_df.columns):
        print(f"Warning: Missing required columns in coefficients df for '{model_label}'. Skipping plot '{filename}'.")
        return

    # Sort by absolute mean coefficient value, handle NaNs
    coeffs_df_sorted = coeffs_df.reindex(coeffs_df['Mean_Coefficient'].abs().sort_values(ascending=False, na_position='last').index)
    plot_df = coeffs_df_sorted.head(top_n).copy()
    plot_df.dropna(subset=['Mean_Coefficient'], inplace=True) # Drop features with NaN mean coeff

    if plot_df.empty:
        # print(f"Warning: No coefficients left after filtering/sorting for '{model_label}'. Skipping plot '{filename}'.") # Verbose
        return

    plot_df = plot_df.iloc[::-1] # Highest absolute value at top for horizontal bar plot
    colors = ['#d62728' if c < 0 else '#1f77b4' for c in plot_df['Mean_Coefficient']]

    plt.figure(figsize=(10, max(5, len(plot_df) * 0.4))) # Adjust height based on N features
    plt.barh(
        plot_df.index, plot_df['Mean_Coefficient'],
        xerr=plot_df['Std_Coefficient'].fillna(0), # Use 0 error if std is NaN
        color=colors,
        alpha=0.85, edgecolor='black', linewidth=0.7, capsize=4
    )
    plt.axvline(0, color='dimgrey', linestyle='--', linewidth=1)

    xlabel = 'Mean Coefficient (Log-Odds Change)'
    ylabel = 'Kinematic Feature'
    if 'meta' in model_label.lower() or 'stacked' in model_label.lower():
         xlabel = 'Mean Meta-Model Coefficient'
         ylabel = 'Base Model Prediction'

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, weight='bold')
    plt.grid(axis='x', linestyle=':', alpha=0.7)
    plt.yticks(fontsize=10) # Adjust if labels overlap
    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Aggregated coefficients plot for {model_label} saved to: {filename}")
    except Exception as e: print(f"Error saving coefficients plot for {model_label}: {filename}: {e}")
    finally: plt.close()

# -----------------------------------------------------
# Helper Function to Build Base Pipeline (Modified for RobustScaler)
# -----------------------------------------------------
def build_base_pipeline(model_name: str, imputation_strategy: str, resampling_strategy: str, random_state: int, use_robust_scaler: bool): # << Added use_robust_scaler
    """Builds the scikit-learn pipeline for a base model."""
    pipeline_steps = []
    CurrentPipeline = Pipeline # Default
    scaler_name = 'StandardScaler' # Default name for output

    if imputation_strategy in ['mean', 'median']:
        pipeline_steps.append(('imputer', SimpleImputer(strategy=imputation_strategy)))
    elif imputation_strategy: # Warn if strategy is non-null but not recognized
        print(f"Warning: Unrecognized imputation_strategy '{imputation_strategy}'. No imputer added.")

    if resampling_strategy == 'smote' and SMOTE and IMBLEARN_AVAILABLE:
        pipeline_steps.append(('resampler', SMOTE(random_state=random_state, k_neighbors=4))) # Ensure k_neighbors < samples in smallest class
        CurrentPipeline = ImbPipeline

    # <<< ROBUST SCALER: Conditional Scaler Choice >>>
    if use_robust_scaler:
        pipeline_steps.append(('scaler', RobustScaler()))
        scaler_name = 'RobustScaler'
    else:
        pipeline_steps.append(('scaler', StandardScaler()))
    # <<< END SCALER CHOICE >>>

    if model_name == 'logistic':
        classifier = LogisticRegression(random_state=random_state, class_weight='balanced', max_iter=2000, solver='liblinear') # Liblinear often robust
        # Define hyperparameter search space (remains the same regardless of scaler)
        param_dist = {'classifier__C': loguniform(1e-4, 1e4),
                      # 'classifier__solver': ['liblinear', 'saga'], # Saga can be slow, stick to liblinear?
                      'classifier__penalty': ['l1', 'l2']} # Liblinear supports both
    else:
        raise ValueError(f"Invalid base model_name '{model_name}'")

    pipeline_steps.append(('classifier', classifier))
    pipeline = CurrentPipeline(pipeline_steps)

    # print(f"        Pipeline built using: Imputer='{imputation_strategy}', Resampler='{resampling_strategy}', Scaler='{scaler_name}', Model='{model_name}'") # Reduced verbosity
    return pipeline, param_dist

# -----------------------------------------------------
# Function to Evaluate Predictions (Unchanged)
# -----------------------------------------------------
def evaluate_predictions(y_true, y_pred, y_pred_proba=None):
    """Calculates standard classification metrics."""
    metrics = {}
    metrics['roc_auc'] = np.nan
    roc_curve_data = None # Initialize

    # Ensure y_true and y_pred are numpy arrays for consistent handling
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)

    if y_pred_proba is not None:
        y_pred_proba_np = np.asarray(y_pred_proba)
        try:
            # Check if both classes are present in y_true
            if len(np.unique(y_true_np)) > 1:
                metrics['roc_auc'] = roc_auc_score(y_true_np, y_pred_proba_np)
                fpr, tpr, _ = roc_curve(y_true_np, y_pred_proba_np)
                # Ensure fpr/tpr are valid before storing
                if fpr is not None and tpr is not None and len(fpr) > 1 and len(tpr) > 1:
                     roc_curve_data = (fpr.tolist(), tpr.tolist())
                else: roc_curve_data = None # Invalid curve data
            # else: print("Warning: Only one class present in y_true. ROC AUC is not defined.") # Verbose
        except ValueError as e:
            # print(f"Warning: Could not calculate ROC AUC score: {e}") # Verbose
             roc_curve_data = None # Set to None if calculation fails
    else: roc_curve_data = None

    try: metrics['accuracy'] = accuracy_score(y_true_np, y_pred_np)
    except ValueError: metrics['accuracy'] = np.nan
    try: metrics['f1_macro'] = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    except ValueError: metrics['f1_macro'] = np.nan
    try: metrics['precision_macro'] = precision_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    except ValueError: metrics['precision_macro'] = np.nan
    try: metrics['recall_macro'] = recall_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    except ValueError: metrics['recall_macro'] = np.nan
    try:
         # Ensure y_true has more than one class before calculating confusion matrix if needed by specific sklearn versions
         if len(np.unique(y_true_np)) > 1:
              metrics['confusion_matrix'] = confusion_matrix(y_true_np, y_pred_np).tolist()
         else: metrics['confusion_matrix'] = [] # Return empty if only one class
    except ValueError: metrics['confusion_matrix'] = []

    return metrics, roc_curve_data


# -----------------------------------------------------
# Main Execution Block
# -----------------------------------------------------
if __name__ == '__main__':
    start_time_script = time()
    print("--- Starting DatScan Prediction Script (Task-Specific + Stacking) ---")
    print(f"*** Using Group-Based Splitting by Patient ID ({GROUP_ID_COL}) ***")
    print(f"Number of repetitions: {N_REPETITIONS}")
    if ENABLE_STACKING: print(f"Model Stacking Enabled using tasks: {STACKING_TASKS}")
    print(f"Using scaler: {'RobustScaler' if USE_ROBUST_SCALER else 'StandardScaler'}")
    if RESAMPLING_STRATEGY: print(f"Resampling strategy: {RESAMPLING_STRATEGY}")
    if ENABLE_TUNING: print(f"Hyperparameter tuning: Enabled ({N_ITER_RANDOM_SEARCH} iterations, score: {TUNING_SCORING_METRIC})")

    # --- Create Output Folders ---
    try:
        os.makedirs(DATA_OUTPUT_FOLDER, exist_ok=True)
        if GENERATE_PLOTS: os.makedirs(PLOT_OUTPUT_FOLDER, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directories: {e}. Check permissions or path.")
        sys.exit(1) # <<< USE SYS.EXIT >>>

    # --- Print Folder Paths ---
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Data Output folder: {DATA_OUTPUT_FOLDER}")
    if GENERATE_PLOTS: print(f"Plot Output folder: {PLOT_OUTPUT_FOLDER}")

    # --- 1. Load Data (Once) ---
    input_file_path = os.path.join(INPUT_FOLDER, INPUT_CSV_NAME)
    print(f"\nLoading data from: {input_file_path}")
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at {input_file_path}")
        sys.exit(1) # <<< USE SYS.EXIT >>>
    try:
        # Try semicolon first, then comma
        try: df = pd.read_csv(input_file_path, sep=';', decimal='.')
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
             print("Failed read with ';', trying ','...")
             df = pd.read_csv(input_file_path, sep=',', decimal='.')

        print(f"Data loaded successfully. Shape: {df.shape}")
        if TARGET_Z_SCORE_COL not in df.columns: raise ValueError(f"Target column '{TARGET_Z_SCORE_COL}' not found.")
        if GROUP_ID_COL not in df.columns: raise ValueError(f"Group ID column '{GROUP_ID_COL}' not found.")

    except Exception as e:
        print(f"Error loading or parsing data: {e}")
        sys.exit(1) # <<< USE SYS.EXIT >>>

    # --- 2. Prepare Target Variable and Group ID (Once) ---
    data_full = df.copy()
    # Convert Group ID to string early to handle potential numeric IDs consistently
    data_full[GROUP_ID_COL] = data_full[GROUP_ID_COL].astype(str).str.strip()

    # Convert target Z-score, handling potential non-numeric values
    data_full[TARGET_Z_SCORE_COL] = pd.to_numeric(data_full[TARGET_Z_SCORE_COL].astype(str).str.replace(',', '.'), errors='coerce')
    initial_rows = len(data_full)
    data_full.dropna(subset=[TARGET_Z_SCORE_COL, GROUP_ID_COL], inplace=True) # Drop rows missing target OR group ID
    if len(data_full) < initial_rows:
        print(f"Dropped {initial_rows - len(data_full)} rows with missing target ('{TARGET_Z_SCORE_COL}') or group ID ('{GROUP_ID_COL}').")
    if data_full.empty:
        print("Error: No data after dropping missing target/group ID.")
        sys.exit(1) # <<< USE SYS.EXIT >>>

    target_col_name = 'DatScan_Status'
    data_full[target_col_name] = (data_full[TARGET_Z_SCORE_COL] <= ABNORMALITY_THRESHOLD).astype(int)
    y_full = data_full[target_col_name]
    patient_ids_full = data_full[GROUP_ID_COL] # Group IDs for splitting

    print(f"Target variable '{target_col_name}' distribution: {y_full.value_counts(normalize=True).round(3).to_dict()}")
    print(f"Number of unique patients: {patient_ids_full.nunique()}")
    if len(y_full.unique()) < 2:
        print("Error: Target variable has only one class after preparation. Cannot perform classification.")
        sys.exit(1) # <<< USE SYS.EXIT >>>

    # --- 3. Feature Preparation (Once) ---
    all_feature_cols = []
    task_features_map = {}
    print("\nIdentifying features for tasks:")
    for task_prefix in TASKS_TO_RUN_SEPARATELY:
        task_cols = [col for base in BASE_KINEMATIC_COLS if (col := f"{task_prefix}_{base}") in data_full.columns]
        if task_cols:
            print(f"  - Task '{task_prefix}': Found {len(task_cols)} features.")
            task_features_map[task_prefix] = task_cols
            all_feature_cols.extend(task_cols)
        else: print(f"  - Task '{task_prefix}': Warning - No features found matching BASE_KINEMATIC_COLS.")

    # Ensure we only keep unique feature columns if base names overlap (unlikely with prefixes)
    all_feature_cols = sorted(list(set(all_feature_cols)))
    if not all_feature_cols:
        print("Error: No kinematic features found across all tasks. Exiting.")
        sys.exit(1) # <<< USE SYS.EXIT >>>
    print(f"Total unique kinematic features identified: {len(all_feature_cols)}")

    X_full = data_full[all_feature_cols].copy()
    # Convert all feature columns to numeric, coercing errors
    for col in all_feature_cols:
        X_full[col] = pd.to_numeric(X_full[col].astype(str).str.replace(',', '.'), errors='coerce')

    # Handle missing feature values (Note: Imputation happens inside pipeline now)
    if IMPUTATION_STRATEGY is None:
        initial_rows = len(X_full)
        rows_with_nan = X_full.isnull().any(axis=1)
        if rows_with_nan.any():
            valid_indices = X_full.dropna().index
            X_full = X_full.loc[valid_indices]
            y_full = y_full.loc[valid_indices]
            patient_ids_full = patient_ids_full.loc[valid_indices] # Keep groups aligned
            print(f"Dropped {initial_rows - len(X_full)} rows with missing feature values (Imputation strategy is None).")
        if X_full.empty:
            print("Error: No data left after dropping rows with NaNs in features.")
            sys.exit(1) # <<< USE SYS.EXIT >>>

    # Final check on data size
    if len(X_full) < 20 or patient_ids_full.nunique() < N_SPLITS_CV * 2 : # Need enough patients for splits
        print(f"Error: Insufficient data after preparation. Rows={len(X_full)}, Unique Patients={patient_ids_full.nunique()}. Need more data/patients for reliable splitting/CV. Exiting.")
        sys.exit(1) # <<< USE SYS.EXIT >>>
    print(f"Final data shape for modeling: X={X_full.shape}, y={len(y_full)}, Groups={patient_ids_full.nunique()}")

    # --- 4. Robustness Check Loop with Group Splitting and Stacking ---
    # ... (Rest of the script, including the main loop and aggregation, remains unchanged) ...
    all_runs_metrics = collections.defaultdict(lambda: collections.defaultdict(list))
    all_runs_base_importances = collections.defaultdict(lambda: collections.defaultdict(list))
    all_runs_roc_data = collections.defaultdict(lambda: collections.defaultdict(list))
    all_runs_meta_importances = collections.defaultdict(list)

    print(f"\n--- Starting {N_REPETITIONS} Repetitions (Group-Split by Patient ID) ---")
    for i in range(N_REPETITIONS):
        current_random_state = BASE_RANDOM_STATE + i
        if (i+1) % max(1, N_REPETITIONS // 10) == 0 or i == 0 or N_REPETITIONS <= 10: # Adjust print frequency
             print(f"\n  Repetition {i+1}/{N_REPETITIONS} (Seed: {current_random_state})...")

        # --- 4a. Group-Based Train/Test Split ---
        unique_patients = patient_ids_full.unique()
        n_patients = len(unique_patients)
        n_test_patients = int(np.ceil(n_patients * TEST_SET_SIZE))

        # Ensure feasible split sizes
        if n_test_patients >= n_patients: n_test_patients = max(1, n_patients - 1) # Keep at least one for training
        if n_test_patients < 1: n_test_patients = 1

        # Shuffle patient IDs using a dedicated RandomState for this repetition
        rng = np.random.RandomState(current_random_state)
        shuffled_patients = rng.permutation(unique_patients)

        test_patient_ids = set(shuffled_patients[:n_test_patients])
        train_patient_ids = set(shuffled_patients[n_test_patients:])

        # Create boolean masks based on whether the row's Patient ID is in the respective set
        train_mask = patient_ids_full.isin(train_patient_ids)
        test_mask = patient_ids_full.isin(test_patient_ids)

        # Apply masks to get the actual data splits
        X_train_val = X_full[train_mask].copy() # Use .copy() to avoid SettingWithCopyWarning later
        y_train_val = y_full[train_mask].copy()
        X_test = X_full[test_mask].copy()
        y_test = y_full[test_mask].copy()
        groups_train_val = patient_ids_full[train_mask].copy() # Patient IDs corresponding to train_val rows

        print(f"   Split: {len(train_patient_ids)} patients ({len(X_train_val)} rows) train | {len(test_patient_ids)} patients ({len(X_test)} rows) test")

        # --- Validity Checks for the Split ---
        if X_train_val.empty or X_test.empty:
            print(f"   Warning: Rep {i+1} group split resulted in an empty train or test set. Skipping run.")
            for task_prefix in TASKS_TO_RUN_SEPARATELY: all_runs_metrics[task_prefix][MODEL_NAME].append({})
            if ENABLE_STACKING: all_runs_metrics['stacked'][META_MODEL_NAME].append({})
            continue # Skip to next repetition

        # Check if train or test set has only one class (can cause issues with stratified CV / metrics)
        if len(y_train_val.unique()) < 2 or len(y_test.unique()) < 2:
            print(f"   Warning: Rep {i+1} group split resulted in a single class in train ({len(y_train_val.unique())}) or test ({len(y_test.unique())}) set. "
                  "Models might fail or metrics (like ROC AUC) be undefined. Skipping run.")
            for task_prefix in TASKS_TO_RUN_SEPARATELY: all_runs_metrics[task_prefix][MODEL_NAME].append({})
            if ENABLE_STACKING: all_runs_metrics['stacked'][META_MODEL_NAME].append({})
            continue # Skip this repetition

        # --- 4b. Train Base Models, Generate Predictions ---
        oof_preds = {}
        test_preds = {}
        base_model_pipelines = {}
        # print(f"    Training Base Models ({', '.join(TASKS_TO_RUN_SEPARATELY)})...") # Reduced verbosity
        base_model_training_successful = False

        for task_prefix in TASKS_TO_RUN_SEPARATELY:
            task_feature_cols = task_features_map.get(task_prefix)
            if not task_feature_cols:
                print(f"      [{task_prefix}] No features for this task. Skipping.")
                continue

            X_train_val_task = X_train_val[task_feature_cols]
            X_test_task = X_test[task_feature_cols]

            # Build pipeline (pass scaler choice)
            try:
                pipeline, param_dist = build_base_pipeline(
                    MODEL_NAME, IMPUTATION_STRATEGY, RESAMPLING_STRATEGY, current_random_state,
                    use_robust_scaler=USE_ROBUST_SCALER
                )
            except ValueError as e_build:
                print(f"      [{task_prefix}] Error building pipeline: {e_build}. Skipping task for this run.")
                all_runs_metrics[task_prefix][MODEL_NAME].append({})
                continue

            best_pipeline = clone(pipeline)
            tuning_best_score = None
            best_params = None
            fit_successful = False

            if ENABLE_TUNING:
                # Use StratifiedGroupKFold for tuning, requires 'groups'
                cv_tune = StratifiedGroupKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=current_random_state)
                random_search = RandomizedSearchCV(
                    estimator=pipeline, param_distributions=param_dist, n_iter=N_ITER_RANDOM_SEARCH,
                    scoring=TUNING_SCORING_METRIC, cv=cv_tune, # Use group-aware CV
                    random_state=current_random_state,
                    n_jobs=-1, refit=True, error_score='raise' # Raise error during tuning for easier debug
                )
                try:
                    # Pass groups to the fit method for group-aware CV
                    random_search.fit(X_train_val_task, y_train_val, groups=groups_train_val)
                    best_params = random_search.best_params_
                    best_pipeline = random_search.best_estimator_
                    tuning_best_score = random_search.best_score_
                    fit_successful = True
                except Exception as e_tune:
                    print(f"      [{task_prefix}] Tuning Error with GroupKFold: {repr(e_tune)}. Trying default fit.")
                    # Fallback: Fit the original (non-tuned) pipeline
                    try:
                        best_pipeline = clone(pipeline) # Re-clone the original pipeline
                        best_pipeline.fit(X_train_val_task, y_train_val)
                        fit_successful = True
                        best_params = "Default (Tuning Failed)"
                    except Exception as e_fit_fb: print(f"      [{task_prefix}] Fallback Fit Error after Tuning Failure: {repr(e_fit_fb)}")
            else: # No tuning
                 try:
                     best_pipeline.fit(X_train_val_task, y_train_val)
                     fit_successful = True
                     best_params = "Default (Tuning Disabled)"
                 except Exception as e_fit_def: print(f"      [{task_prefix}] Default Fit Error (No Tuning): {repr(e_fit_def)}")

            # --- Post-Fit Processing ---
            if fit_successful:
                base_model_pipelines[task_prefix] = best_pipeline
                base_model_training_successful = True

                # Generate OOF predictions using StratifiedGroupKFold
                try:
                    cv_predict_fold = StratifiedGroupKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=current_random_state)
                    oof_pred_proba = cross_val_predict(
                        best_pipeline, X_train_val_task, y_train_val,
                        cv=cv_predict_fold, # Use group-aware CV
                        method='predict_proba',
                        n_jobs=-1,
                        groups=groups_train_val # Pass groups
                    )[:, 1] # Probability of the positive class
                    oof_preds[task_prefix] = oof_pred_proba
                except Exception as e_oof:
                     print(f"      [{task_prefix}] OOF Prediction Error with GroupKFold: {repr(e_oof)}")
                     oof_preds[task_prefix] = None # Mark as failed

                # Generate Test predictions and Evaluate
                try:
                    test_pred_proba = best_pipeline.predict_proba(X_test_task)[:, 1]
                    test_pred_labels = best_pipeline.predict(X_test_task)
                    test_preds[task_prefix] = test_pred_proba # Store probabilities for potential stacking

                    # Evaluate using the test set (which was properly group-split)
                    base_metrics, base_roc_data = evaluate_predictions(y_test, test_pred_labels, test_pred_proba)
                    base_metrics['best_cv_score'] = tuning_best_score # Can be None if tuning failed/disabled
                    # base_metrics['best_params'] = best_params # Can be large, maybe omit from summary?
                    all_runs_metrics[task_prefix][MODEL_NAME].append(convert_numpy_types(base_metrics))
                    if base_roc_data: all_runs_roc_data[task_prefix][MODEL_NAME].append(base_roc_data)

                    # Extract coefficients if model allows
                    try:
                        # Access the final step (classifier) which might be inside imblearn Pipeline
                        if isinstance(best_pipeline, ImbPipeline):
                            final_classifier_step = best_pipeline.steps[-1][1]
                        else: # Standard Pipeline
                            final_classifier_step = best_pipeline.steps[-1][1]

                        if hasattr(final_classifier_step, 'coef_'):
                             coeffs = final_classifier_step.coef_[0]
                             # Get feature names *after* potential transformations (e.g., scaling) if possible
                             # Simple case: assumes feature order is preserved by imputer/scaler
                             imp_series = pd.Series(coeffs, index=task_feature_cols)
                             all_runs_base_importances[task_prefix][MODEL_NAME].append(imp_series)
                    except Exception as e_imp: pass # print(f"[{task_prefix}] Coeff extract warning: {e_imp}") # Verbose

                except Exception as e_test:
                     print(f"      [{task_prefix}] Test Predict/Eval Error: {repr(e_test)}")
                     test_preds[task_prefix] = None; all_runs_metrics[task_prefix][MODEL_NAME].append({}) # Log failure
            else: # Fit failed
                 print(f"      [{task_prefix}] Fit failed. Skipping predictions and evaluation for this task/run.")
                 oof_preds[task_prefix] = None; test_preds[task_prefix] = None
                 all_runs_metrics[task_prefix][MODEL_NAME].append({}) # Log failure

        # --- 4c. Train and Evaluate Meta-Model (Stacking) ---
        if ENABLE_STACKING and base_model_training_successful:
            # print(f"    Training Stacked Model ({META_MODEL_NAME})...") # Reduced verbosity
            meta_train_features_list = []
            meta_test_features_list = []
            valid_stacking_tasks = []
            stacking_feature_names = []

            # Collect predictions from successfully trained base models for the specified stacking tasks
            for task in STACKING_TASKS:
                 if task in oof_preds and oof_preds[task] is not None and \
                    task in test_preds and test_preds[task] is not None:
                      # Check if OOF predictions align with y_train_val index/length
                      if len(oof_preds[task]) == len(y_train_val):
                           feature_name = f"{task}_pred_proba"
                           meta_train_features_list.append(pd.Series(oof_preds[task], index=y_train_val.index, name=feature_name))
                           meta_test_features_list.append(pd.Series(test_preds[task], index=y_test.index, name=feature_name))
                           valid_stacking_tasks.append(task)
                           stacking_feature_names.append(feature_name)
                      else:
                           print(f"      [Stacking] Warning: Mismatched OOF prediction length for task '{task}' ({len(oof_preds[task])}) vs y_train_val ({len(y_train_val)}). Skipping task for stacking.")
                 else:
                      print(f"      [Stacking] Note: Task '{task}' did not produce valid predictions. Skipping for stacking.")


            # Proceed only if we have at least one valid base model prediction set
            if len(valid_stacking_tasks) >= 1:
                 meta_train_features_df = pd.concat(meta_train_features_list, axis=1)
                 meta_test_features_df = pd.concat(meta_test_features_list, axis=1)

                 # Ensure no NaNs in meta features (shouldn't happen if base models worked, but check)
                 if meta_train_features_df.isnull().any().any() or meta_test_features_df.isnull().any().any():
                      print("      [Stacking] Error: NaN values found in base model predictions. Skipping stacking.")
                      all_runs_metrics['stacked'][META_MODEL_NAME].append({})
                 else:
                      meta_model = clone(META_CLASSIFIER)
                      try:
                          meta_model.fit(meta_train_features_df, y_train_val)
                          meta_y_pred_test = meta_model.predict(meta_test_features_df)
                          meta_y_pred_proba_test = meta_model.predict_proba(meta_test_features_df)[:, 1] # Prob positive class

                          stacked_metrics, stacked_roc_data = evaluate_predictions(y_test, meta_y_pred_test, meta_y_pred_proba_test)

                          all_runs_metrics['stacked'][META_MODEL_NAME].append(convert_numpy_types(stacked_metrics))
                          if stacked_roc_data: all_runs_roc_data['stacked'][META_MODEL_NAME].append(stacked_roc_data)

                          # Extract meta-model coefficients (importance of each base model's prediction)
                          if hasattr(meta_model, 'coef_'):
                               meta_coeffs = pd.Series(meta_model.coef_[0], index=stacking_feature_names) # Use generated names
                               all_runs_meta_importances[META_MODEL_NAME].append(meta_coeffs)
                      except Exception as e_meta:
                           print(f"      [Stacking] Meta-Model Training/Prediction Error: {repr(e_meta)}")
                           all_runs_metrics['stacked'][META_MODEL_NAME].append({}) # Log failure
            else:
                 print("      [Stacking] Error: Not enough valid base model predictions available for meta-model training. Skipping stacking for this run.")
                 all_runs_metrics['stacked'][META_MODEL_NAME].append({}) # Log failure

        elif ENABLE_STACKING: # Stacking enabled, but no base models trained successfully
             print("    Skipping Stacking for this run: No base models were successfully trained.")
             all_runs_metrics['stacked'][META_MODEL_NAME].append({}) # Log stacking failure

    print("\n--- All Repetitions Finished ---")

    # --- 5. Aggregate and Summarize Results ---
    # ... (Aggregation and Summary section remains unchanged) ...
    print("\n===== Aggregated Performance Summary (Mean +/- StdDev Across Repetitions) =====")
    overall_summary_stats = []
    metric_keys_display = ['roc_auc', 'accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    models_to_summarize = list(all_runs_metrics.keys())

    for model_key in models_to_summarize:
        actual_model_name = list(all_runs_metrics[model_key].keys())[0] if all_runs_metrics[model_key] else None
        if not actual_model_name:
            print(f"\n--- Model/Task: {model_key.upper()} ---")
            print("No results found (Model name unknown or no runs completed)."); continue

        print(f"\n--- Model/Task: {model_key.upper()} (Using Model: {actual_model_name}) ---")
        model_metrics_list = all_runs_metrics[model_key].get(actual_model_name, [])
        valid_model_metrics_list = [m for m in model_metrics_list if isinstance(m, dict) and m and any(pd.notna(v) for k, v in m.items() if k in metric_keys_display)]

        if not valid_model_metrics_list:
            print(f"No successful/valid runs recorded with metrics for this model/task (Total runs attempted: {len(model_metrics_list)})."); continue

        metrics_df = pd.DataFrame(valid_model_metrics_list)
        for mkey in metric_keys_display:
            if mkey not in metrics_df.columns: metrics_df[mkey] = np.nan
        metrics_df_summary = metrics_df[metric_keys_display].copy()

        if metrics_df_summary.isnull().all().all():
             print(f"No valid numeric metrics found for summary calculation (Total valid runs parsed: {len(metrics_df)})."); continue

        means = metrics_df_summary.mean(skipna=True)
        stds = metrics_df_summary.std(skipna=True)
        n_valid_runs = metrics_df_summary[metric_keys_display[0]].notna().sum()

        task_summary = {'Model_Key': model_key, 'N_Valid_Runs': int(n_valid_runs) }
        for key in metric_keys_display:
             task_summary[f'{key}_mean'] = means.get(key, np.nan)
             task_summary[f'{key}_std'] = stds.get(key, np.nan)
        overall_summary_stats.append(task_summary)

        summary_df_task = pd.DataFrame([task_summary])
        print(f"Mean +/- Std Dev across {n_valid_runs} VALID runs:")
        for key in metric_keys_display:
            summary_df_task[f'{key}'] = summary_df_task.apply(
                lambda r: f"{r[f'{key}_mean']:.3f} ± {r[f'{key}_std']:.3f}" if pd.notna(r[f'{key}_mean']) and pd.notna(r[f'{key}_std']) else ("N/A" if pd.isna(r[f'{key}_mean']) else f"{r[f'{key}_mean']:.3f}"),
                axis=1
            )
        display_cols = ['Model_Key', 'N_Valid_Runs'] + metric_keys_display
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(summary_df_task[display_cols].to_string(index=False, justify='center'))

    if SAVE_AGGREGATED_SUMMARY and overall_summary_stats:
        summary_filename = os.path.join(DATA_OUTPUT_FOLDER, "prediction_group_split_comparison_summary.csv")
        try:
             summary_df_final = pd.DataFrame(overall_summary_stats)
             cols_order = ['Model_Key', 'N_Valid_Runs'] + sorted([col for col in summary_df_final.columns if col not in ['Model_Key', 'N_Valid_Runs']])
             summary_df_final = summary_df_final[cols_order]
             summary_df_final.to_csv(summary_filename, index=False, sep=';', decimal='.', float_format='%.6f')
             print(f"\nOverall summary saved to: {summary_filename}")
        except Exception as e: print(f"Error saving overall summary: {e}")

    # --- 6. Aggregate and Save Coefficients ---
    # ... (Aggregation and Saving of Coefficients section remains unchanged) ...
    aggregated_coeffs_dfs = {}

    if SAVE_AGGREGATED_IMPORTANCES:
        print("\n--- Aggregating Base Model Coefficients (Per Task) ---")
        for task_prefix in TASKS_TO_RUN_SEPARATELY:
             coeff_lists = all_runs_base_importances[task_prefix].get(MODEL_NAME, [])
             valid_coeffs = [s for s in coeff_lists if isinstance(s, pd.Series) and not s.empty]

             if valid_coeffs:
                 n_valid = len(valid_coeffs)
                 print(f"Aggregating coefficients for base task: {task_prefix.upper()} ({n_valid} valid runs)")
                 try:
                     coeff_df = pd.concat(valid_coeffs, axis=1, join='outer')
                     agg_coeff = pd.DataFrame({
                         'Mean_Coefficient': coeff_df.mean(axis=1, skipna=True),
                         'Std_Coefficient': coeff_df.std(axis=1, skipna=True),
                         'N_Valid_Runs': coeff_df.notna().sum(axis=1).astype(int)
                     })
                     agg_coeff = agg_coeff.reindex(agg_coeff['Mean_Coefficient'].abs().sort_values(ascending=False, na_position='last').index)
                     aggregated_coeffs_dfs[task_prefix] = agg_coeff
                     fname = os.path.join(DATA_OUTPUT_FOLDER, f"prediction_{MODEL_NAME}_{task_prefix}_agg_coeffs_groupsplit.csv")
                     agg_coeff.to_csv(fname, sep=';', decimal='.', index_label='Feature', float_format='%.6f')
                     print(f"  -> Saved to: {fname}")
                 except Exception as e: print(f"  Error aggregating/saving coefficients for task {task_prefix}: {e}")
             else: print(f"No valid coefficient data found for task {task_prefix}.")

    if ENABLE_STACKING and SAVE_META_MODEL_COEFFICIENTS:
        print("\n--- Aggregating Meta-Model Coefficients ---")
        meta_coeff_lists = all_runs_meta_importances.get(META_MODEL_NAME, [])
        valid_meta_coeffs = [s for s in meta_coeff_lists if isinstance(s, pd.Series) and not s.empty]

        if valid_meta_coeffs:
            n_valid = len(valid_meta_coeffs)
            print(f"Aggregating coefficients for meta-model: {META_MODEL_NAME} ({n_valid} valid runs)")
            try:
                meta_coeff_df = pd.concat(valid_meta_coeffs, axis=1, join='outer')
                agg_meta_coeff = pd.DataFrame({
                    'Mean_Coefficient': meta_coeff_df.mean(axis=1, skipna=True),
                    'Std_Coefficient': meta_coeff_df.std(axis=1, skipna=True),
                    'N_Valid_Runs': meta_coeff_df.notna().sum(axis=1).astype(int)
                })
                agg_meta_coeff = agg_meta_coeff.reindex(agg_meta_coeff['Mean_Coefficient'].abs().sort_values(ascending=False, na_position='last').index)
                aggregated_coeffs_dfs['stacked'] = agg_meta_coeff
                fname = os.path.join(DATA_OUTPUT_FOLDER, f"prediction_{META_MODEL_NAME}_agg_coeffs_groupsplit.csv")
                agg_meta_coeff.to_csv(fname, sep=';', decimal='.', index_label='Base_Model_Prediction_Feature', float_format='%.6f')
                print(f"  -> Saved to: {fname}")
            except Exception as e: print(f"  Error aggregating/saving meta-model coefficients: {e}")
        else: print(f"No valid coefficient data found for meta-model {META_MODEL_NAME}.")


    # --- 7. Generate Plots ---
    # ... (Plot Generation section remains unchanged) ...
    if GENERATE_PLOTS:
        print("\n--- Generating Plots ---")
        sns.set_theme(style="whitegrid")
        plot_keys = []
        if TASKS_TO_RUN_SEPARATELY: plot_keys.extend([task for task in TASKS_TO_RUN_SEPARATELY if task in all_runs_metrics and all_runs_metrics[task]])
        if ENABLE_STACKING and 'stacked' in all_runs_metrics:
             stacked_model_name = list(all_runs_metrics['stacked'].keys())[0] if all_runs_metrics['stacked'] else None
             if stacked_model_name and all_runs_metrics['stacked'].get(stacked_model_name):
                 plot_keys.append('stacked')

        if not plot_keys:
            print("No models produced results suitable for plotting.")
        else:
            print(f"Plotting results for models/tasks: {plot_keys}")
            for metric in ['roc_auc', 'accuracy', 'f1_macro']:
                 plot_title = f'Test {metric.replace("_"," ").title()} Distribution ({N_REPETITIONS} Runs, Group Split)'
                 fname = os.path.join(PLOT_OUTPUT_FOLDER, f"plot_metric_distribution_{metric}_comparison_groupsplit.png")
                 plot_metric_distributions(all_runs_metrics, plot_keys, metric, plot_title, fname)

            has_roc_data = any(all_runs_roc_data.get(key) for key in plot_keys)
            if has_roc_data:
                 plot_title_roc = f'Average ROC Curves Comparison ({N_REPETITIONS} Runs, Group Split)'
                 fname = os.path.join(PLOT_OUTPUT_FOLDER, "plot_aggregated_roc_curves_comparison_groupsplit.png")
                 plot_aggregated_roc_curves(all_runs_roc_data, all_runs_metrics, plot_keys, plot_title_roc, fname)
            else: print("Skipping ROC curve plot: No valid ROC data recorded across runs/models.")

            for model_key, agg_coeff_df in aggregated_coeffs_dfs.items():
                 if agg_coeff_df is not None and not agg_coeff_df.empty:
                     n_valid_runs_list = agg_coeff_df['N_Valid_Runs']
                     if not n_valid_runs_list.empty:
                         n_min, n_max = int(n_valid_runs_list.min()), int(n_valid_runs_list.max())
                         n_runs_info = f"{n_min}" if n_min == n_max else f"{n_min}-{n_max}"
                     else: n_runs_info = "N/A"

                     if model_key == 'stacked':
                         model_label = f"Stacked Model ({META_MODEL_NAME})"
                         title = f'Aggregated Meta-Model Coefficients\n({n_runs_info} Valid Runs, Group Split)'
                         fname = os.path.join(PLOT_OUTPUT_FOLDER, f"plot_aggregated_coefficients_{META_MODEL_NAME}_groupsplit.png")
                         top_n_plot = len(agg_coeff_df)
                     else:
                         model_label = f"Task: {model_key.upper()} ({MODEL_NAME})"
                         title = f'Top {PLOT_TOP_N_COEFFICIENTS} Aggregated Coefficients\n{model_label} ({n_runs_info} Valid Runs, Group Split)'
                         fname = os.path.join(PLOT_OUTPUT_FOLDER, f"plot_aggregated_coefficients_{MODEL_NAME}_{model_key}_groupsplit.png")
                         top_n_plot = PLOT_TOP_N_COEFFICIENTS

                     plot_aggregated_coefficients(agg_coeff_df, model_label, top_n_plot, title, fname)


    print(f"\n--- Script Finished ({time() - start_time_script:.1f}s Total) ---")