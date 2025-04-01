#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone script to predict DatScan status based on kinematic features,
running Logistic Regression models SEPARATELY for Finger Tapping (ft) and
Hand Movement (hm) tasks, INCLUDING MODEL STACKING to combine task predictions.

Includes robustness checks (using StandardScaler or RobustScaler), tuning,
optional resampling, evaluates models, reports aggregated performance
(mean +/- std dev) per task and for stacking, and generates plots.
"""

import os
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
    train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_predict
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

# -----------------------------------------------------
# Configuration Parameters - MODIFY AS NEEDED
# -----------------------------------------------------
# --- Input Data ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
script_parent_dir = os.path.dirname(SCRIPT_DIR)
if os.path.basename(SCRIPT_DIR).lower() in ['code', 'scripts']:
    base_dir = script_parent_dir
else:
    base_dir = SCRIPT_DIR

INPUT_FOLDER = os.path.join(base_dir, "Input")
OUTPUT_FOLDER_BASE = os.path.join(base_dir, "Output")
DATA_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER_BASE, "Data")
PLOT_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER_BASE, "Plots")

INPUT_CSV_NAME = "merged_summary.csv"

# --- Prediction Target ---
TARGET_IMAGING_BASE = "Contralateral_Striatum"
TARGET_Z_SCORE_COL = f"{TARGET_IMAGING_BASE}_Z"
ABNORMALITY_THRESHOLD = -1.96

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
TEST_SET_SIZE = 0.25
N_SPLITS_CV = 5
IMPUTATION_STRATEGY = 'median'
BASE_RANDOM_STATE = 42

# <<< ROBUST SCALER: Configuration Flag >>>
# Set to True to use RobustScaler (less sensitive to outliers), False for StandardScaler
USE_ROBUST_SCALER = True # <<< MODIFY THIS FLAG AS NEEDED

# --- Stacking Configuration ---
ENABLE_STACKING = True
META_MODEL_NAME = 'logistic_meta'
META_CLASSIFIER = LogisticRegression(random_state=BASE_RANDOM_STATE + 100,
                                       class_weight='balanced',
                                       max_iter=1000)
STACKING_TASKS = ['ft', 'hm']

# --- Robustness Check ---
N_REPETITIONS = 1000 # Consider increasing this (e.g., 30, 50, 100) for more stable results

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
SAVE_INDIVIDUAL_RUN_RESULTS = False
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
# Plotting Functions (Unchanged from previous version)
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
    if len(df_plot) <= N_REPETITIONS * len(valid_tasks_to_plot) * 1.5:
        sns.stripplot(x='Task', y=metric_key, data=df_plot, color=".25", size=5, alpha=0.6, order=order)

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
            # print(f"Warning: Insufficient ROC data or AUCs for '{task_label}'. Skipping its curve.") # Can be verbose
            continue

        tprs_interp = []
        for run_data in task_roc_runs:
             if run_data and len(run_data) == 2 and run_data[0] is not None and run_data[1] is not None:
                 fpr, tpr = run_data; fpr, tpr = np.array(fpr), np.array(tpr)
                 if len(fpr) < 2 or len(tpr) < 2: continue
                 tpr_interp = np.interp(base_fpr, fpr, tpr); tpr_interp[0] = 0.0
                 tprs_interp.append(tpr_interp)

        if len(tprs_interp) != len(task_aucs_list):
             # print(f"Warning: Mismatch ROC curves ({len(tprs_interp)}) vs AUCs ({len(task_aucs_list)}) for {task_label}. Using {len(tprs_interp)} curves.") # Can be verbose
             task_aucs_list = task_aucs_list[:len(tprs_interp)]
             if not task_aucs_list: continue

        if not tprs_interp:
            # print(f"Warning: No valid interpolated TPRs for '{task_label}'. Skipping curve.") # Can be verbose
            continue

        mean_tprs = np.mean(tprs_interp, axis=0); std_tprs = np.std(tprs_interp, axis=0)
        tprs_upper = np.minimum(mean_tprs + std_tprs, 1); tprs_lower = np.maximum(mean_tprs - std_tprs, 0)
        mean_auc = np.mean(task_aucs_list); std_auc = np.std(task_aucs_list)

        label = f'{task_label} (AUC = {mean_auc:.2f} ± {std_auc:.2f})'
        plt.plot(base_fpr, mean_tprs, label=label, color=colors[i], lw=2.5)
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, color=colors[i], alpha=0.2)
        plot_successful = True

    if not plot_successful:
         print("Warning: No ROC curves were plotted.")
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

    coeffs_df_sorted = coeffs_df.reindex(coeffs_df['Mean_Coefficient'].abs().sort_values(ascending=False, na_position='last').index)
    plot_df = coeffs_df_sorted.head(top_n).copy()
    if plot_df.empty:
        # print(f"Warning: No coefficients left after filtering/sorting for '{model_label}'. Skipping plot '{filename}'.") # Verbose
        return

    plot_df = plot_df.iloc[::-1] # Highest absolute value at top
    colors = ['#d62728' if c < 0 else '#1f77b4' for c in plot_df['Mean_Coefficient']]

    plt.figure(figsize=(10, max(5, len(plot_df) * 0.35)))
    plt.barh(
        plot_df.index, plot_df['Mean_Coefficient'],
        xerr=plot_df['Std_Coefficient'].fillna(0), color=colors,
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
    plt.yticks(fontsize=10)
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

    if resampling_strategy == 'smote' and SMOTE and IMBLEARN_AVAILABLE:
        pipeline_steps.append(('resampler', SMOTE(random_state=random_state)))
        CurrentPipeline = ImbPipeline

    # <<< ROBUST SCALER: Conditional Scaler Choice >>>
    if use_robust_scaler:
        pipeline_steps.append(('scaler', RobustScaler()))
        scaler_name = 'RobustScaler'
    else:
        pipeline_steps.append(('scaler', StandardScaler()))
    # <<< END SCALER CHOICE >>>

    if model_name == 'logistic':
        classifier = LogisticRegression(random_state=random_state, class_weight='balanced', max_iter=2000)
        # Define hyperparameter search space (remains the same regardless of scaler)
        param_dist = {'classifier__C': loguniform(1e-4, 1e4),
                      'classifier__solver': ['liblinear', 'saga'],
                      'classifier__penalty': ['l1', 'l2']}
    else:
        raise ValueError(f"Invalid base model_name '{model_name}'")

    pipeline_steps.append(('classifier', classifier))
    pipeline = CurrentPipeline(pipeline_steps)

    print(f"        Pipeline built using: Imputer='{imputation_strategy}', Resampler='{resampling_strategy}', Scaler='{scaler_name}', Model='{model_name}'") # Info print
    return pipeline, param_dist

# -----------------------------------------------------
# Function to Evaluate Predictions (Unchanged from previous version)
# -----------------------------------------------------
def evaluate_predictions(y_true, y_pred, y_pred_proba=None):
    """Calculates standard classification metrics."""
    metrics = {}
    metrics['roc_auc'] = np.nan
    roc_curve_data = None # Initialize
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) > 1:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                roc_curve_data = (fpr.tolist(), tpr.tolist())
            else: pass # print("Warning: Only one class present in y_true. ROC AUC is not defined.") # Verbose
        except ValueError as e:
            # print(f"Warning: Could not calculate ROC AUC score: {e}") # Verbose
            roc_curve_data = None
    else: roc_curve_data = None

    try: metrics['accuracy'] = accuracy_score(y_true, y_pred)
    except ValueError: metrics['accuracy'] = np.nan
    try: metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    except ValueError: metrics['f1_macro'] = np.nan
    try: metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    except ValueError: metrics['precision_macro'] = np.nan
    try: metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    except ValueError: metrics['recall_macro'] = np.nan
    try: metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    except ValueError: metrics['confusion_matrix'] = []

    return metrics, roc_curve_data

# -----------------------------------------------------
# Main Execution Block
# -----------------------------------------------------
if __name__ == '__main__':
    start_time_script = time()
    print("--- Starting DatScan Prediction Script (Task-Specific + Stacking) ---")
    print(f"Number of repetitions: {N_REPETITIONS}")
    if ENABLE_STACKING: print(f"Model Stacking Enabled using tasks: {STACKING_TASKS}")
    # <<< ROBUST SCALER: Report which scaler is used >>>
    print(f"Using scaler: {'RobustScaler' if USE_ROBUST_SCALER else 'StandardScaler'}")

    # --- Create Output Folders ---
    try:
        os.makedirs(DATA_OUTPUT_FOLDER, exist_ok=True)
        if GENERATE_PLOTS: os.makedirs(PLOT_OUTPUT_FOLDER, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directories: {e}. Check permissions or path."); exit()

    # --- Print Folder Paths ---
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Data Output folder: {DATA_OUTPUT_FOLDER}")
    if GENERATE_PLOTS: print(f"Plot Output folder: {PLOT_OUTPUT_FOLDER}")

    # --- 1. Load Data (Once) ---
    input_file_path = os.path.join(INPUT_FOLDER, INPUT_CSV_NAME)
    print(f"Loading data from: {input_file_path}")
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at {input_file_path}"); exit()
    try:
        try: df = pd.read_csv(input_file_path, sep=';', decimal='.')
        except Exception: df = pd.read_csv(input_file_path, sep=',', decimal='.')
        print(f"Data loaded successfully. Shape: {df.shape}")
        if TARGET_Z_SCORE_COL not in df.columns: raise ValueError(f"Target column '{TARGET_Z_SCORE_COL}' not found.")
    except Exception as e: print(f"Error loading data: {e}"); exit()

    # --- 2. Prepare Target Variable (Once) ---
    data_full = df.copy()
    data_full[TARGET_Z_SCORE_COL] = pd.to_numeric(data_full[TARGET_Z_SCORE_COL].astype(str).str.replace(',', '.'), errors='coerce')
    data_full.dropna(subset=[TARGET_Z_SCORE_COL], inplace=True)
    if data_full.empty: print("Error: No data after dropping missing target."); exit()
    target_col_name = 'DatScan_Status'
    data_full[target_col_name] = (data_full[TARGET_Z_SCORE_COL] <= ABNORMALITY_THRESHOLD).astype(int)
    y_full = data_full[target_col_name]
    print(f"Target variable '{target_col_name}' distribution: {y_full.value_counts().to_dict()}")
    if len(y_full.unique()) < 2:
        print("Error: Target variable has only one class. Cannot perform classification."); exit()

    # --- 3. Feature Preparation (Once) ---
    all_feature_cols = []
    task_features_map = {}
    for task_prefix in TASKS_TO_RUN_SEPARATELY:
        task_cols = [col for base in BASE_KINEMATIC_COLS if (col := f"{task_prefix}_{base}") in data_full.columns]
        if task_cols:
            print(f"Found {len(task_cols)} features for task '{task_prefix}'.")
            task_features_map[task_prefix] = task_cols
            all_feature_cols.extend(task_cols)
        else: print(f"Warning: No features found for task '{task_prefix}'.")

    if not all_feature_cols: print("Error: No kinematic features found. Exiting."); exit()

    X_full = data_full[all_feature_cols].copy()
    for col in all_feature_cols:
        X_full[col] = pd.to_numeric(X_full[col].astype(str).str.replace(',', '.'), errors='coerce')

    if IMPUTATION_STRATEGY is None:
        rows_before_drop = len(X_full)
        valid_indices = X_full.dropna().index
        if len(valid_indices) < len(X_full):
             X_full = X_full.loc[valid_indices]
             y_full = y_full.loc[valid_indices]
             print(f"Dropped {rows_before_drop - len(X_full)} rows with missing features (no imputation).")
        if X_full.empty: print("Error: No data left after dropping rows with NaNs."); exit()

    if len(X_full) < 20: print(f"Error: Insufficient data (N={len(X_full)}) after preparation. Exiting."); exit()
    print(f"Final data shape for modeling: X={X_full.shape}, y={len(y_full)}")

    # --- 4. Robustness Check Loop with Stacking ---
    all_runs_metrics = collections.defaultdict(lambda: collections.defaultdict(list))
    all_runs_base_importances = collections.defaultdict(lambda: collections.defaultdict(list))
    all_runs_roc_data = collections.defaultdict(lambda: collections.defaultdict(list))
    all_runs_meta_importances = collections.defaultdict(list)

    print(f"\n--- Starting {N_REPETITIONS} Repetitions (Base Models + Stacking) ---")
    for i in range(N_REPETITIONS):
        current_random_state = BASE_RANDOM_STATE + i
        if (i+1) % 5 == 0 or i == 0 or N_REPETITIONS <= 10:
             print(f"\n  Repetition {i+1}/{N_REPETITIONS} (Seed: {current_random_state})...")

        # 4a. Split Data
        try:
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X_full, y_full, test_size=TEST_SET_SIZE,
                random_state=current_random_state, stratify=y_full
            )
            if len(y_train_val.unique()) < 2 or len(y_test.unique()) < 2:
                 print(f"   Warning: Rep {i+1} split resulted in one class in train/test. Skipping run.")
                 for task_prefix in TASKS_TO_RUN_SEPARATELY: all_runs_metrics[task_prefix][MODEL_NAME].append({})
                 if ENABLE_STACKING: all_runs_metrics['stacked'][META_MODEL_NAME].append({})
                 continue
        except ValueError as e:
             print(f"   Error splitting data in rep {i+1}: {e}. Skipping run.")
             for task_prefix in TASKS_TO_RUN_SEPARATELY: all_runs_metrics[task_prefix][MODEL_NAME].append({})
             if ENABLE_STACKING: all_runs_metrics['stacked'][META_MODEL_NAME].append({})
             continue

        # 4b. Train Base Models, Generate Predictions
        oof_preds = {}
        test_preds = {}
        base_model_pipelines = {}

        print(f"    Training Base Models ({', '.join(TASKS_TO_RUN_SEPARATELY)})...")
        base_model_training_successful = False
        for task_prefix in TASKS_TO_RUN_SEPARATELY:
            task_feature_cols = task_features_map.get(task_prefix)
            if not task_feature_cols: continue

            X_train_val_task = X_train_val[task_feature_cols]
            X_test_task = X_test[task_feature_cols]

            # <<< ROBUST SCALER: Pass flag to build_base_pipeline >>>
            try:
                pipeline, param_dist = build_base_pipeline(
                    MODEL_NAME, IMPUTATION_STRATEGY, RESAMPLING_STRATEGY, current_random_state,
                    use_robust_scaler=USE_ROBUST_SCALER
                )
            except ValueError as e_build:
                print(f"      [{task_prefix}] Error building pipeline: {e_build}. Skipping task.")
                all_runs_metrics[task_prefix][MODEL_NAME].append({})
                continue

            best_pipeline = clone(pipeline)
            tuning_best_score = None
            best_params = None
            fit_successful = False

            if ENABLE_TUNING:
                cv_tune = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=current_random_state)
                random_search = RandomizedSearchCV(
                    estimator=pipeline, param_distributions=param_dist, n_iter=N_ITER_RANDOM_SEARCH,
                    scoring=TUNING_SCORING_METRIC, cv=cv_tune, random_state=current_random_state,
                    n_jobs=-1, refit=True, error_score=0.0 # Consider 'raise' for debug
                )
                try:
                    random_search.fit(X_train_val_task, y_train_val)
                    best_params = random_search.best_params_
                    best_pipeline = random_search.best_estimator_
                    tuning_best_score = random_search.best_score_
                    fit_successful = True
                except Exception as e_tune:
                    print(f"      [{task_prefix}] Tuning Error: {e_tune}. Trying default fit.")
                    try: best_pipeline.fit(X_train_val_task, y_train_val); fit_successful = True
                    except Exception as e_fit_fb: print(f"      [{task_prefix}] Fallback Fit Error: {e_fit_fb}")
            else: # No tuning
                 try: best_pipeline.fit(X_train_val_task, y_train_val); fit_successful = True
                 except Exception as e_fit_def: print(f"      [{task_prefix}] Default Fit Error: {e_fit_def}")

            if fit_successful:
                base_model_pipelines[task_prefix] = best_pipeline
                base_model_training_successful = True

                # Generate OOF predictions
                try:
                    cv_predict_fold = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=current_random_state)
                    oof_pred_proba = cross_val_predict(best_pipeline, X_train_val_task, y_train_val, cv=cv_predict_fold, method='predict_proba', n_jobs=-1)[:, 1]
                    oof_preds[task_prefix] = oof_pred_proba
                except Exception as e_oof:
                     print(f"      [{task_prefix}] OOF Prediction Error: {e_oof}"); oof_preds[task_prefix] = None

                # Generate Test predictions and Evaluate
                try:
                    test_pred_proba = best_pipeline.predict_proba(X_test_task)[:, 1]
                    test_pred_labels = best_pipeline.predict(X_test_task)
                    test_preds[task_prefix] = test_pred_proba

                    base_metrics, base_roc_data = evaluate_predictions(y_test, test_pred_labels, test_pred_proba)
                    base_metrics['best_cv_score'] = tuning_best_score
                    base_metrics['best_params'] = best_params
                    all_runs_metrics[task_prefix][MODEL_NAME].append(convert_numpy_types(base_metrics))
                    if base_roc_data: all_runs_roc_data[task_prefix][MODEL_NAME].append(base_roc_data)

                    # Extract coefficients
                    try:
                        final_classifier_step = best_pipeline.steps[-1][1]
                        if isinstance(final_classifier_step, LogisticRegression):
                             coeffs = final_classifier_step.coef_[0]
                             imp_series = pd.Series(coeffs, index=task_feature_cols)
                             all_runs_base_importances[task_prefix][MODEL_NAME].append(imp_series)
                    except Exception as e_imp: pass # print(f"[{task_prefix}] Coeff extract warning: {e_imp}") # Verbose

                except Exception as e_test:
                     print(f"      [{task_prefix}] Test Predict/Eval Error: {e_test}")
                     test_preds[task_prefix] = None; all_runs_metrics[task_prefix][MODEL_NAME].append({})
            else:
                 print(f"      [{task_prefix}] Fit failed. Skipping predictions.")
                 oof_preds[task_prefix] = None; test_preds[task_prefix] = None
                 all_runs_metrics[task_prefix][MODEL_NAME].append({})

        # 4c. Train and Evaluate Meta-Model
        if ENABLE_STACKING and base_model_training_successful:
            print(f"    Training Stacked Model ({META_MODEL_NAME})...")
            meta_train_features_list = []
            meta_test_features_list = []
            valid_stacking_tasks = []

            for task in STACKING_TASKS:
                 if task in oof_preds and oof_preds[task] is not None and \
                    task in test_preds and test_preds[task] is not None:
                      if len(oof_preds[task]) == len(y_train_val):
                           meta_train_features_list.append(pd.Series(oof_preds[task], index=y_train_val.index, name=f"{task}_pred"))
                           meta_test_features_list.append(pd.Series(test_preds[task], index=y_test.index, name=f"{task}_pred"))
                           valid_stacking_tasks.append(task)
                      else: pass # print(f"Warn: Mismatch OOF length for {task}.") # Verbose
                 # else: Optional warning if expected preds are missing

            if len(valid_stacking_tasks) < 1:
                 print("      Error: Not enough valid base preds for meta-model. Skipping stacking."); all_runs_metrics['stacked'][META_MODEL_NAME].append({})
            else:
                 meta_train_features_df = pd.concat(meta_train_features_list, axis=1)
                 meta_test_features_df = pd.concat(meta_test_features_list, axis=1)
                 meta_model = clone(META_CLASSIFIER)
                 try:
                     meta_model.fit(meta_train_features_df, y_train_val)
                     meta_y_pred_test = meta_model.predict(meta_test_features_df)
                     meta_y_pred_proba_test = meta_model.predict_proba(meta_test_features_df)[:, 1]
                     stacked_metrics, stacked_roc_data = evaluate_predictions(y_test, meta_y_pred_test, meta_y_pred_proba_test)

                     all_runs_metrics['stacked'][META_MODEL_NAME].append(convert_numpy_types(stacked_metrics))
                     if stacked_roc_data: all_runs_roc_data['stacked'][META_MODEL_NAME].append(stacked_roc_data)
                     if hasattr(meta_model, 'coef_'):
                          meta_coeffs = pd.Series(meta_model.coef_[0], index=meta_train_features_df.columns)
                          all_runs_meta_importances[META_MODEL_NAME].append(meta_coeffs)
                 except Exception as e_meta:
                      print(f"      Meta-Model Error: {e_meta}"); all_runs_metrics['stacked'][META_MODEL_NAME].append({})
        elif ENABLE_STACKING: # Implies base_model_training_successful was False
             print("    Skipping Stacking: No base models trained successfully."); all_runs_metrics['stacked'][META_MODEL_NAME].append({})

    print("\n--- All Repetitions Finished ---")

    # --- 5. Aggregate and Summarize Results ---
    print("\n===== Aggregated Performance Summary (Mean +/- StdDev) =====")
    overall_summary_stats = []
    metric_keys = ['accuracy', 'roc_auc', 'f1_macro', 'precision_macro', 'recall_macro']
    models_to_summarize = list(all_runs_metrics.keys())

    for model_key in models_to_summarize:
        actual_model_name = list(all_runs_metrics[model_key].keys())[0] if all_runs_metrics[model_key] else None
        if not actual_model_name: print(f"\n--- Model/Task: {model_key.upper()} ---\nNo results found."); continue

        print(f"\n--- Model/Task: {model_key.upper()} ({actual_model_name}) ---")
        model_metrics_list = all_runs_metrics[model_key].get(actual_model_name, [])
        valid_model_metrics_list = [m for m in model_metrics_list if isinstance(m, dict) and m]
        if not valid_model_metrics_list: print("No successful/valid runs recorded."); continue

        metrics_df = pd.DataFrame(valid_model_metrics_list)
        for mkey in metric_keys:
            if mkey not in metrics_df.columns: metrics_df[mkey] = np.nan
        metrics_df = metrics_df.reindex(columns=metric_keys + list(metrics_df.columns.difference(metric_keys)))

        if metrics_df[metric_keys].isnull().all().all(): print(f"No valid metrics found."); continue

        means = metrics_df[metric_keys].mean(skipna=True); stds = metrics_df[metric_keys].std(skipna=True)
        n_valid_runs = metrics_df[metric_keys[0]].notna().sum()

        task_summary = {'Model_Key': model_key, 'N_Valid_Runs': int(n_valid_runs) } # Ensure int
        for key in metric_keys: task_summary[f'{key}_mean'] = means.get(key, np.nan); task_summary[f'{key}_std'] = stds.get(key, np.nan)
        overall_summary_stats.append(task_summary)

        summary_df_task = pd.DataFrame([task_summary])
        print(f"Mean +/- Std Dev across {n_valid_runs} VALID runs:")
        for key in metric_keys: summary_df_task[f'{key}'] = summary_df_task.apply(lambda r: f"{r[f'{key}_mean']:.4f} +/- {r[f'{key}_std']:.4f}" if pd.notna(r[f'{key}_mean']) else "N/A", axis=1)
        display_cols = ['Model_Key', 'N_Valid_Runs'] + metric_keys
        print(summary_df_task[display_cols].to_string(index=False))

    if SAVE_AGGREGATED_SUMMARY and overall_summary_stats:
        summary_filename = os.path.join(DATA_OUTPUT_FOLDER, "prediction_task_stacking_comparison_summary.csv")
        try:
             summary_df_final = pd.DataFrame(overall_summary_stats)
             summary_df_final.to_csv(summary_filename, index=False, sep=';', decimal='.', float_format='%.6f')
             print(f"\nOverall summary saved to: {summary_filename}")
        except Exception as e: print(f"Error saving overall summary: {e}")

    # --- 6. Aggregate and Save Coefficients ---
    aggregated_coeffs_dfs = {}

    # Base Model Coefficients
    if SAVE_AGGREGATED_IMPORTANCES:
        print("\n--- Aggregating Base Model Coefficients (Per Task) ---")
        for task_prefix in TASKS_TO_RUN_SEPARATELY:
             coeff_lists = all_runs_base_importances[task_prefix].get(MODEL_NAME, [])
             valid_coeffs = [s for s in coeff_lists if isinstance(s, pd.Series) and not s.empty]
             if valid_coeffs:
                 n_valid = len(valid_coeffs)
                 print(f"Aggregating for base task: {task_prefix.upper()} ({n_valid} valid runs)")
                 try:
                     coeff_df = pd.concat(valid_coeffs, axis=1, join='outer')
                     agg_coeff = pd.DataFrame({
                         'Mean_Coefficient': coeff_df.mean(axis=1, skipna=True),
                         'Std_Coefficient': coeff_df.std(axis=1, skipna=True),
                         'N_Valid_Runs': coeff_df.notna().sum(axis=1).astype(int)
                     })
                     agg_coeff = agg_coeff.reindex(agg_coeff['Mean_Coefficient'].abs().sort_values(ascending=False, na_position='last').index)
                     aggregated_coeffs_dfs[task_prefix] = agg_coeff
                     fname = os.path.join(DATA_OUTPUT_FOLDER, f"prediction_{MODEL_NAME}_{task_prefix}_aggregated_coefficients.csv")
                     agg_coeff.to_csv(fname, sep=';', decimal='.', index_label='Feature', float_format='%.6f')
                     print(f"Aggregated coefficients for task {task_prefix} saved to: {fname}")
                 except Exception as e: print(f"Error aggregating coefficients for task {task_prefix}: {e}")
             else: print(f"No valid coefficient data for task {task_prefix}.")

    # Meta Model Coefficients
    if ENABLE_STACKING and SAVE_META_MODEL_COEFFICIENTS:
        print("\n--- Aggregating Meta-Model Coefficients ---")
        meta_coeff_lists = all_runs_meta_importances.get(META_MODEL_NAME, [])
        valid_meta_coeffs = [s for s in meta_coeff_lists if isinstance(s, pd.Series) and not s.empty]
        if valid_meta_coeffs:
            n_valid = len(valid_meta_coeffs)
            print(f"Aggregating for meta-model: {META_MODEL_NAME} ({n_valid} valid runs)")
            try:
                meta_coeff_df = pd.concat(valid_meta_coeffs, axis=1, join='outer')
                agg_meta_coeff = pd.DataFrame({
                    'Mean_Coefficient': meta_coeff_df.mean(axis=1, skipna=True),
                    'Std_Coefficient': meta_coeff_df.std(axis=1, skipna=True),
                    'N_Valid_Runs': meta_coeff_df.notna().sum(axis=1).astype(int)
                })
                agg_meta_coeff = agg_meta_coeff.reindex(agg_meta_coeff['Mean_Coefficient'].abs().sort_values(ascending=False, na_position='last').index)
                aggregated_coeffs_dfs['stacked'] = agg_meta_coeff
                fname = os.path.join(DATA_OUTPUT_FOLDER, f"prediction_{META_MODEL_NAME}_aggregated_coefficients.csv")
                agg_meta_coeff.to_csv(fname, sep=';', decimal='.', index_label='Base_Model_Prediction', float_format='%.6f')
                print(f"Aggregated meta-model coefficients saved to: {fname}")
            except Exception as e: print(f"Error aggregating meta-model coefficients: {e}")
        else: print(f"No valid coefficient data for meta-model {META_MODEL_NAME}.")

    # --- 7. Generate Plots ---
    if GENERATE_PLOTS:
        print("\n--- Generating Plots ---")
        sns.set_theme(style="whitegrid")
        plot_keys = TASKS_TO_RUN_SEPARATELY[:]
        if ENABLE_STACKING and 'stacked' in all_runs_metrics:
            if all_runs_metrics['stacked'] and any(all_runs_metrics['stacked'].values()): plot_keys.append('stacked')

        # Metric Distributions
        for metric in ['roc_auc', 'accuracy', 'f1_macro']:
             plot_title = f'Test {metric.replace("_"," ").title()} Distribution ({N_REPETITIONS} Runs)'
             fname = os.path.join(PLOT_OUTPUT_FOLDER, f"plot_metric_distribution_{metric}_comparison.png")
             plot_metric_distributions(all_runs_metrics, plot_keys, metric, plot_title, fname)

        # Aggregated ROC Curves
        if any(all_runs_roc_data.values()):
             plot_title_roc = f'Average ROC Curves Comparison ({N_REPETITIONS} Runs)'
             fname = os.path.join(PLOT_OUTPUT_FOLDER, "plot_aggregated_roc_curves_comparison.png")
             plot_aggregated_roc_curves(all_runs_roc_data, all_runs_metrics, plot_keys, plot_title_roc, fname)
        else: print("Skipping ROC curve plot: No ROC data.")

        # Aggregated Coefficients
        for model_key, agg_coeff_df in aggregated_coeffs_dfs.items():
            if agg_coeff_df is not None and not agg_coeff_df.empty:
                n_valid_runs_list = agg_coeff_df['N_Valid_Runs']
                if not n_valid_runs_list.empty: n_runs_info = f"{int(n_valid_runs_list.min())}-{int(n_valid_runs_list.max())}"
                else: n_runs_info = "N/A"

                if model_key == 'stacked':
                    model_label = f"Stacked Model ({META_MODEL_NAME})"
                    title = f'Aggregated Meta-Model Coefficients\n({n_runs_info} Valid Runs)'
                    fname = os.path.join(PLOT_OUTPUT_FOLDER, f"plot_aggregated_coefficients_{META_MODEL_NAME}.png")
                    top_n = len(agg_coeff_df)
                else:
                    model_label = f"Task: {model_key.upper()} ({MODEL_NAME})"
                    title = f'Top {PLOT_TOP_N_COEFFICIENTS} Aggregated Coefficients\n{model_label} ({n_runs_info} Valid Runs)'
                    fname = os.path.join(PLOT_OUTPUT_FOLDER, f"plot_aggregated_coefficients_{MODEL_NAME}_{model_key}.png")
                    top_n = PLOT_TOP_N_COEFFICIENTS

                plot_aggregated_coefficients(agg_coeff_df, model_label, top_n, title, fname)
            # else: print(f"Skipping coeff plot for '{model_key}': No aggregated data.") # Verbose

    print(f"\n--- Script Finished ({time() - start_time_script:.1f}s Total) ---")