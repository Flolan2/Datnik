#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone script to predict DatScan status based on kinematic features,
running Logistic Regression models SEPARATELY for Finger Tapping (ft) and
Hand Movement (hm) tasks, including robustness checks and plotting.

Includes tuning, optional resampling, evaluates models, reports aggregated
performance (mean +/- std dev) per task, and generates plots for metric
distributions, average ROC curves, and aggregated coefficients.
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
    train_test_split, StratifiedKFold, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score,
    f1_score, precision_score, recall_score, roc_curve # Added roc_curve
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Imbalanced-learn imports (optional)
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    # print("Warning: 'imbalanced-learn' not found. Resampling options unavailable.") # Less verbose
    ImbPipeline = Pipeline; SMOTE = None; IMBLEARN_AVAILABLE = False

# Suppress specific warnings (e.g., ConvergenceWarning) for cleaner output if desired
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn') # Often warns about dual coef

# -----------------------------------------------------
# Configuration Parameters - MODIFY AS NEEDED
# -----------------------------------------------------
# --- Input Data ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Try to find parent directory if script is in a 'Code' subfolder
script_parent_dir = os.path.dirname(SCRIPT_DIR)
if os.path.basename(SCRIPT_DIR).lower() in ['code', 'scripts']:
    base_dir = script_parent_dir
else:
    base_dir = SCRIPT_DIR

INPUT_FOLDER = os.path.join(base_dir, "Input")
OUTPUT_FOLDER_BASE = os.path.join(base_dir, "Output")
DATA_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER_BASE, "Data")
PLOT_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER_BASE, "Plots") # <<< Plot Folder

INPUT_CSV_NAME = "merged_summary.csv"

# --- Prediction Target ---
TARGET_IMAGING_BASE = "Contralateral_Striatum"
TARGET_Z_SCORE_COL = f"{TARGET_IMAGING_BASE}_Z"
ABNORMALITY_THRESHOLD = -1.96

# --- Features ---
BASE_KINEMATIC_COLS = [ # Base names used to find task-specific features
    "meanamplitude", "stdamplitude", "meanspeed", "stdspeed", "meanrmsvelocity",
    "stdrmsvelocity", "meanopeningspeed", "stdopeningspeed", "meanclosingspeed",
    "stdclosingspeed", "meancycleduration", "stdcycleduration", "rangecycleduration",
    "rate", "amplitudedecay", "velocitydecay", "ratedecay", "cvamplitude",
    "cvcycleduration", "cvspeed", "cvrmsvelocity", "cvopeningspeed", "cvclosingspeed"
]
# Tasks to run models for separately
TASKS_TO_RUN_SEPARATELY = ['ft', 'hm'] # <<< Defines the outer loop

# --- Model to Test ---
MODEL_NAME = 'logistic' # Focus on Logistic Regression

# --- Common Evaluation Settings ---
TEST_SET_SIZE = 0.25
N_SPLITS_CV = 5
IMPUTATION_STRATEGY = 'median'
BASE_RANDOM_STATE = 42

# --- Robustness Check ---
N_REPETITIONS = 10 # Number of repetitions per task

# --- Resampling Strategy ---
RESAMPLING_STRATEGY = None # 'smote' or None
if RESAMPLING_STRATEGY == 'smote' and not IMBLEARN_AVAILABLE:
    print(f"Warning: RESAMPLING_STRATEGY='smote' but imblearn not available. Disabling.")
    RESAMPLING_STRATEGY = None

# --- Hyperparameter Tuning ---
ENABLE_TUNING = True
N_ITER_RANDOM_SEARCH = 50
TUNING_SCORING_METRIC = 'roc_auc'

# --- Output Options ---
SAVE_INDIVIDUAL_RUN_RESULTS = False
SAVE_AGGREGATED_SUMMARY = True
SAVE_AGGREGATED_IMPORTANCES = True # Saves aggregated coefficients per task
GENERATE_PLOTS = True             # <<< Option to generate plots
PLOT_TOP_N_COEFFICIENTS = 15      # <<< How many top coefficients to plot

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
        elif np.isinf(obj): return None # Or "Infinity" string
        else: return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.void)):
        return None
    return obj

# -----------------------------------------------------
# Plotting Functions
# -----------------------------------------------------

def plot_metric_distributions(metrics_dict, tasks, metric_key, title, filename):
    """Generates box plots comparing a metric across tasks."""
    plot_data = []
    for task in tasks:
        # Retrieve list of metric dicts for this task, handle if model failed
        task_metrics = metrics_dict.get(task, {}).get(MODEL_NAME, [])
        for run_metric_dict in task_metrics:
            value = run_metric_dict.get(metric_key, np.nan) # Get specific metric
            if pd.notna(value):
                 plot_data.append({'Task': task.upper(), metric_key: value})

    if not plot_data:
        print(f"Warning: No data to plot for metric '{metric_key}'. Skipping plot '{filename}'.")
        return

    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(6, 5))
    sns.boxplot(x='Task', y=metric_key, data=df_plot, palette='viridis', width=0.5)
    # Optional: Add swarmplot/stripplot for individual points if N_REPETITIONS is small
    if len(df_plot) <= 50: # Adjust threshold as needed
        sns.stripplot(x='Task', y=metric_key, data=df_plot, color=".25", size=5, alpha=0.6)
    plt.title(title, fontsize=14, weight='bold')
    plt.xlabel("Kinematic Task", fontsize=12)
    plt.ylabel(metric_key.replace('_', ' ').title(), fontsize=12)
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


def plot_aggregated_roc_curves(roc_data_dict, auc_dict, tasks, title, filename):
    """Plots average ROC curves with variability for multiple tasks."""
    plt.style.use('seaborn-v0_8-whitegrid') # Use a clean seaborn style
    plt.figure(figsize=(7, 7))
    base_fpr = np.linspace(0, 1, 101) # Common FPR base for interpolation
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(tasks))) # Use viridis colormap

    plot_successful = False
    for i, task in enumerate(tasks):
        task_roc_runs = roc_data_dict.get(task, {}).get(MODEL_NAME, [])
        # Get list of AUCs for this task/model
        task_aucs_list = [m.get('roc_auc') for m in auc_dict.get(task, {}).get(MODEL_NAME, []) if pd.notna(m.get('roc_auc'))]

        if not task_roc_runs or not task_aucs_list:
            print(f"Warning: Insufficient ROC data or AUCs for task '{task}'. Skipping its curve.")
            continue

        tprs_interp = []
        valid_aucs = [] # Keep track of AUCs for valid ROC curves
        for run_data in task_roc_runs:
             # run_data should be a tuple (fpr, tpr)
             if run_data and len(run_data) == 2 and run_data[0] is not None and run_data[1] is not None:
                 fpr, tpr = run_data
                 # Ensure fpr/tpr are numpy arrays for interpolation
                 fpr, tpr = np.array(fpr), np.array(tpr)
                 # Skip if ROC curve is degenerate (e.g., only one point)
                 if len(fpr) < 2 or len(tpr) < 2: continue
                 tpr_interp = np.interp(base_fpr, fpr, tpr)
                 tpr_interp[0] = 0.0 # Ensure start at 0
                 tprs_interp.append(tpr_interp)
                 # Find corresponding AUC - needs robust linking or recalculation if possible
                 # For simplicity, we use pre-calculated AUCs assuming order is preserved
                 # A more robust way would be to recalculate AUC from fpr, tpr if needed

        # Use the pre-calculated valid AUCs stored earlier
        if len(tprs_interp) != len(task_aucs_list):
            print(f"Warning: Mismatch between number of valid ROC curves ({len(tprs_interp)}) and AUCs ({len(task_aucs_list)}) for task {task}. Using available AUCs.")
            # Fallback: use only AUCs that correspond to successful interpolations if possible (complex linkage)
            # For now, proceed with the collected AUC list, but be aware of potential mismatch if runs failed differently
            if not task_aucs_list: continue # Cannot calculate mean AUC

        if not tprs_interp:
            print(f"Warning: No valid interpolated TPRs for task '{task}'. Skipping curve.")
            continue

        mean_tprs = np.mean(tprs_interp, axis=0)
        std_tprs = np.std(tprs_interp, axis=0)
        tprs_upper = np.minimum(mean_tprs + std_tprs, 1) # Clip at 1
        tprs_lower = np.maximum(mean_tprs - std_tprs, 0) # Clip at 0

        mean_auc = np.mean(task_aucs_list)
        std_auc = np.std(task_aucs_list)

        label = f'{task.upper()} (AUC = {mean_auc:.2f} ± {std_auc:.2f})'
        plt.plot(base_fpr, mean_tprs, label=label, color=colors[i], lw=2.5)
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, color=colors[i], alpha=0.2) # Removed redundant label
        plot_successful = True

    if not plot_successful:
         print("Warning: No ROC curves were plotted.")
         plt.close()
         return

    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title(title, fontsize=14, weight='bold')
    plt.legend(loc='lower right', fontsize=10, frameon=True, facecolor='white', framealpha=0.8)
    plt.grid(True, linestyle='--', alpha=0.6) # Ensure grid is visible
    plt.tight_layout()
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Aggregated ROC curve plot saved to: {filename}")
    except Exception as e:
        print(f"Error saving ROC curve plot {filename}: {e}")
    finally:
        plt.close()


def plot_aggregated_coefficients(coeffs_df, task_name, top_n, title, filename):
    """Plots aggregated coefficients with error bars for a single task."""
    plt.style.use('seaborn-v0_8-whitegrid')
    if coeffs_df is None or coeffs_df.empty or top_n <= 0:
        print(f"Warning: No coefficient data or non-positive top_n for task '{task_name}'. Skipping plot '{filename}'.")
        return

    # Ensure columns exist
    if not {'Mean_Coefficient', 'Std_Coefficient'}.issubset(coeffs_df.columns):
        print(f"Warning: Missing required columns in coefficients df for task '{task_name}'. Skipping plot '{filename}'.")
        return

    # Sort by absolute mean coefficient and take top N
    # Make sure sorting handles potential NaN values gracefully
    coeffs_df_sorted = coeffs_df.reindex(coeffs_df['Mean_Coefficient'].abs().sort_values(ascending=False, na_position='last').index)
    plot_df = coeffs_df_sorted.head(top_n).copy()

    if plot_df.empty:
        print(f"Warning: No coefficients left after filtering/sorting for task '{task_name}'. Skipping plot '{filename}'.")
        return

    # Sort for plotting (highest absolute value at top)
    plot_df = plot_df.iloc[::-1]

    # Create colors based on sign
    colors = ['#d62728' if c < 0 else '#1f77b4' for c in plot_df['Mean_Coefficient']] # Red for negative, Blue for positive

    plt.figure(figsize=(10, max(5, len(plot_df) * 0.35)))
    bars = plt.barh(
        plot_df.index,
        plot_df['Mean_Coefficient'],
        xerr=plot_df['Std_Coefficient'].fillna(0), # Fill NaN std dev with 0 for plotting
        color=colors,
        alpha=0.85,
        edgecolor='black',
        linewidth=0.7,
        capsize=4 # Add caps to error bars
    )
    plt.axvline(0, color='dimgrey', linestyle='--', linewidth=1)
    plt.xlabel('Mean Coefficient (Log-Odds Change)', fontsize=12)
    plt.ylabel('Kinematic Feature', fontsize=12)
    plt.title(title, fontsize=14, weight='bold')
    plt.grid(axis='x', linestyle=':', alpha=0.7)
    plt.yticks(fontsize=10)
    sns.despine(left=True, bottom=False) # Remove left spine, keep bottom
    plt.tight_layout()
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Aggregated coefficients plot for {task_name} saved to: {filename}")
    except Exception as e:
        print(f"Error saving coefficients plot for {task_name}: {filename}: {e}")
    finally:
        plt.close()


# -----------------------------------------------------
# Core Prediction & Tuning Function (Modified to return ROC data)
# -----------------------------------------------------
def tune_predict_single_model(
    model_name: str,
    X_train_val: pd.DataFrame, y_train_val: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    feature_names: list,
    n_splits_cv: int, imputation_strategy: str, resampling_strategy: str,
    enable_tuning: bool, n_iter_tuning: int, tuning_scoring: str, random_state: int
) -> tuple[dict | None, pd.Series | None, tuple | None]: # Return results dict, coeffs series, roc tuple (fpr, tpr)
    """
    Tunes (optionally), trains, and evaluates Logistic Regression FOR ONE RUN/TASK.
    Returns results dict, coefficients series, and ROC curve data (fpr, tpr).
    """
    # (Pipeline setup, tuning, training logic remains the same)
    pipeline_steps = []; CurrentPipeline = Pipeline
    if imputation_strategy in ['mean', 'median']: pipeline_steps.append(('imputer', SimpleImputer(strategy=imputation_strategy)))
    if resampling_strategy == 'smote' and SMOTE: pipeline_steps.append(('resampler', SMOTE(random_state=random_state))); CurrentPipeline = ImbPipeline
    pipeline_steps.append(('scaler', StandardScaler()))
    if model_name == 'logistic':
        classifier = LogisticRegression(random_state=random_state, class_weight='balanced', max_iter=2000)
        param_dist = {'classifier__C': loguniform(1e-4, 1e4), 'classifier__solver': ['liblinear', 'saga'], 'classifier__penalty': ['l1', 'l2']}
    else: print(f"Error: Invalid model_name '{model_name}'"); return None, None, None
    pipeline_steps.append(('classifier', classifier))
    pipeline = CurrentPipeline(pipeline_steps)
    best_params = None; best_pipeline = pipeline; tuning_best_score = None
    cv = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=random_state)
    if enable_tuning:
        start_time = time()
        random_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_dist, n_iter=n_iter_tuning, scoring=tuning_scoring, cv=cv, random_state=random_state, n_jobs=-1, refit=True, error_score=0.0)
        try:
            random_search.fit(X_train_val, y_train_val)
            best_params = random_search.best_params_; best_pipeline = random_search.best_estimator_; tuning_best_score = random_search.best_score_
        except Exception as e_tune:
            print(f"[{model_name}] Tuning Error: {e_tune}. Using default.")
            try: best_pipeline.fit(X_train_val, y_train_val)
            except Exception as e_fit_fb: print(f"[{model_name}] Fallback Fit Error: {e_fit_fb}"); return None, None, None
    else:
        try: best_pipeline.fit(X_train_val, y_train_val)
        except Exception as e_fit_def: print(f"[{model_name}] Default Fit Error: {e_fit_def}"); return None, None, None

    # Evaluation on Hold-out Test Set
    holdout_metrics = {}; roc_curve_data = None
    try:
        y_pred_test = best_pipeline.predict(X_test)
        holdout_metrics['roc_auc'] = np.nan
        if hasattr(best_pipeline, "predict_proba"):
            y_pred_proba_test = best_pipeline.predict_proba(X_test)[:, 1]
            try:
                 holdout_metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba_test)
                 fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test) # Calculate ROC points
                 roc_curve_data = (fpr.tolist(), tpr.tolist()) # Store as lists for JSON/aggregation
            except ValueError: pass # Keep ROC AUC as NaN if only one class
        holdout_metrics['accuracy'] = accuracy_score(y_test, y_pred_test)
        holdout_metrics['f1_macro'] = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
        holdout_metrics['precision_macro'] = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
        holdout_metrics['recall_macro'] = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
        holdout_metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred_test)
    except Exception as e_test:
        print(f"[{model_name}] Test Error: {e_test}")
        holdout_metrics = {k: np.nan for k in ['accuracy','roc_auc','f1_macro','precision_macro','recall_macro']}
        holdout_metrics['confusion_matrix'] = np.array([])

    # Feature Coefficients
    feature_coefficients_series = None
    try:
        final_classifier_step = best_pipeline.steps[-1][1]
        if isinstance(final_classifier_step, LogisticRegression):
             coefficients = final_classifier_step.coef_[0]
             feature_coefficients_series = pd.Series(coefficients, index=feature_names)
    except Exception: pass

    results_this_run = { 'model_name': model_name, 'best_cv_score': tuning_best_score, 'best_params': best_params, 'holdout_metrics': holdout_metrics }
    return results_this_run, feature_coefficients_series, roc_curve_data

# -----------------------------------------------------
# Main Execution Block
# -----------------------------------------------------
if __name__ == '__main__':
    start_time_script = time() # Record script start time
    print("--- Starting DatScan Task-Specific Logistic Regression Script (Robustness + Plotting) ---")
    print(f"Number of repetitions per task: {N_REPETITIONS}")

    # --- Create Output Folders ---
    os.makedirs(DATA_OUTPUT_FOLDER, exist_ok=True)
    if GENERATE_PLOTS: os.makedirs(PLOT_OUTPUT_FOLDER, exist_ok=True)
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Data Output folder: {DATA_OUTPUT_FOLDER}")
    if GENERATE_PLOTS: print(f"Plot Output folder: {PLOT_OUTPUT_FOLDER}")

    # --- 1. Load Data (Once) ---
    input_file_path = os.path.join(INPUT_FOLDER, INPUT_CSV_NAME)
    print(f"Loading data from: {input_file_path}")
    try:
        try: df = pd.read_csv(input_file_path, sep=';', decimal='.')
        except: df = pd.read_csv(input_file_path, sep=',', decimal='.')
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
    print(f"Prepared target variable. Class distribution: {y_full.value_counts().to_dict()}")


    # --- 3. Task Loop for Robustness Check ---
    all_tasks_runs_metrics = collections.defaultdict(lambda: collections.defaultdict(list))
    all_tasks_runs_importances = collections.defaultdict(lambda: collections.defaultdict(list))
    all_tasks_runs_roc_data = collections.defaultdict(lambda: collections.defaultdict(list))


    for task_prefix in TASKS_TO_RUN_SEPARATELY:
        print(f"\n===== Processing Task: {task_prefix.upper()} =====")

        # 3a. Select Features and Prepare X for this task
        task_feature_cols = [col for base in BASE_KINEMATIC_COLS if (col := f"{task_prefix}_{base}") in data_full.columns]
        if not task_feature_cols: print(f"No features for task '{task_prefix}'. Skipping."); continue
        print(f"Using {len(task_feature_cols)} features for task '{task_prefix}'.")
        X_task_full = data_full[task_feature_cols].copy()
        for col in task_feature_cols: X_task_full[col] = pd.to_numeric(X_task_full[col].astype(str).str.replace(',', '.'), errors='coerce')
        y_task_full = y_full.loc[X_task_full.index] # Align y with current X indices
        if IMPUTATION_STRATEGY is None:
            rows_before_drop = len(X_task_full)
            indices_to_keep = X_task_full.dropna().index
            X_task_full = X_task_full.loc[indices_to_keep]
            y_task_full = y_task_full.loc[indices_to_keep] # Keep y aligned after drop
            # print(f"[{task_prefix.upper()}] Dropped {rows_before_drop - len(X_task_full)} rows.")
        if len(X_task_full) < 20: print(f"Error: Insufficient data (N={len(X_task_full)}) for task '{task_prefix}'. Skipping."); continue
        # print(f"[{task_prefix.upper()}] Final data shape: {X_task_full.shape}")


        # 3c. Robustness Check Loop for this task
        print(f"--- Starting {N_REPETITIONS} Repetitions for Task: {task_prefix.upper()} ---")
        for i in range(N_REPETITIONS):
            current_random_state = BASE_RANDOM_STATE + i
            if (i+1) % 5 == 0 or i == 0 or N_REPETITIONS <= 10: print(f"  Repetition {i+1}/{N_REPETITIONS} (Seed: {current_random_state})...")

            # Split Data
            try:
                X_train_val, X_test, y_train_val, y_test = train_test_split(X_task_full, y_task_full, test_size=TEST_SET_SIZE, random_state=current_random_state, stratify=y_task_full)
            except ValueError as e:
                 print(f"   Error splitting data in repetition {i+1} for task {task_prefix}: {e}. Skipping run.")
                 all_tasks_runs_metrics[task_prefix][MODEL_NAME].append({})
                 continue

            # Run model
            model_results_this_run, model_feat_imp_series, roc_data = tune_predict_single_model(
                model_name=MODEL_NAME, X_train_val=X_train_val, y_train_val=y_train_val, X_test=X_test, y_test=y_test,
                feature_names=task_feature_cols, n_splits_cv=N_SPLITS_CV, imputation_strategy=IMPUTATION_STRATEGY,
                resampling_strategy=RESAMPLING_STRATEGY, enable_tuning=ENABLE_TUNING, n_iter_tuning=N_ITER_RANDOM_SEARCH,
                tuning_scoring=TUNING_SCORING_METRIC, random_state=current_random_state
            )

            # Collect results
            if model_results_this_run:
                metrics = model_results_this_run.get('holdout_metrics', {})
                all_tasks_runs_metrics[task_prefix][MODEL_NAME].append(metrics)
                if model_feat_imp_series is not None: all_tasks_runs_importances[task_prefix][MODEL_NAME].append(model_feat_imp_series)
                if roc_data is not None: all_tasks_runs_roc_data[task_prefix][MODEL_NAME].append(roc_data)
            else:
                print(f"!!!!! LogReg failed in repetition {i+1} for task {task_prefix} !!!!!")
                all_tasks_runs_metrics[task_prefix][MODEL_NAME].append({})

            # Save individual run results (optional)
            if SAVE_INDIVIDUAL_RUN_RESULTS:
                 # ... [unchanged saving logic] ...
                 pass # Keep code minimal here

    print("\n--- All Task Repetitions Finished ---")


    # --- 4. Aggregate and Summarize Results Across Runs PER TASK ---
    print("\n===== Aggregated Logistic Regression Performance Summary (Per Task) =====")
    overall_summary_stats = []
    # No longer need separate AUC dict, it's in the main metrics dict
    metric_keys = ['accuracy', 'roc_auc', 'f1_macro', 'precision_macro', 'recall_macro']

    for task_prefix in TASKS_TO_RUN_SEPARATELY:
        print(f"\n--- Task: {task_prefix.upper()} ---")
        model_metrics_list = all_tasks_runs_metrics[task_prefix].get(MODEL_NAME, [])

        if not model_metrics_list: print(f"No successful runs recorded."); continue

        metrics_df = pd.DataFrame(model_metrics_list)
        metrics_df = metrics_df.reindex(columns=metric_keys + list(metrics_df.columns.difference(metric_keys)))

        if metric_keys[0] not in metrics_df.columns or metrics_df[metric_keys[0]].isnull().all():
             print(f"No valid metrics found."); continue

        means = metrics_df[metric_keys].mean(); stds = metrics_df[metric_keys].std()
        n_valid_runs = metrics_df[metric_keys[0]].notna().sum()

        task_summary = {'Task': task_prefix, 'N_Valid_Runs': n_valid_runs }
        for key in metric_keys: task_summary[f'{key}_mean'] = means.get(key, np.nan); task_summary[f'{key}_std'] = stds.get(key, np.nan)
        overall_summary_stats.append(task_summary)

        summary_df_task = pd.DataFrame([task_summary])
        print("Mean +/- Std Dev across VALID runs:")
        for key in metric_keys: summary_df_task[f'{key}'] = summary_df_task.apply(lambda r: f"{r[f'{key}_mean']:.4f} +/- {r[f'{key}_std']:.4f}" if pd.notna(r[f'{key}_mean']) else "N/A", axis=1)
        print(summary_df_task[['Task', 'N_Valid_Runs'] + metric_keys].to_string(index=False))

    # Save overall summary comparison
    if SAVE_AGGREGATED_SUMMARY and overall_summary_stats:
        summary_filename = os.path.join(DATA_OUTPUT_FOLDER, "prediction_logistic_task_comparison_summary.csv")
        try: pd.DataFrame(overall_summary_stats).to_csv(summary_filename, index=False, sep=';', decimal='.')
        except Exception as e: print(f"Error saving overall summary: {e}")
        else: print(f"\nOverall task comparison summary saved to: {summary_filename}")


    # --- 5. Aggregate and Save Feature Coefficients PER TASK ---
    aggregated_coeffs_dfs = {} # Store aggregated DFs for plotting
    if SAVE_AGGREGATED_IMPORTANCES:
        print("\n--- Aggregating Feature Coefficients (Per Task) ---")
        for task_prefix in TASKS_TO_RUN_SEPARATELY:
             coefficient_lists = all_tasks_runs_importances[task_prefix].get(MODEL_NAME, [])
             valid_coefficient_lists = [s for s in coefficient_lists if s is not None and not s.empty] # Ensure Series is not empty
             if valid_coefficient_lists:
                 print(f"Aggregating for task: {task_prefix.upper()} ({len(valid_coefficient_lists)} valid runs)")
                 try:
                     coeff_df = pd.concat(valid_coefficient_lists, axis=1)
                     agg_coeff = pd.DataFrame({'Mean_Coefficient': coeff_df.mean(axis=1), 'Std_Coefficient': coeff_df.std(axis=1), 'N_Valid_Runs': coeff_df.notna().sum(axis=1)})
                     agg_coeff = agg_coeff.reindex(agg_coeff['Mean_Coefficient'].abs().sort_values(ascending=False).index)
                     aggregated_coeffs_dfs[task_prefix] = agg_coeff

                     agg_imp_filename = os.path.join(DATA_OUTPUT_FOLDER, f"prediction_{MODEL_NAME}_{task_prefix}_aggregated_coefficients.csv")
                     agg_coeff.to_csv(agg_imp_filename, sep=';', decimal='.', index_label='Feature')
                     print(f"Aggregated coefficients for task {task_prefix} saved to: {agg_imp_filename}")
                 except Exception as e: print(f"Error aggregating coefficients for task {task_prefix}: {e}")
             else: print(f"No valid coefficient data to aggregate for task {task_prefix}.")


    # --- 6. Generate Plots ---
    if GENERATE_PLOTS:
        print("\n--- Generating Plots ---")
        sns.set_theme(style="whitegrid") # Set a global theme

        # Plot Metric Distributions
        for metric in ['roc_auc', 'accuracy', 'f1_macro']:
             plot_title = f'Test {metric.replace("_"," ").title()} Distribution ({N_REPETITIONS} Runs)'
             plot_filename = os.path.join(PLOT_OUTPUT_FOLDER, f"plot_metric_distribution_{metric}.png")
             plot_metric_distributions(all_tasks_runs_metrics, TASKS_TO_RUN_SEPARATELY, metric, plot_title, plot_filename)

        # Plot Aggregated ROC Curves
        plot_title_roc = f'Average ROC Curves - Logistic Regression ({N_REPETITIONS} Runs)'
        plot_filename_roc = os.path.join(PLOT_OUTPUT_FOLDER, "plot_aggregated_roc_curves_logistic.png")
        plot_aggregated_roc_curves(all_tasks_runs_roc_data, all_tasks_runs_metrics, TASKS_TO_RUN_SEPARATELY, plot_title_roc, plot_filename_roc) # Pass full metrics dict

        # Plot Aggregated Coefficients (per task)
        for task_prefix, agg_coeff_df in aggregated_coeffs_dfs.items():
            # Check if dataframe exists and is not empty before plotting
            if agg_coeff_df is not None and not agg_coeff_df.empty:
                n_runs_info = f"{int(agg_coeff_df['N_Valid_Runs'].min())}-{int(agg_coeff_df['N_Valid_Runs'].max())}" if not agg_coeff_df['N_Valid_Runs'].empty else "N/A"
                plot_title_coeffs = f'Top {PLOT_TOP_N_COEFFICIENTS} Aggregated Coefficients\nTask: {task_prefix.upper()} ({n_runs_info} Valid Runs)'
                plot_filename_coeffs = os.path.join(PLOT_OUTPUT_FOLDER, f"plot_aggregated_coefficients_{task_prefix}.png")
                plot_aggregated_coefficients(agg_coeff_df, task_prefix, PLOT_TOP_N_COEFFICIENTS, plot_title_coeffs, plot_filename_coeffs)
            else:
                print(f"Skipping coefficient plot for task '{task_prefix}' due to missing data.")

    print(f"\n--- Script Finished ({time() - start_time_script:.1f}s Total) ---")