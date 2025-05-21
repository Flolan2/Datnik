#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to run DatScan prediction experiments.
FOCUS: BINARY Classification ONLY.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import collections
import time
import logging
import datetime
import seaborn as sns

# Scikit-learn imports
from sklearn.model_selection import (
    StratifiedGroupKFold, RandomizedSearchCV, cross_val_predict,
    train_test_split, StratifiedKFold, GridSearchCV
)
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

# --- Force Add Parent Directory to Path ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
if current_script_dir not in sys.path: sys.path.insert(0, current_script_dir)

# --- CHOOSE CONFIGURATION ----
# <<< MODIFIED: Hardcoded import for the standard BINARY config file >>>
from prediction import config # For BINARY classification
# <<< REMOVED: Multi-class import line >>>
#
print(f"INFO: Using BINARY configuration from {config.__name__}.py")
# --- End Configuration Choice ---

# --- Now Import from 'prediction' Package ---
# Modules will now use the settings passed via the 'config' object
try:
    from prediction import utils
    from prediction import data_loader
    from prediction import pipeline_builder
    from prediction import evaluation
    from prediction import results_processor
    from prediction import plotting
    print("Successfully imported modules from 'prediction' package.")
except ImportError as e:
    print(f"ERROR: Failed to import from the 'prediction' package.")
    print(f"Ensure the 'prediction' directory with __init__.py exists inside '{current_script_dir}'.")
    print(f"Current sys.path: {sys.path}")
    raise e

# --- Logging Configuration ---
try:
    os.makedirs(config.DATA_OUTPUT_FOLDER, exist_ok=True)
    log_dir = config.DATA_OUTPUT_FOLDER
    if config.GENERATE_PLOTS: os.makedirs(config.PLOT_OUTPUT_FOLDER, exist_ok=True)
except OSError as e_dir: print(f"FATAL: Error creating output directories: {e_dir}. Cannot proceed."); sys.exit(1)
log_format = '%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s'
log_level = logging.INFO; timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f"experiment_log_{timestamp}.log")
logger = logging.getLogger('DatnikExperiment'); logger.setLevel(log_level)
if logger.hasHandlers(): logger.handlers.clear()
ch = logging.StreamHandler(sys.stdout); ch.setLevel(log_level); ch_formatter = logging.Formatter(log_format); ch.setFormatter(ch_formatter); logger.addHandler(ch)
try:
    fh = logging.FileHandler(log_filename, mode='w', encoding='utf-8'); fh.setLevel(log_level); fh_formatter = logging.Formatter(log_format); fh.setFormatter(fh_formatter); logger.addHandler(fh)
    print(f"Logging configured. Log file will be saved to: {log_filename}")
except Exception as e_log: print(f"Warning: Could not set up file logging to {log_filename}: {e_log}")

# --- Basic Setup ---
start_time_script = time.time()
utils.setup_warnings()
logger.info(f"Script starting for BINARY classification.") # <<< MODIFIED: Simplified message
logger.info(f"Running splitting modes: {config.SPLITTING_MODES_TO_RUN}")
logger.info(f"Using Input folder: {config.INPUT_FOLDER}")
logger.info(f"Using Data Output folder: {config.DATA_OUTPUT_FOLDER}")
if config.GENERATE_PLOTS: logger.info(f"Using Plot Output folder: {config.PLOT_OUTPUT_FOLDER}")

# --- Load Data ---
logger.info("Loading data...")
df_raw = data_loader.load_data(config.INPUT_FOLDER, config.INPUT_CSV_NAME)
logger.info("Preparing data...")
# Pass the 'config' object (pointing to config.py) to prepare_data
X_full, y_full, groups_full, task_features_map, all_feature_cols = data_loader.prepare_data(df_raw, config)
N_CLASSES = y_full.nunique() # Should always be 2 now
logger.info(f"!!!!!! N_CLASSES determined in main script as: {N_CLASSES} !!!!!!")
logger.info(f"!!!!!! y_full value_counts in main script: \n{y_full.value_counts().sort_index()} !!!!!!")
if N_CLASSES != 2:
    logger.error(f"Expected 2 classes for binary classification, but found {N_CLASSES}. Check data_loader.py and config.py.")
    # sys.exit(1) # Optional: Exit if not binary

# --- Initialize Results Storage ---
all_runs_metrics = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
all_runs_roc_data = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
all_runs_base_importances = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
all_runs_meta_importances = collections.defaultdict(lambda: collections.defaultdict(list)) # Stacking disabled in config, but keep structure

# --- Main Experiment Loop ---
for split_mode in config.SPLITTING_MODES_TO_RUN:
    logger.info(f"\n===== Starting Mode: {split_mode.upper()} Split =====")
    for i_rep in range(config.N_REPETITIONS):
        current_random_state = config.BASE_RANDOM_STATE + i_rep; rep_start_time = time.time()
        logger.info(f"\n>>> Repetition {i_rep + 1}/{config.N_REPETITIONS} (Mode: {split_mode}, Seed: {current_random_state}) <<<")

        # --- 1. Perform Train/Test Split ---
        X_train_val, X_test, y_train_val, y_test, groups_train_val = None, None, None, None, None
        inner_cv = None; use_groups_for_cv = True # Only group mode now

        try:
            logger.info(f"  Performing GROUP-BASED split (Patient ID)...")
            unique_patients=groups_full.unique(); n_patients=len(unique_patients)
            n_test_patients=int(np.ceil(n_patients*config.TEST_SET_SIZE)); min_train_patients_needed=config.N_SPLITS_CV
            n_test_patients=max(1,min(n_test_patients,n_patients-min_train_patients_needed)); n_train_patients=n_patients-n_test_patients
            if n_train_patients < min_train_patients_needed: logger.warning(f"Rep {i_rep+1}: Not enough unique patients ({n_train_patients}) for CV ({min_train_patients_needed} needed). Skipping."); continue
            rng=np.random.RandomState(current_random_state); shuffled_patients=rng.permutation(unique_patients)
            test_patient_ids=set(shuffled_patients[:n_test_patients]); train_patient_ids=set(shuffled_patients[n_test_patients:])
            train_mask=groups_full.isin(train_patient_ids); test_mask=groups_full.isin(test_patient_ids)
            X_train_val=X_full[train_mask].copy(); y_train_val=y_full[train_mask].copy()
            X_test=X_full[test_mask].copy(); y_test=y_full[test_mask].copy(); groups_train_val=groups_full[train_mask].copy()

            n_unique_groups_train=groups_train_val.nunique(); actual_n_splits_group=min(config.N_SPLITS_CV, n_unique_groups_train)
            if actual_n_splits_group>=2: inner_cv = StratifiedGroupKFold(n_splits=actual_n_splits_group, shuffle=True, random_state=current_random_state)
            else: logger.warning(f"Rep {i_rep+1}: Too few unique groups ({n_unique_groups_train}) for inner CV. Skipping inner CV."); inner_cv = None

            logger.info(f"  Split: {len(X_train_val)} train rows | {len(X_test)} test rows")
            if X_train_val.empty or X_test.empty: logger.warning("Empty train/test set after split."); continue
            train_classes=y_train_val.unique(); test_classes=y_test.unique()
            # Allow single class in test, but not train for CV/fitting
            if len(train_classes)<2: logger.warning(f"Single class in train set ({len(train_classes)}). Skipping rep."); continue
            if len(test_classes)<1: logger.warning(f"No classes in test set ({len(test_classes)}). Skipping rep."); continue
            if len(test_classes)<2: logger.warning(f"Single class in test set ({len(test_classes)}). Evaluation metrics might be limited.")

        except Exception as e_split_outer: logger.exception("UNEXPECTED ERROR during outer split setup:"); continue

        repetition_oof_preds = collections.defaultdict(dict)
        repetition_test_preds_proba = collections.defaultdict(dict)
        repetition_base_pipelines = collections.defaultdict(dict)

        # --- 2. Loop through Configurations and Tasks ---
        for exp_config in config.CONFIGURATIONS_TO_RUN:
            config_name = exp_config['config_name']
            logger.info(f"  --- Running Config: {config_name} (Mode: {split_mode}) ---")
            for task_prefix in config.TASKS_TO_RUN_SEPARATELY:
                logger.info(f"    Task: {task_prefix.upper()}")
                task_feature_cols = task_features_map.get(task_prefix)
                if not task_feature_cols: logger.warning("No features found."); all_runs_metrics[split_mode][config_name][task_prefix].append({}); continue
                valid_task_cols = [col for col in task_feature_cols if col in X_train_val.columns]
                if not valid_task_cols: logger.warning("Features not in train data."); all_runs_metrics[split_mode][config_name][task_prefix].append({}); continue
                X_train_val_task = X_train_val[valid_task_cols]; X_test_task = X_test[valid_task_cols]
                if X_train_val_task.empty: logger.warning("Train task data empty."); all_runs_metrics[split_mode][config_name][task_prefix].append({}); continue

                try: # Build pipeline
                    # Pass the 'config' object (pointing to config.py)
                    pipeline, search_params = pipeline_builder.build_pipeline_from_config(exp_config, current_random_state, config)
                    if not pipeline.steps: raise ValueError("Pipeline building returned empty pipeline.")
                except Exception as e_build: logger.error(f"Pipeline build error: {e_build}"); all_runs_metrics[split_mode][config_name][task_prefix].append({}); continue

                best_pipeline = None; tuning_best_score = None
                fit_successful = False; oof_pred_proba = None

                try: # Fit/Tune/Eval Loop
                    if inner_cv is None: # Handle case where inner CV isn't possible (e.g., < N_SPLITS groups)
                        logger.warning("Inner CV not possible (too few groups or previous error). Fitting on full train_val...")
                        best_pipeline = clone(pipeline); best_pipeline.fit(X_train_val_task, y_train_val)
                        fit_successful = True; oof_pred_proba = None # No OOF preds possible
                        repetition_oof_preds[config_name][task_prefix] = None
                    else: # Proceed with inner CV
                        if config.ENABLE_TUNING and search_params:
                             search_type = exp_config.get('search_type', 'random'); search_cv = None
                             if search_type == 'grid':
                                 logger.info(f"Tuning with GridSearchCV...")
                                 search_cv = GridSearchCV(pipeline, search_params, scoring=config.TUNING_SCORING_METRIC, cv=inner_cv, n_jobs=-1, refit=True, error_score='raise')
                             else:
                                 logger.info(f"Tuning with RandomizedSearchCV ({config.N_ITER_RANDOM_SEARCH} iters)...")
                                 search_cv = RandomizedSearchCV(pipeline, search_params, n_iter=config.N_ITER_RANDOM_SEARCH, scoring=config.TUNING_SCORING_METRIC, cv=inner_cv, random_state=current_random_state, n_jobs=-1, refit=True, error_score='raise')

                             fit_params = {'groups': groups_train_val} if use_groups_for_cv and groups_train_val is not None else {}
                             search_cv.fit(X_train_val_task, y_train_val, **fit_params)
                             best_pipeline = search_cv.best_estimator_; tuning_best_score = search_cv.best_score_
                             logger.info(f"Tuning complete. Best CV Score ({config.TUNING_SCORING_METRIC}): {tuning_best_score:.4f}")
                             fit_successful = True
                        else:
                             logger.info("Fitting model (tuning disabled or no search params)...")
                             best_pipeline = clone(pipeline); best_pipeline.fit(X_train_val_task, y_train_val)
                             fit_successful = True

                        if fit_successful:
                             logger.info("Generating OOF predictions...")
                             cv_predict_params = {'groups': groups_train_val} if use_groups_for_cv and groups_train_val is not None else {}
                             # OOF predictions: Use predict_proba for AUC calculation
                             try:
                                 # Get probabilities for all classes (shape: n_samples, n_classes=2)
                                 oof_pred_proba_all_classes = cross_val_predict(best_pipeline, X_train_val_task, y_train_val, cv=inner_cv, method='predict_proba', n_jobs=-1, **cv_predict_params)
                                 # Store prob of positive class (index 1)
                                 oof_pred_proba = oof_pred_proba_all_classes[:, 1]
                                 repetition_oof_preds[config_name][task_prefix] = oof_pred_proba # Store just positive class prob

                                 # Calculate OOF AUC
                                 oof_auc = roc_auc_score(y_train_val, oof_pred_proba)
                                 logger.info(f"OOF ROC AUC (binary): {oof_auc:.4f}")

                             except ValueError as e_oof_val:
                                  logger.warning(f"Could not generate/evaluate OOF predictions (likely due to single class in a fold): {e_oof_val}")
                                  repetition_oof_preds[config_name][task_prefix] = None
                             except Exception as e_oof_other:
                                  logger.exception(f"Unexpected error during OOF prediction/evaluation:")
                                  repetition_oof_preds[config_name][task_prefix] = None


                    if fit_successful:
                        logger.info("Predicting and evaluating on test set...")
                        repetition_base_pipelines[config_name][task_prefix] = best_pipeline
                        test_pred_proba_all_classes = best_pipeline.predict_proba(X_test_task) # Shape: n_test_samples, 2
                        test_pred_labels = best_pipeline.predict(X_test_task)
                        repetition_test_preds_proba[config_name][task_prefix] = test_pred_proba_all_classes # Store all probs for potential later use

                        # Pass prob of positive class for binary evaluation
                        eval_probs = test_pred_proba_all_classes[:, 1]
                        test_metrics, test_roc_data = evaluation.evaluate_predictions(y_test, test_pred_labels, eval_probs)
                        test_metrics['best_cv_score'] = tuning_best_score
                        all_runs_metrics[split_mode][config_name][task_prefix].append(test_metrics)
                        # test_roc_data will be populated by evaluate_predictions for binary case
                        if test_roc_data: all_runs_roc_data[split_mode][config_name][task_prefix].append(test_roc_data)

                        auc_val=test_metrics.get('roc_auc','N/A'); acc_val=test_metrics.get('accuracy','N/A'); f1_val=test_metrics.get('f1_macro','N/A')
                        auc_str=f"{auc_val:.4f}" if isinstance(auc_val,(float,np.number)) else auc_val; acc_str=f"{acc_val:.4f}" if isinstance(acc_val,(float,np.number)) else acc_val; f1_str=f"{f1_val:.4f}" if isinstance(f1_val,(float,np.number)) else f1_val
                        logger.info(f"Test ROC AUC: {auc_str}, Accuracy: {acc_str}, F1-Macro: {f1_str}")

                        imp = utils.get_feature_importances(best_pipeline, valid_task_cols)
                        if imp is not None: all_runs_base_importances[split_mode][config_name][task_prefix].append(imp)

                except Exception as e_fit_eval:
                     # Log the error with traceback for the specific task/config
                     logger.exception(f"Detailed error during fitting/tuning/prediction/evaluation:")
                     logger.error(f"Error processing task {task_prefix} for config {config_name}: {repr(e_fit_eval)}")
                     # Append empty results to avoid breaking aggregation, but indicate failure
                     all_runs_metrics[split_mode][config_name][task_prefix].append({})
                     repetition_oof_preds[config_name][task_prefix] = None
                     repetition_test_preds_proba[config_name][task_prefix] = None
                     # Continue to the next task/config


        # --- Stacking --- (Keep disabled as per config.py)
        if config.ENABLE_STACKING:
             logger.warning("Stacking is enabled in config but currently not recommended/tested in this binary-only script version.")
             # Stacking logic would need careful review if re-enabled.

        logger.info(f">>> Repetition {i_rep + 1} (Mode: {split_mode}) finished in {(time.time() - rep_start_time):.1f} seconds <<<")

logger.info("\n--- All Repetitions and Modes Finished ---")

# --- 4. Aggregate and Summarize Results ---
logger.info("\n--- Aggregating Results ---")
logger.info("Aggregating Metrics...")
# Pass the 'config' object (pointing to config.py)
agg_metrics_summary_df = results_processor.aggregate_metrics(all_runs_metrics, config)
logger.info("Aggregating Base Importances...")
# Pass the 'config' object (pointing to config.py)
agg_base_importances = results_processor.aggregate_importances(all_runs_base_importances, config, file_prefix="base_model_importance")

if config.ENABLE_STACKING: # Stacking disabled, this block will be skipped
    logger.warning("Stacking is disabled. Skipping meta-importance aggregation.")
    agg_meta_importances = None
else:
     agg_meta_importances = None

# --- 5. Generate Plots ---
if config.GENERATE_PLOTS:
    logger.info("\n--- Generating Plots ---")
    if agg_metrics_summary_df is None or agg_metrics_summary_df.empty:
        logger.warning("Aggregated metrics summary is empty. Skipping plot generation.")
    else:
        try: sns.set_theme(style="whitegrid")
        except Exception as e_sns: logger.error(f"Error setting seaborn theme: {e_sns}")

        plot_summary_df = agg_metrics_summary_df[
            (agg_metrics_summary_df['N_Valid_Runs'] > 0) &
            (agg_metrics_summary_df[f"{config.TUNING_SCORING_METRIC}_mean"].notna() if config.TUNING_SCORING_METRIC+'_mean' in agg_metrics_summary_df else True)
        ].copy()

        if plot_summary_df.empty: logger.warning("No valid runs found for plotting after filtering.")
        else:
            logger.info(f"Plotting results for {len(plot_summary_df)} successful configuration/task combinations.")

            # Plot Metrics (Boxplots per mode)
            for metric in ['roc_auc', 'accuracy', 'f1_macro']:
                 if f"{metric}_mean" in plot_summary_df.columns:
                     plot_title = f'Mean Test {metric.replace("_"," ").title()} Distribution ({config.N_REPETITIONS} Reps)'
                     fname_base = os.path.join(config.PLOT_OUTPUT_FOLDER, f"plot_agg_metric_{metric}_comparison")
                     try:
                         # Pass the 'config' object (pointing to config.py)
                         if hasattr(plotting, 'plot_metric_distributions'): plotting.plot_metric_distributions(plot_summary_df, metric, plot_title, fname_base, config)
                         else: logger.warning(f"Plot function 'plot_metric_distributions' missing.")
                     except Exception as e_plot: logger.error(f"Error generating metric plot for {metric}: {e_plot}")
                 else: logger.warning(f"Metric {metric}_mean not found in summary.")

            # Plot ROC Curves (Now always attempted as it's binary)
            logger.info("Generating separate ROC plots per mode.")
            for mode in plot_summary_df['Mode'].unique(): # Iterate through modes found in results ('group' only now)
                 mode_roc_data = all_runs_roc_data.get(mode); mode_metrics = all_runs_metrics.get(mode)
                 if not mode_roc_data or not mode_metrics:
                     logger.warning(f"No ROC or metric data found for mode '{mode}'. Skipping ROC plot."); continue

                 configs_tasks_for_mode_roc = []
                 mode_plot_summary = plot_summary_df[plot_summary_df['Mode'] == mode]
                 for idx, row in mode_plot_summary.iterrows():
                      config_name = row['Config_Name']; task_name = row['Task_Name']
                      # Check if ROC data exists and is not empty for this combo
                      if config_name in mode_roc_data and task_name in mode_roc_data[config_name]:
                            if any(roc_tuple for roc_tuple in mode_roc_data[config_name][task_name]): # Check if list contains non-empty tuples
                                configs_tasks_for_mode_roc.append((config_name, task_name))

                 if configs_tasks_for_mode_roc:
                      plot_title_roc = f'Aggregated ROC Curves ({mode.upper()} Split, {config.N_REPETITIONS} Reps)'
                      fname_roc = os.path.join(config.PLOT_OUTPUT_FOLDER, f"plot_aggregated_roc_curves_{mode}.png")
                      try:
                           # Pass the 'config' object (pointing to config.py)
                           if hasattr(plotting, 'plot_aggregated_roc_curves'): plotting.plot_aggregated_roc_curves(mode_roc_data, mode_metrics, configs_tasks_for_mode_roc, plot_title_roc, fname_roc, config)
                           else: logger.warning(f"Plot function 'plot_aggregated_roc_curves' missing.")
                      except Exception as e_plot_roc: logger.error(f"Error generating ROC plot for mode {mode}: {e_plot_roc}")
                 else: logger.warning(f"No valid ROC data found to plot for mode '{mode}'.")


            # Plot Importances (Separate plots per mode/config/task)
            logger.info("Generating separate importance plots per mode/config/task.")
            all_aggregated_imps_structured = {}
            if agg_base_importances:
                 for mode, mode_data in agg_base_importances.items():
                     for cfg, tasks in mode_data.items():
                         for task, df in tasks.items(): all_aggregated_imps_structured[(mode, cfg, task)] = df
            # Stacking is disabled, agg_meta_importances is None

            if not all_aggregated_imps_structured: logger.warning("Skipping importance plots: No aggregated importance data found.")
            else:
                 for (mode, config_name_imp, task_name_imp), imp_df in all_aggregated_imps_structured.items():
                      if imp_df is not None and not imp_df.empty:
                           try:
                               n_valid_runs_series = imp_df.get('N_Valid_Runs'); n_runs_info = "N/A"
                               if n_valid_runs_series is not None and not n_valid_runs_series.empty:
                                    n_min, n_max = int(n_valid_runs_series.min()), int(n_valid_runs_series.max())
                                    n_runs_info = f"{n_min}" if n_min == n_max else f"{n_min}-{n_max}"
                               # is_meta will always be False now
                               plot_label = f"Base: {config_name_imp} ({task_name_imp.upper()})"; task_plot_name = task_name_imp
                               title = f'Top {config.PLOT_TOP_N_FEATURES} Features ({mode.upper()} Split)\n{plot_label} ({n_runs_info} Runs)'
                               fname = os.path.join(config.PLOT_OUTPUT_FOLDER, f"plot_importance_{mode}_{config_name_imp}_{task_plot_name}.png")
                               top_n_plot = config.PLOT_TOP_N_FEATURES
                               # Pass the 'config' object (pointing to config.py)
                               if hasattr(plotting, 'plot_aggregated_importances'): plotting.plot_aggregated_importances(imp_df, config_name_imp, task_name_imp, top_n_plot, title, fname, config)
                               else: logger.warning(f"Plot function 'plot_aggregated_importances' missing.")
                           except Exception as e_plot_imp: logger.error(f"Error generating importance plot for {mode}/{config_name_imp}/{task_name_imp}: {e_plot_imp}")

logger.info(f"\n--- Script Finished ({time.time() - start_time_script:.1f} seconds Total) ---")