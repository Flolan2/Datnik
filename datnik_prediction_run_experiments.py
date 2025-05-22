#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to run DatScan prediction experiments.
FOCUS: BINARY Classification ONLY.
Includes logging of RFE selected features.
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
    GridSearchCV # Added for completeness, though current config uses Random
)
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFE # Import RFE to check instance type

# --- Force Add Parent Directory to Path ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming datnik_prediction_run_experiments.py is in Datnik/Online/
# And the 'prediction' package is Datnik/Online/prediction/
if current_script_dir not in sys.path:
    sys.path.insert(0, current_script_dir)
# If 'prediction' package is one level down from where this script is:
prediction_package_dir = os.path.join(current_script_dir, "prediction")
if prediction_package_dir not in sys.path:
     sys.path.insert(0, os.path.dirname(prediction_package_dir)) # Add 'Online' to path


# --- CHOOSE CONFIGURATION ----
from prediction import config # For BINARY classification
#
print(f"INFO: Using BINARY configuration from prediction.{config.__name__}.py") # Adjusted print

# --- Now Import from 'prediction' Package ---
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
    print(f"Ensure the 'prediction' directory with __init__.py exists relative to the script or in sys.path.")
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
logger.info(f"Script starting for BINARY classification.")
logger.info(f"Running splitting modes: {config.SPLITTING_MODES_TO_RUN}")
logger.info(f"Using Input folder: {config.INPUT_FOLDER}")
logger.info(f"Using Data Output folder: {config.DATA_OUTPUT_FOLDER}")
if config.GENERATE_PLOTS: logger.info(f"Using Plot Output folder: {config.PLOT_OUTPUT_FOLDER}")

# --- Load Data ---
logger.info("Loading data...")
df_raw = data_loader.load_data(config.INPUT_FOLDER, config.INPUT_CSV_NAME)
logger.info("Preparing data...")
X_full, y_full, groups_full, task_features_map, all_feature_cols = data_loader.prepare_data(df_raw, config)
N_CLASSES = y_full.nunique()
logger.info(f"!!!!!! N_CLASSES determined in main script as: {N_CLASSES} !!!!!!")
logger.info(f"!!!!!! y_full value_counts in main script: \n{y_full.value_counts(normalize=False).sort_index()} !!!!!!")
if N_CLASSES != 2:
    logger.error(f"Expected 2 classes for binary classification, but found {N_CLASSES}. Check data_loader.py and config.py.")
    # sys.exit(1) # Optional: Exit if not binary

# --- Initialize Results Storage ---
all_runs_metrics = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
all_runs_roc_data = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
all_runs_base_importances = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
all_runs_meta_importances = collections.defaultdict(lambda: collections.defaultdict(list)) # Stacking disabled
# <<< NEW: Initialize storage for RFE selected features >>>
all_runs_rfe_selected_features = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
# <<< END NEW >>>

# --- Main Experiment Loop ---
for split_mode in config.SPLITTING_MODES_TO_RUN:
    logger.info(f"\n===== Starting Mode: {split_mode.upper()} Split =====")
    for i_rep in range(config.N_REPETITIONS):
        current_random_state = config.BASE_RANDOM_STATE + i_rep; rep_start_time = time.time()
        logger.info(f"\n>>> Repetition {i_rep + 1}/{config.N_REPETITIONS} (Mode: {split_mode}, Seed: {current_random_state}) <<<")

        X_train_val, X_test, y_train_val, y_test, groups_train_val = None, None, None, None, None
        inner_cv = None; use_groups_for_cv = True

        try:
            logger.info(f"  Performing GROUP-BASED split (Patient ID)...")
            unique_patients=groups_full.unique(); n_patients=len(unique_patients)
            n_test_patients=int(np.ceil(n_patients*config.TEST_SET_SIZE)); min_train_patients_needed=config.N_SPLITS_CV
            n_test_patients=max(1,min(n_test_patients,n_patients-min_train_patients_needed)); n_train_patients=n_patients-n_test_patients
            if n_train_patients < min_train_patients_needed: logger.warning(f"Rep {i_rep+1}: Not enough unique patients ({n_train_patients}) for CV ({min_train_patients_needed} needed). Skipping rep."); continue
            rng=np.random.RandomState(current_random_state); shuffled_patients=rng.permutation(unique_patients)
            test_patient_ids=set(shuffled_patients[:n_test_patients]); train_patient_ids=set(shuffled_patients[n_test_patients:])
            train_mask=groups_full.isin(train_patient_ids); test_mask=groups_full.isin(test_patient_ids)

            X_train_val=X_full[train_mask].copy(); y_train_val=y_full[train_mask].copy()
            X_test=X_full[test_mask].copy(); y_test=y_full[test_mask].copy(); groups_train_val=groups_full[train_mask].copy()

            n_unique_groups_train=groups_train_val.nunique(); actual_n_splits_group=min(config.N_SPLITS_CV, n_unique_groups_train)
            if actual_n_splits_group>=2: inner_cv = StratifiedGroupKFold(n_splits=actual_n_splits_group, shuffle=True, random_state=current_random_state)
            else: logger.warning(f"Rep {i_rep+1}: Too few unique groups ({n_unique_groups_train}) for inner CV (need at least 2). Skipping inner CV for this rep."); inner_cv = None

            logger.info(f"  Split: {len(X_train_val)} train_val rows | {len(X_test)} test rows")
            if X_train_val.empty or X_test.empty: logger.warning("Empty train_val/test set after split. Skipping rep."); continue
            train_classes=np.unique(y_train_val); test_classes=np.unique(y_test)
            if len(train_classes)<2: logger.warning(f"Single class in train_val set ({train_classes}). Skipping rep."); continue
            if len(test_classes)<1: logger.warning(f"No classes in test set ({test_classes}). Skipping rep."); continue
            if len(test_classes)<2: logger.warning(f"Single class in test set ({test_classes}). Evaluation metrics might be limited.")

        except Exception as e_split_outer: logger.exception("UNEXPECTED ERROR during outer split setup:"); continue

        repetition_oof_preds = collections.defaultdict(dict)
        repetition_test_preds_proba = collections.defaultdict(dict)
        repetition_base_pipelines = collections.defaultdict(dict)

        for exp_config in config.CONFIGURATIONS_TO_RUN:
            config_name = exp_config['config_name']
            logger.info(f"  --- Running Config: {config_name} (Mode: {split_mode}) ---")
            for task_prefix in config.TASKS_TO_RUN_SEPARATELY:
                logger.info(f"    Task: {task_prefix.upper()}")
                task_feature_cols = task_features_map.get(task_prefix)
                if not task_feature_cols: logger.warning(f"    No features found for task '{task_prefix}'. Skipping."); all_runs_metrics[split_mode][config_name][task_prefix].append({}); continue
                
                valid_task_cols_train = [col for col in task_feature_cols if col in X_train_val.columns]
                valid_task_cols_test = [col for col in task_feature_cols if col in X_test.columns]

                if not valid_task_cols_train: logger.warning(f"    Features for task '{task_prefix}' not in train_val data. Skipping."); all_runs_metrics[split_mode][config_name][task_prefix].append({}); continue
                if not valid_task_cols_test: logger.warning(f"    Features for task '{task_prefix}' not in test data. Skipping test evaluation for this task might occur."); # Don't skip entirely yet, might fit

                X_train_val_task = X_train_val[valid_task_cols_train].copy()
                X_test_task = X_test[valid_task_cols_test].copy() if valid_task_cols_test else pd.DataFrame() # Handle if test cols missing

                if X_train_val_task.empty: logger.warning(f"    Train task data for '{task_prefix}' is empty. Skipping."); all_runs_metrics[split_mode][config_name][task_prefix].append({}); continue
                
                # Ensure y_train_val and groups_train_val are aligned with X_train_val_task (if rows were dropped in X_train_val due to FE NaNs)
                current_y_train_val = y_train_val.loc[X_train_val_task.index]
                current_groups_train_val = groups_train_val.loc[X_train_val_task.index] if groups_train_val is not None else None

                if len(np.unique(current_y_train_val)) < 2:
                    logger.warning(f"    Single class in y_train_val for task '{task_prefix}' after data prep. Skipping task for this config.")
                    all_runs_metrics[split_mode][config_name][task_prefix].append({})
                    continue

                try:
                    pipeline, search_params = pipeline_builder.build_pipeline_from_config(exp_config, current_random_state, config)
                    if not pipeline.steps: raise ValueError("Pipeline building returned empty pipeline.")
                except Exception as e_build: logger.error(f"    Pipeline build error for task {task_prefix}: {e_build}"); all_runs_metrics[split_mode][config_name][task_prefix].append({}); continue

                best_pipeline = None; tuning_best_score = None
                fit_successful = False; oof_pred_proba_positive_class = None

                try:
                    # Adjust inner_cv if current_groups_train_val has too few groups
                    current_inner_cv = inner_cv
                    if inner_cv is not None and current_groups_train_val is not None:
                        n_unique_groups_current_train = current_groups_train_val.nunique()
                        actual_n_splits_current_train = min(inner_cv.get_n_splits(), n_unique_groups_current_train)
                        if actual_n_splits_current_train < 2:
                            logger.warning(f"    Too few unique groups ({n_unique_groups_current_train}) in current_groups_train_val for inner CV. Disabling inner CV for this fit.")
                            current_inner_cv = None
                        elif actual_n_splits_current_train < inner_cv.get_n_splits():
                            logger.info(f"    Adjusting inner CV splits from {inner_cv.get_n_splits()} to {actual_n_splits_current_train} due to fewer groups in current_groups_train_val.")
                            current_inner_cv = StratifiedGroupKFold(n_splits=actual_n_splits_current_train, shuffle=True, random_state=current_random_state)
                    
                    if current_inner_cv is None:
                        logger.warning(f"    Inner CV not possible for task {task_prefix}. Fitting on full train_val...")
                        best_pipeline = clone(pipeline); best_pipeline.fit(X_train_val_task, current_y_train_val)
                        fit_successful = True; oof_pred_proba_positive_class = None
                        repetition_oof_preds[config_name][task_prefix] = None
                    else:
                        if config.ENABLE_TUNING and search_params:
                             search_type = exp_config.get('search_type', 'random'); search_cv_model = None
                             search_class = RandomizedSearchCV if search_type == 'random' else GridSearchCV
                             search_cv_kwargs = {
                                 'estimator': pipeline, 'param_distributions': search_params,
                                 'scoring': config.TUNING_SCORING_METRIC, 'cv': current_inner_cv,
                                 'n_jobs': -1, 'refit': True, 'error_score': 'raise' # Changed from 0 to 'raise' for debugging
                             }
                             if search_type == 'random':
                                 search_cv_kwargs['n_iter'] = config.N_ITER_RANDOM_SEARCH
                                 search_cv_kwargs['random_state'] = current_random_state
                                 logger.info(f"    Tuning with RandomizedSearchCV ({config.N_ITER_RANDOM_SEARCH} iters)...")
                             else: # grid
                                 search_cv_kwargs['param_grid'] = search_params # GridSearchCV expects param_grid
                                 del search_cv_kwargs['param_distributions'] # Remove this for GridSearchCV
                                 logger.info(f"    Tuning with GridSearchCV...")
                             
                             search_cv_model = search_class(**search_cv_kwargs)
                             fit_params_cv = {'groups': current_groups_train_val} if use_groups_for_cv and current_groups_train_val is not None else {}
                             search_cv_model.fit(X_train_val_task, current_y_train_val, **fit_params_cv)
                             best_pipeline = search_cv_model.best_estimator_; tuning_best_score = search_cv_model.best_score_
                             if isinstance(tuning_best_score, (float, np.floating, int, np.integer)):
                                logger.info(f"    Tuning complete. Best CV Score ({config.TUNING_SCORING_METRIC}): {tuning_best_score:.4f}")
                             elif tuning_best_score is None:
                                logger.info(f"    Tuning complete. Best CV Score ({config.TUNING_SCORING_METRIC}): N/A (was None)")
                             else:
                                logger.warning(f"    Tuning complete. Best CV Score ({config.TUNING_SCORING_METRIC}): Not a format-able number (type: {type(tuning_best_score)}, value: {tuning_best_score}). Tuning might have failed internally.")
                             fit_successful = True
                        else:
                             logger.info(f"    Fitting model (tuning disabled or no search params for task {task_prefix})...")
                             best_pipeline = clone(pipeline); best_pipeline.fit(X_train_val_task, current_y_train_val)
                             fit_successful = True

                        if fit_successful and current_inner_cv is not None: # OOF only if CV was possible
                             logger.info(f"    Generating OOF predictions for task {task_prefix}...")
                             cv_predict_params_oof = {'groups': current_groups_train_val} if use_groups_for_cv and current_groups_train_val is not None else {}
                             try:
                                 oof_pred_proba_all_classes = cross_val_predict(best_pipeline, X_train_val_task, current_y_train_val, cv=current_inner_cv, method='predict_proba', n_jobs=-1, **cv_predict_params_oof)
                                 oof_pred_proba_positive_class = oof_pred_proba_all_classes[:, 1]
                                 repetition_oof_preds[config_name][task_prefix] = oof_pred_proba_positive_class
                                 oof_auc = roc_auc_score(current_y_train_val, oof_pred_proba_positive_class)
                                 logger.info(f"    OOF ROC AUC (binary) for task {task_prefix}: {oof_auc:.4f}")
                             except ValueError as e_oof_val:
                                  logger.warning(f"    Could not generate/evaluate OOF predictions for task {task_prefix} (likely due to single class in a fold): {e_oof_val}")
                                  repetition_oof_preds[config_name][task_prefix] = None
                             except Exception as e_oof_other:
                                  logger.exception(f"    Unexpected error during OOF prediction/evaluation for task {task_prefix}:")
                                  repetition_oof_preds[config_name][task_prefix] = None
                    
                    if fit_successful:
                        repetition_base_pipelines[config_name][task_prefix] = best_pipeline
                        
                        # <<< NEW: Store RFE selected features >>>
                        if exp_config.get('feature_selector') == 'rfe' and 'feature_selector' in best_pipeline.named_steps:
                            final_selector_step = best_pipeline.named_steps['feature_selector']
                            if isinstance(final_selector_step, RFE):
                                try:
                                    selected_mask = final_selector_step.support_
                                    # Feature names are from X_train_val_task.columns, which were columns *before* RFE
                                    # If there was a selector *before* RFE (e.g. SelectKBest then RFE), this needs careful handling of column names
                                    # Assuming RFE is applied to the columns present in X_train_val_task directly after scaling/imputation
                                    rfe_selected_names = X_train_val_task.columns[selected_mask].tolist()
                                    all_runs_rfe_selected_features[split_mode][config_name][task_prefix].append(rfe_selected_names)
                                    logger.info(f"    RFE selected {len(rfe_selected_names)} features for task {task_prefix}.")
                                    logger.debug(f"    RFE selected features: {rfe_selected_names[:5]}...") # Log first 5
                                except Exception as e_rfe_log:
                                    logger.error(f"    Error extracting RFE features for task {task_prefix}: {e_rfe_log}")
                        # <<< END NEW >>>

                        if X_test_task.empty or len(np.unique(y_test.loc[X_test_task.index])) < 2 :
                            logger.warning(f"    Test data for task {task_prefix} is empty or has single class in y_test. Skipping test set evaluation.")
                            all_runs_metrics[split_mode][config_name][task_prefix].append({}) # Append empty dict
                        else:
                            logger.info(f"    Predicting and evaluating on test set for task {task_prefix}...")
                            current_y_test = y_test.loc[X_test_task.index] # Align y_test
                            test_pred_proba_all_classes = best_pipeline.predict_proba(X_test_task)
                            test_pred_labels = best_pipeline.predict(X_test_task)
                            repetition_test_preds_proba[config_name][task_prefix] = test_pred_proba_all_classes

                            eval_probs_positive_class = test_pred_proba_all_classes[:, 1]
                            test_metrics, test_roc_data_tuple = evaluation.evaluate_predictions(current_y_test, test_pred_labels, eval_probs_positive_class)
                            test_metrics['best_cv_score'] = tuning_best_score
                            all_runs_metrics[split_mode][config_name][task_prefix].append(test_metrics)
                            if test_roc_data_tuple: all_runs_roc_data[split_mode][config_name][task_prefix].append(test_roc_data_tuple)

                            auc_val=test_metrics.get('roc_auc','N/A'); acc_val=test_metrics.get('accuracy','N/A'); f1_val=test_metrics.get('f1_macro','N/A')
                            auc_str=f"{auc_val:.4f}" if isinstance(auc_val,(float,np.number)) else auc_val; acc_str=f"{acc_val:.4f}" if isinstance(acc_val,(float,np.number)) else acc_val; f1_str=f"{f1_val:.4f}" if isinstance(f1_val,(float,np.number)) else f1_val
                            logger.info(f"    Test ROC AUC: {auc_str}, Accuracy: {acc_str}, F1-Macro: {f1_str} for task {task_prefix}")
                        
                        imp = utils.get_feature_importances(best_pipeline, X_train_val_task.columns) # Get importances based on columns used for training
                        if imp is not None: all_runs_base_importances[split_mode][config_name][task_prefix].append(imp)

                except ValueError as ve: # Catch specific ValueError for "Only one class present"
                    if "Only one class present in y_true" in str(ve) or "must be >= 2." in str(ve) : # Latter for k_neighbors in SMOTE or n_splits
                        logger.error(f"    SKIPPING Config: {config_name}, Task: {task_prefix.upper()} due to ValueError: {ve} (likely single class in CV fold or invalid CV param). Appending empty metrics.")
                        all_runs_metrics[split_mode][config_name][task_prefix].append({})
                        repetition_oof_preds[config_name][task_prefix] = None
                        repetition_test_preds_proba[config_name][task_prefix] = None
                    else: # Other ValueErrors
                        logger.exception(f"    Detailed ValueError during fitting/tuning/prediction for task {task_prefix}:")
                        all_runs_metrics[split_mode][config_name][task_prefix].append({})
                except Exception as e_fit_eval:
                     logger.exception(f"    Detailed error during fitting/tuning/prediction/evaluation for task {task_prefix}:")
                     all_runs_metrics[split_mode][config_name][task_prefix].append({})
                     repetition_oof_preds[config_name][task_prefix] = None
                     repetition_test_preds_proba[config_name][task_prefix] = None

        if config.ENABLE_STACKING:
             logger.warning("Stacking is enabled in config but currently not recommended/tested in this binary-only script version.")

        logger.info(f">>> Repetition {i_rep + 1} (Mode: {split_mode}) finished in {(time.time() - rep_start_time):.1f} seconds <<<")

logger.info("\n--- All Repetitions and Modes Finished ---")

logger.info("\n--- Aggregating Results ---")
logger.info("Aggregating Metrics...")
agg_metrics_summary_df = results_processor.aggregate_metrics(all_runs_metrics, config)
logger.info("Aggregating Base Importances...")
agg_base_importances = results_processor.aggregate_importances(all_runs_base_importances, config, file_prefix="base_model_importance")

# <<< NEW: Aggregate RFE selected features >>>
logger.info("Aggregating RFE Selected Features...")
agg_rfe_selected_features = results_processor.aggregate_rfe_features(all_runs_rfe_selected_features, config)
# <<< END NEW >>>

if config.ENABLE_STACKING:
    logger.warning("Stacking is disabled. Skipping meta-importance aggregation.")
    agg_meta_importances = None
else:
     agg_meta_importances = None

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
            for metric in ['roc_auc', 'accuracy', 'f1_macro']:
                 if f"{metric}_mean" in plot_summary_df.columns:
                     plot_title = f'Mean Test {metric.replace("_"," ").title()} Distribution ({config.N_REPETITIONS} Reps)'
                     fname_base = os.path.join(config.PLOT_OUTPUT_FOLDER, f"plot_agg_metric_{metric}_comparison")
                     try:
                         if hasattr(plotting, 'plot_metric_distributions'): plotting.plot_metric_distributions(plot_summary_df, metric, plot_title, fname_base, config)
                         else: logger.warning(f"Plot function 'plot_metric_distributions' missing.")
                     except Exception as e_plot: logger.error(f"Error generating metric plot for {metric}: {e_plot}")
                 else: logger.warning(f"Metric {metric}_mean not found in summary.")

            logger.info("Generating separate ROC plots per mode.")
            for mode_roc in plot_summary_df['Mode'].unique():
                 mode_roc_data = all_runs_roc_data.get(mode_roc); mode_metrics = all_runs_metrics.get(mode_roc)
                 if not mode_roc_data or not mode_metrics:
                     logger.warning(f"No ROC or metric data found for mode '{mode_roc}'. Skipping ROC plot."); continue
                 configs_tasks_for_mode_roc = []
                 mode_plot_summary_roc = plot_summary_df[plot_summary_df['Mode'] == mode_roc]
                 for idx, row_roc in mode_plot_summary_roc.iterrows():
                      config_name_roc = row_roc['Config_Name']; task_name_roc = row_roc['Task_Name']
                      if config_name_roc in mode_roc_data and task_name_roc in mode_roc_data[config_name_roc]:
                            if any(roc_tuple for roc_tuple in mode_roc_data[config_name_roc][task_name_roc]):
                                configs_tasks_for_mode_roc.append((config_name_roc, task_name_roc))
                 if configs_tasks_for_mode_roc:
                      plot_title_roc = f'Aggregated ROC Curves ({mode_roc.upper()} Split, {config.N_REPETITIONS} Reps)'
                      fname_roc = os.path.join(config.PLOT_OUTPUT_FOLDER, f"plot_aggregated_roc_curves_{mode_roc}.png")
                      try:
                           if hasattr(plotting, 'plot_aggregated_roc_curves'): plotting.plot_aggregated_roc_curves(mode_roc_data, mode_metrics, configs_tasks_for_mode_roc, plot_title_roc, fname_roc, config)
                           else: logger.warning(f"Plot function 'plot_aggregated_roc_curves' missing.")
                      except Exception as e_plot_roc: logger.error(f"Error generating ROC plot for mode {mode_roc}: {e_plot_roc}")
                 else: logger.warning(f"No valid ROC data found to plot for mode '{mode_roc}'.")

            logger.info("Generating separate importance plots per mode/config/task.")
            all_aggregated_imps_structured = {}
            if agg_base_importances:
                 for mode_imp, mode_data_imp in agg_base_importances.items():
                     for cfg_imp, tasks_imp in mode_data_imp.items():
                         for task_imp, df_imp in tasks_imp.items(): all_aggregated_imps_structured[(mode_imp, cfg_imp, task_imp)] = df_imp
            if not all_aggregated_imps_structured: logger.warning("Skipping importance plots: No aggregated importance data found.")
            else:
                 for (mode_plot_imp, config_name_plot_imp, task_name_plot_imp), imp_df_plot in all_aggregated_imps_structured.items():
                      if imp_df_plot is not None and not imp_df_plot.empty:
                           try:
                               n_valid_runs_series = imp_df_plot.get('N_Valid_Runs'); n_runs_info = "N/A"
                               if n_valid_runs_series is not None and not n_valid_runs_series.empty:
                                    n_min_imp, n_max_imp = int(n_valid_runs_series.min()), int(n_valid_runs_series.max())
                                    n_runs_info = f"{n_min_imp}" if n_min_imp == n_max_imp else f"{n_min_imp}-{n_max_imp}"
                               plot_label = f"Base: {config_name_plot_imp} ({task_name_plot_imp.upper()})"; task_plot_name_file = task_name_plot_imp
                               title = f'Top {config.PLOT_TOP_N_FEATURES} Features ({mode_plot_imp.upper()} Split)\n{plot_label} ({n_runs_info} Runs)'
                               fname = os.path.join(config.PLOT_OUTPUT_FOLDER, f"plot_importance_{mode_plot_imp}_{config_name_plot_imp}_{task_plot_name_file}.png")
                               top_n_plot = config.PLOT_TOP_N_FEATURES
                               if hasattr(plotting, 'plot_aggregated_importances'): plotting.plot_aggregated_importances(imp_df_plot, config_name_plot_imp, task_name_plot_imp, top_n_plot, title, fname, config)
                               else: logger.warning(f"Plot function 'plot_aggregated_importances' missing.")
                           except Exception as e_plot_imp: logger.error(f"Error generating importance plot for {mode_plot_imp}/{config_name_plot_imp}/{task_name_plot_imp}: {e_plot_imp}")

logger.info(f"\n--- Script Finished ({time.time() - start_time_script:.1f} seconds Total) ---")