#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to run DatScan prediction experiments, comparing splitting modes.
# ... (keep existing docstring) ...
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
    train_test_split, StratifiedKFold, GridSearchCV # <<< ADDED GridSearchCV
)
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

# --- Force Add Parent Directory to Path ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
if current_script_dir not in sys.path:
    sys.path.insert(0, current_script_dir)
    print(f"DEBUG: Added '{current_script_dir}' to sys.path")
else:
    print(f"DEBUG: '{current_script_dir}' is already in sys.path")


# --- Now Import from 'prediction' Package ---
try:
    from prediction import config
    from prediction import utils
    from prediction import data_loader
    from prediction import pipeline_builder
    from prediction import evaluation
    from prediction import results_processor
    from prediction import plotting
    print("Successfully imported modules from 'prediction' package.")
except ImportError as e:
    print(f"ERROR: Failed to import from the 'prediction' package even after modifying sys.path.")
    print(f"Ensure the 'prediction' directory with __init__.py exists inside '{current_script_dir}'.")
    print(f"Current sys.path: {sys.path}")
    raise e

# ================================================
# ============ Logging Configuration =============
# ================================================
try:
    os.makedirs(config.DATA_OUTPUT_FOLDER, exist_ok=True)
    log_dir = config.DATA_OUTPUT_FOLDER
    if config.GENERATE_PLOTS: os.makedirs(config.PLOT_OUTPUT_FOLDER, exist_ok=True)
except OSError as e_dir:
    print(f"FATAL: Error creating output directories: {e_dir}. Cannot proceed.")
    sys.exit(1)

log_format = '%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s'
log_level = logging.INFO
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f"experiment_log_{timestamp}.log")
logger = logging.getLogger('DatnikExperiment')
logger.setLevel(log_level)
if logger.hasHandlers(): logger.handlers.clear()
ch = logging.StreamHandler(sys.stdout); ch.setLevel(log_level); ch_formatter = logging.Formatter(log_format); ch.setFormatter(ch_formatter); logger.addHandler(ch)
try:
    fh = logging.FileHandler(log_filename, mode='w', encoding='utf-8'); fh.setLevel(log_level); fh_formatter = logging.Formatter(log_format); fh.setFormatter(fh_formatter); logger.addHandler(fh)
    print(f"Logging configured. Log file will be saved to: {log_filename}")
except Exception as e_log: print(f"Warning: Could not set up file logging to {log_filename}: {e_log}")
# ================================================
# ============== End Logging Setup ===============
# ================================================

# --- Basic Setup ---
start_time_script = time.time()
utils.setup_warnings()
logger.info("Script starting.")
logger.info(f"Running splitting modes: {config.SPLITTING_MODES_TO_RUN}")
logger.info(f"Using Input folder: {config.INPUT_FOLDER}")
logger.info(f"Using Data Output folder: {config.DATA_OUTPUT_FOLDER}")
if config.GENERATE_PLOTS: logger.info(f"Using Plot Output folder: {config.PLOT_OUTPUT_FOLDER}")

# --- Load Data ---
logger.info("Loading data...")
df_raw = data_loader.load_data(config.INPUT_FOLDER, config.INPUT_CSV_NAME)
logger.info("Preparing data...")
X_full, y_full, groups_full, task_features_map, all_feature_cols = data_loader.prepare_data(df_raw)

# --- Initialize Results Storage ---
all_runs_metrics = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
all_runs_roc_data = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
all_runs_base_importances = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
all_runs_meta_importances = collections.defaultdict(lambda: collections.defaultdict(list)) # CORRECTED: Inner default is list

# --- Main Experiment Loop ---
for split_mode in config.SPLITTING_MODES_TO_RUN: # Should only be 'group' now unless changed in config
    logger.info(f"\n=================================================")
    logger.info(f"===== Starting Mode: {split_mode.upper()} Split =====")
    logger.info(f"=================================================")

    for i_rep in range(config.N_REPETITIONS):
        current_random_state = config.BASE_RANDOM_STATE + i_rep
        rep_start_time = time.time()
        logger.info(f"\n>>> Repetition {i_rep + 1}/{config.N_REPETITIONS} (Mode: {split_mode}, Seed: {current_random_state}) <<<")

        # --- 1. Perform Train/Test Split ---
        # (Keep splitting logic from previous version, using StratifiedGroupKFold for 'group')
        X_train_val, X_test, y_train_val, y_test, groups_train_val = None, None, None, None, None
        inner_cv = None
        use_groups_for_cv = True # Always True now since only running 'group' mode

        try:
            logger.info(f"  Performing GROUP-BASED split (Patient ID)...")
            unique_patients = groups_full.unique()
            n_patients = len(unique_patients)
            n_test_patients = int(np.ceil(n_patients * config.TEST_SET_SIZE))
            min_train_patients_needed = config.N_SPLITS_CV
            n_test_patients = max(1, min(n_test_patients, n_patients - min_train_patients_needed))
            n_train_patients = n_patients - n_test_patients

            if n_train_patients < min_train_patients_needed:
                logger.warning(f"Rep {i_rep+1} ({split_mode}): Not enough unique patients for {config.N_SPLITS_CV}-fold Group CV. Skipping repetition.")
                continue

            rng = np.random.RandomState(current_random_state)
            shuffled_patients = rng.permutation(unique_patients)
            test_patient_ids = set(shuffled_patients[:n_test_patients])
            train_patient_ids = set(shuffled_patients[n_test_patients:])
            train_mask = groups_full.isin(train_patient_ids)
            test_mask = groups_full.isin(test_patient_ids)
            X_train_val = X_full[train_mask].copy()
            y_train_val = y_full[train_mask].copy()
            X_test = X_full[test_mask].copy()
            y_test = y_full[test_mask].copy()
            groups_train_val = groups_full[train_mask].copy()

            n_unique_groups_train = groups_train_val.nunique()
            actual_n_splits_group = min(config.N_SPLITS_CV, n_unique_groups_train)
            if actual_n_splits_group >= 2:
                 inner_cv = StratifiedGroupKFold(n_splits=actual_n_splits_group, shuffle=True, random_state=current_random_state)
            else:
                 logger.warning(f"Rep {i_rep+1} ({split_mode}): Too few unique groups ({n_unique_groups_train}) in training data for inner CV. OOF/Tuning might fail.")
                 inner_cv = None

            logger.info(f"  Mode '{split_mode}' Split: {len(X_train_val)} train rows | {len(X_test)} test rows")

            if X_train_val is None or X_train_val.empty or X_test is None or X_test.empty: continue
            train_classes = y_train_val.unique(); test_classes = y_test.unique()
            if len(train_classes) < 2 or len(test_classes) < 2: continue

        except Exception as e_split_outer:
             logger.exception(f"  Rep {i_rep+1} ({split_mode}): UNEXPECTED ERROR during outer split setup:")
             logger.error(f"  Error details: {repr(e_split_outer)}")
             continue

        repetition_oof_preds = collections.defaultdict(dict)
        repetition_test_preds_proba = collections.defaultdict(dict)
        repetition_base_pipelines = collections.defaultdict(dict)

        # --- 2. Loop through Configurations and Tasks ---
        for exp_config in config.CONFIGURATIONS_TO_RUN:
            config_name = exp_config['config_name']
            logger.info(f"  --- Running Config: {config_name} (Mode: {split_mode}) ---")

            for task_prefix in config.TASKS_TO_RUN_SEPARATELY:
                logger.info(f"    Task: {task_prefix.upper()}")
                # ... (Get features, check validity) ...
                task_feature_cols = task_features_map.get(task_prefix)
                if not task_feature_cols:
                    logger.warning(f"      No features found for task '{task_prefix}'. Skipping.")
                    all_runs_metrics[split_mode][config_name][task_prefix].append({})
                    continue
                valid_task_cols = [col for col in task_feature_cols if col in X_train_val.columns]
                if not valid_task_cols:
                    logger.warning(f"      Feature columns for task '{task_prefix}' not found in training data subset. Skipping.")
                    all_runs_metrics[split_mode][config_name][task_prefix].append({})
                    continue
                X_train_val_task = X_train_val[valid_task_cols]
                X_test_task = X_test[valid_task_cols]
                if X_train_val_task.empty:
                     logger.warning(f"      Train data for task '{task_prefix}' is empty after column selection. Skipping.")
                     all_runs_metrics[split_mode][config_name][task_prefix].append({})
                     continue

                try:
                    pipeline, search_params = pipeline_builder.build_pipeline_from_config(
                        exp_config=exp_config, random_state=current_random_state )
                    if not pipeline.steps: raise ValueError("Pipeline building returned empty pipeline.")
                except Exception as e_build:
                    logger.error(f"      [{task_prefix}] Error building pipeline: {e_build}. Skipping task.")
                    all_runs_metrics[split_mode][config_name][task_prefix].append({})
                    continue

                best_pipeline = None; tuning_best_score = None
                fit_successful = False; oof_pred_proba = None

                try:
                    if inner_cv is None:
                         logger.warning(f"      [{task_prefix}] Inner CV not possible. Fitting on full train_val...")
                         best_pipeline = clone(pipeline)
                         best_pipeline.fit(X_train_val_task, y_train_val)
                         fit_successful = True; oof_pred_proba = None
                         repetition_oof_preds[config_name][task_prefix] = None
                    else:
                         if config.ENABLE_TUNING and search_params:
                             search_type = exp_config.get('search_type', 'random')
                             search_cv = None # Initialize

                             if search_type == 'grid':
                                 logger.info(f"      [{task_prefix}] Tuning hyperparameters with GridSearchCV...")
                                 search_cv = GridSearchCV(
                                     estimator=pipeline, param_grid=search_params,
                                     scoring=config.TUNING_SCORING_METRIC, cv=inner_cv,
                                     n_jobs=-1, refit=True, error_score='raise'
                                 )
                             else: # Default to random
                                 logger.info(f"      [{task_prefix}] Tuning hyperparameters with RandomizedSearchCV ({config.N_ITER_RANDOM_SEARCH} iterations)...")
                                 search_cv = RandomizedSearchCV(
                                     estimator=pipeline, param_distributions=search_params, n_iter=config.N_ITER_RANDOM_SEARCH,
                                     scoring=config.TUNING_SCORING_METRIC, cv=inner_cv, random_state=current_random_state,
                                     n_jobs=-1, refit=True, error_score='raise'
                                 )

                             fit_params = {}
                             if use_groups_for_cv and groups_train_val is not None: fit_params['groups'] = groups_train_val
                             search_cv.fit(X_train_val_task, y_train_val, **fit_params)

                             best_pipeline = search_cv.best_estimator_
                             tuning_best_score = search_cv.best_score_
                             logger.info(f"      [{task_prefix}] Tuning complete. Best CV Score: {tuning_best_score:.4f}")
                             fit_successful = True
                         else:
                             logger.info(f"      [{task_prefix}] Fitting model (tuning disabled or no params)...")
                             best_pipeline = clone(pipeline)
                             best_pipeline.fit(X_train_val_task, y_train_val)
                             fit_successful = True

                         if fit_successful:
                             logger.info(f"      [{task_prefix}] Generating OOF predictions...")
                             cv_predict_params = {}
                             if use_groups_for_cv and groups_train_val is not None: cv_predict_params['groups'] = groups_train_val
                             oof_pred_proba = cross_val_predict(
                                 best_pipeline, X_train_val_task, y_train_val, cv=inner_cv,
                                 method='predict_proba', n_jobs=-1, **cv_predict_params
                             )[:, 1]
                             repetition_oof_preds[config_name][task_prefix] = oof_pred_proba
                             try:
                                  if len(np.unique(y_train_val)) > 1: logger.info(f"      [{task_prefix}] OOF ROC AUC: {roc_auc_score(y_train_val, oof_pred_proba):.4f}")
                                  else: logger.warning(f"      [{task_prefix}] Cannot calculate OOF ROC AUC (single class).")
                             except Exception as e_oof_auc: logger.warning(f"      [{task_prefix}] Could not calculate OOF ROC AUC: {e_oof_auc}")

                    if fit_successful:
                        logger.info(f"      [{task_prefix}] Predicting and evaluating on test set...")
                        repetition_base_pipelines[config_name][task_prefix] = best_pipeline
                        test_pred_proba = best_pipeline.predict_proba(X_test_task)[:, 1]
                        test_pred_labels = best_pipeline.predict(X_test_task)
                        repetition_test_preds_proba[config_name][task_prefix] = test_pred_proba

                        test_metrics, test_roc_data = evaluation.evaluate_predictions(y_test, test_pred_labels, test_pred_proba)
                        test_metrics['best_cv_score'] = tuning_best_score
                        all_runs_metrics[split_mode][config_name][task_prefix].append(test_metrics)
                        if test_roc_data: all_runs_roc_data[split_mode][config_name][task_prefix].append(test_roc_data)

                        auc_val=test_metrics.get('roc_auc','N/A'); acc_val=test_metrics.get('accuracy','N/A'); f1_val=test_metrics.get('f1_macro','N/A')
                        auc_str=f"{auc_val:.4f}" if isinstance(auc_val,(float,np.number)) else auc_val; acc_str=f"{acc_val:.4f}" if isinstance(acc_val,(float,np.number)) else acc_val; f1_str=f"{f1_val:.4f}" if isinstance(f1_val,(float,np.number)) else f1_val
                        logger.info(f"      [{task_prefix}] Test ROC AUC: {auc_str}, Accuracy: {acc_str}, F1-Macro: {f1_str}")

                        imp = utils.get_feature_importances(best_pipeline, valid_task_cols)
                        if imp is not None: all_runs_base_importances[split_mode][config_name][task_prefix].append(imp)

                except Exception as e_fit_eval:
                     logger.exception(f"      [{task_prefix}] Detailed error during fitting/tuning/prediction/evaluation (Mode: {split_mode}):")
                     logger.error(f"      [{task_prefix}] Error processing task: {repr(e_fit_eval)}")
                     all_runs_metrics[split_mode][config_name][task_prefix].append({})
                     repetition_oof_preds[config_name][task_prefix] = None
                     repetition_test_preds_proba[config_name][task_prefix] = None

        # --- 3. Stacking ---
        if config.ENABLE_STACKING:
            # ... (Keep stacking logic as is, it will use the base config name from config.py) ...
            # ... (Ensure the all_runs_meta_importances line uses the corrected initialization) ...
             base_cfg_name = config.STACKING_BASE_CONFIG_NAME
             stacking_results_key = f"stacked_meta_on_{base_cfg_name}"
             logger.info(f"  --- Preparing for Stacking (Base: {base_cfg_name}, Mode: {split_mode}) ---")

             meta_train_features_list=[]; meta_test_features_list=[]; valid_stacking_tasks=[]; stacking_feature_names=[]

             if base_cfg_name not in repetition_oof_preds and base_cfg_name not in repetition_test_preds_proba:
                 logger.warning(f"    Stacking skipped ({split_mode}): Base config '{base_cfg_name}' predictions not found.")
                 all_runs_metrics[split_mode][stacking_results_key]['meta'].append({})
             else:
                 for task in config.STACKING_TASKS_TO_COMBINE:
                     oof_preds = repetition_oof_preds.get(base_cfg_name, {}).get(task)
                     test_preds = repetition_test_preds_proba.get(base_cfg_name, {}).get(task)
                     if oof_preds is not None and test_preds is not None:
                          if len(oof_preds) == len(y_train_val):
                               feature_name = f"{task}_pred_{base_cfg_name}"
                               meta_train_features_list.append(pd.Series(oof_preds, name=feature_name))
                               meta_test_features_list.append(pd.Series(test_preds, name=feature_name))
                               valid_stacking_tasks.append(task); stacking_feature_names.append(feature_name)
                               logger.info(f"    Added '{task}' predictions from '{base_cfg_name}' for stacking ({split_mode}).")
                          else: logger.error(f"    Stacking Error ({split_mode}): Mismatched OOF length for '{task}'. Skipping.")
                     elif oof_preds is None and test_preds is not None: logger.warning(f"    Stacking Warning ({split_mode}): OOF missing for '{task}'. Skipping.")
                     else: logger.warning(f"    Stacking Info ({split_mode}): Valid OOF/Test missing for '{task}'. Skipping.")

                 if len(valid_stacking_tasks) == len(config.STACKING_TASKS_TO_COMBINE):
                     meta_train_features_df = pd.concat(meta_train_features_list, axis=1)
                     meta_test_features_df = pd.concat(meta_test_features_list, axis=1)
                     if meta_train_features_df.isnull().any().any() or meta_test_features_df.isnull().any().any():
                          logger.error(f"    Stacking Error ({split_mode}): NaN values found in base predictions.")
                          all_runs_metrics[split_mode][stacking_results_key]['meta'].append({})
                     else:
                          meta_model_name = config.META_CLASSIFIER_CONFIG['model_name']
                          logger.info(f"    Training Meta-Model ({meta_model_name}, Mode: {split_mode})...")
                          try:
                              meta_model_info = config.META_CLASSIFIER_CONFIG; meta_model = clone(meta_model_info['estimator'])
                              try: meta_model.set_params(random_state=current_random_state + 100)
                              except ValueError: pass
                              meta_model.fit(meta_train_features_df, y_train_val)
                              meta_y_pred_test = meta_model.predict(meta_test_features_df)
                              meta_y_pred_proba_test = meta_model.predict_proba(meta_test_features_df)[:, 1]
                              stacked_metrics, stacked_roc_data = evaluation.evaluate_predictions(y_test, meta_y_pred_test, meta_y_pred_proba_test)

                              all_runs_metrics[split_mode][stacking_results_key]['meta'].append(stacked_metrics)
                              if stacked_roc_data: all_runs_roc_data[split_mode][stacking_results_key]['meta'].append(stacked_roc_data)

                              auc_val=stacked_metrics.get('roc_auc','N/A'); acc_val=stacked_metrics.get('accuracy','N/A'); f1_val=stacked_metrics.get('f1_macro','N/A')
                              auc_str=f"{auc_val:.4f}" if isinstance(auc_val,(float,np.number)) else auc_val; acc_str=f"{acc_val:.4f}" if isinstance(acc_val,(float,np.number)) else acc_val; f1_str=f"{f1_val:.4f}" if isinstance(f1_val,(float,np.number)) else f1_val
                              logger.info(f"    Stacking Test ({split_mode}) ROC AUC: {auc_str}, Accuracy: {acc_str}, F1-Macro: {f1_str}")

                              meta_imp = utils.get_feature_importances(meta_model, stacking_feature_names)
                              if meta_imp is not None:
                                   meta_storage_key = f"{meta_model_name}_on_{base_cfg_name}"
                                   # Use the corrected initialization structure
                                   all_runs_meta_importances[split_mode][meta_storage_key].append(meta_imp)
                          except Exception as e_meta:
                               logger.exception(f"    Stacking Error ({split_mode}) during meta-model training/prediction:")
                               logger.error(f"    Meta-model failed ({split_mode}): {repr(e_meta)}")
                               all_runs_metrics[split_mode][stacking_results_key]['meta'].append({})
                 else:
                     logger.warning(f"    Stacking Skipped ({split_mode}): Did not get valid predictions for all required base tasks.")
                     all_runs_metrics[split_mode][stacking_results_key]['meta'].append({})

        logger.info(f">>> Repetition {i_rep + 1} (Mode: {split_mode}) finished in {(time.time() - rep_start_time):.1f} seconds <<<")
    # <<< End Repetition Loop >>>
# <<< End Mode Loop >>>

logger.info("\n--- All Repetitions and Modes Finished ---")

# --- 4. Aggregate and Summarize Results ---
# (Keep calls, assuming results_processor.py updated for 'mode')
logger.info("\n--- Aggregating Results ---")
logger.info("Aggregating Metrics...")
agg_metrics_summary_df = results_processor.aggregate_metrics(all_runs_metrics)

logger.info("Aggregating Base Importances...")
agg_base_importances = results_processor.aggregate_importances( all_runs_base_importances, file_prefix="base_model_importance")

logger.info("Aggregating Meta Importances...")
formatted_meta_importances = collections.defaultdict(lambda: collections.defaultdict(dict))
for mode, mode_data in all_runs_meta_importances.items():
    for meta_key, imp_list in mode_data.items():
        formatted_meta_importances[mode][meta_key]['meta_coeffs'] = imp_list
agg_meta_importances = results_processor.aggregate_importances( formatted_meta_importances, file_prefix="meta_model_importance")

# --- 5. Generate Plots ---
# (Keep calls, assuming plotting.py updated for 'mode' or generating separate plots)
if config.GENERATE_PLOTS:
    logger.info("\n--- Generating Plots ---")
    if agg_metrics_summary_df is None or agg_metrics_summary_df.empty:
        logger.warning("Aggregated metrics summary is empty. Skipping plot generation.")
    else:
        try: sns.set_theme(style="whitegrid")
        except Exception as e_sns: logger.error(f"Error setting seaborn theme: {e_sns}")

        # Filter only valid runs for plotting base
        plot_summary_df = agg_metrics_summary_df[
            (agg_metrics_summary_df['N_Valid_Runs'] > 0) &
            (agg_metrics_summary_df[f"{config.TUNING_SCORING_METRIC}_mean"].notna())
        ].copy()

        if plot_summary_df.empty:
             logger.warning("No valid runs found for plotting after filtering.")
        else:
            logger.info(f"Plotting results for {len(plot_summary_df)} successful mode/configuration/task combinations.")

            # --- Plot Metrics --- (Will generate separate plots per mode now)
            for metric in ['roc_auc', 'accuracy', 'f1_macro']:
                 if f"{metric}_mean" in plot_summary_df.columns:
                     plot_title = f'Mean Test {metric.replace("_"," ").title()} Distribution ({config.N_REPETITIONS} Reps)'
                     fname_base = os.path.join(config.PLOT_OUTPUT_FOLDER, f"plot_agg_metric_{metric}_comparison") # Base name
                     try:
                         if hasattr(plotting, 'plot_metric_distributions'):
                             plotting.plot_metric_distributions(plot_summary_df, metric, plot_title, fname_base) # Pass base name
                         else: logger.warning(f"Plot function 'plot_metric_distributions' missing.")
                     except Exception as e_plot: logger.error(f"Error generating metric plot for {metric}: {e_plot}")
                 else: logger.warning(f"Metric {metric}_mean not found in summary.")

            # --- Plot ROC --- (Generates separate plots per mode)
            logger.info("Generating separate ROC plots per mode.")
            for mode in plot_summary_df['Mode'].unique(): # Use modes present in valid results
                mode_roc_data = all_runs_roc_data.get(mode)
                mode_metrics = all_runs_metrics.get(mode)
                if not mode_roc_data or not mode_metrics: continue

                configs_tasks_for_mode_roc = []
                mode_plot_summary = plot_summary_df[plot_summary_df['Mode'] == mode]
                for idx, row in mode_plot_summary.iterrows():
                     config_name = row['Config_Name']; task_name = row['Task_Name']
                     if config_name in mode_roc_data and task_name in mode_roc_data[config_name]:
                           if any(mode_roc_data[config_name][task_name]):
                               configs_tasks_for_mode_roc.append((config_name, task_name))

                if configs_tasks_for_mode_roc:
                     plot_title_roc = f'Aggregated ROC Curves ({mode.upper()} Split, {config.N_REPETITIONS} Reps)'
                     fname_roc = os.path.join(config.PLOT_OUTPUT_FOLDER, f"plot_aggregated_roc_curves_{mode}.png")
                     try:
                          if hasattr(plotting, 'plot_aggregated_roc_curves'):
                              plotting.plot_aggregated_roc_curves(mode_roc_data, mode_metrics, configs_tasks_for_mode_roc, plot_title_roc, fname_roc)
                          else: logger.warning(f"Plot function 'plot_aggregated_roc_curves' missing.")
                     except Exception as e_plot_roc: logger.error(f"Error generating ROC plot for mode {mode}: {e_plot_roc}")
                else: logger.warning(f"No valid ROC data found to plot for mode '{mode}'.")


            # --- Plot Importances --- (Generates separate plots per mode/config/task)
            logger.info("Generating separate importance plots per mode/config/task.")
            # Assumes results_processor returns the mode->config->task structure
            all_aggregated_imps = {**agg_base_importances, **agg_meta_importances}

            if not all_aggregated_imps:
                 logger.warning("Skipping importance plots: No aggregated importance data found.")
            else:
                 for mode, mode_data in all_aggregated_imps.items():
                     for config_name_imp, config_imps_dict in mode_data.items():
                          for task_name_imp, imp_df in config_imps_dict.items():
                              if imp_df is not None and not imp_df.empty:
                                   try:
                                       n_valid_runs_series = imp_df.get('N_Valid_Runs'); n_runs_info = "N/A"
                                       if n_valid_runs_series is not None and not n_valid_runs_series.empty:
                                            n_min, n_max = int(n_valid_runs_series.min()), int(n_valid_runs_series.max())
                                            n_runs_info = f"{n_min}" if n_min == n_max else f"{n_min}-{n_max}"

                                       is_meta = task_name_imp == 'meta_coeffs'

                                       if is_meta: plot_label = f"Meta: {config_name_imp}"; task_plot_name = "meta_coeffs"
                                       else: plot_label = f"Base: {config_name_imp} ({task_name_imp.upper()})"; task_plot_name = task_name_imp

                                       title = f'Top {config.PLOT_TOP_N_FEATURES} Features ({mode.upper()} Split)\n{plot_label} ({n_runs_info} Runs)'
                                       fname = os.path.join(config.PLOT_OUTPUT_FOLDER, f"plot_importance_{mode}_{config_name_imp}_{task_plot_name}.png")
                                       top_n_plot = len(imp_df) if is_meta else config.PLOT_TOP_N_FEATURES

                                       if hasattr(plotting, 'plot_aggregated_importances'):
                                            plotting.plot_aggregated_importances(imp_df, config_name_imp, task_name_imp, top_n_plot, title, fname)
                                       else: logger.warning(f"Plot function 'plot_aggregated_importances' missing.")
                                   except Exception as e_plot_imp: logger.error(f"Error generating importance plot for {mode}/{config_name_imp}/{task_name_imp}: {e_plot_imp}")

logger.info(f"\n--- Script Finished ({time.time() - start_time_script:.1f} seconds Total) ---")