# -*- coding: utf-8 -*-
"""
Main script to run DatScan prediction experiments for BINARY classification.
This script orchestrates data loading, preprocessing, model training,
tuning, evaluation, and results aggregation for various configurations
defined in 'prediction.config'.

It supports:
- Focusing on specific kinematic tasks (FT/HM).
- Iterating or focusing on specific DaTscan target regions (Contralateral/Ipsilateral _Z or _Raw).
- Iterating or focusing on specific threshold values (Z-scores or Raw values).
Results for each run are saved into a timestamped subfolder.
"""

import os
import sys
import pandas as pd
import numpy as np
import collections
import time
import logging
import datetime

# --- Path Setup for 'prediction' Package ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
if current_script_dir not in sys.path:
    sys.path.insert(0, current_script_dir)

# --- Configuration Import ---
try:
    from prediction import config
    print(f"INFO: Using BINARY configuration from prediction.{config.__name__}.py")
except ImportError as e_config:
    print(f"FATAL ERROR: Could not import 'prediction.config'. Error: {e_config}")
    sys.exit(1)

# --- Standard Library & Third-Party Imports ---
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    StratifiedGroupKFold, RandomizedSearchCV, cross_val_predict, GridSearchCV
)
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFE

# --- Import from 'prediction' Package ---
try:
    from prediction import utils
    from prediction import data_loader
    from prediction import pipeline_builder
    from prediction import evaluation
    from prediction import results_processor # Original processor
    from prediction import plotting # Original plotting functions
    print("Successfully imported all required modules from 'prediction' package.")
except ImportError as e_pred_pkg:
    print(f"ERROR: Failed to import from 'prediction' package: {e_pred_pkg}")
    raise e_pred_pkg

# --- Experiment Control Flags ---
RUN_FT_MODELS = False # True to run Finger Tapping model configurations
RUN_HM_MODELS = True  # True to run Hand Movement model configurations

ITERATE_DATSCAN_REGIONS = True # True to iterate all found _Z/_Raw regions, False to use FOCUS_REGIONS_TO_TEST
ITERATE_TARGET_TYPE = "RAW_VALUES_ONLY"  # Options: "Z_SCORES_ONLY", "RAW_VALUES_ONLY", "Z_SCORES_AND_RAW"
ITERATE_THRESHOLDS = True    # True to iterate Z/Raw thresholds, False to use FOCUS_threshold

# --- Focused Run Parameters (used when iteration flags are False) ---
FOCUS_DATSCAN_REGIONS_TO_TEST = ["Contralateral_Caudate_Raw", "Ipsilateral_Caudate_Raw"]
FOCUS_Z_SCORE_VALUE = -2.50
FOCUS_RAW_VALUE = 1.75 # Example: Adjust if using raw value targets in focus mode

# --- Timestamp for this Run ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Create Timestamped Output Subfolders ---
run_output_data_folder = os.path.join(config.DATA_OUTPUT_FOLDER, timestamp)
run_output_plot_folder = os.path.join(config.PLOT_OUTPUT_FOLDER, timestamp)
try:
    os.makedirs(run_output_data_folder, exist_ok=True)
    if config.GENERATE_PLOTS: os.makedirs(run_output_plot_folder, exist_ok=True)
except OSError as e_dir: print(f"FATAL: Error creating timestamped output subdirs: {e_dir}."); sys.exit(1)

# --- Logging Configuration ---
log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(funcName)s:%(lineno)d] - %(message)s'
log_level = logging.INFO

run_type_descriptor = "Run"
if RUN_FT_MODELS and RUN_HM_MODELS: run_type_descriptor+="_FT_HM"
elif RUN_FT_MODELS: run_type_descriptor+="_FT_Only"
elif RUN_HM_MODELS: run_type_descriptor+="_HM_Only"

if ITERATE_DATSCAN_REGIONS: run_type_descriptor += "_IterRegions"
else: run_type_descriptor += "_FocusRegions"

if ITERATE_TARGET_TYPE == "Z_SCORES_ONLY": run_type_descriptor += "_TargetZ"
elif ITERATE_TARGET_TYPE == "RAW_VALUES_ONLY": run_type_descriptor += "_TargetRaw"
elif ITERATE_TARGET_TYPE == "Z_SCORES_AND_RAW": run_type_descriptor += "_TargetZRaw"

if ITERATE_THRESHOLDS: run_type_descriptor += "_IterThresh"
else: run_type_descriptor += "_FocusThresh"
if not ITERATE_DATSCAN_REGIONS and not ITERATE_THRESHOLDS and \
   (ITERATE_TARGET_TYPE == "Z_SCORES_ONLY" and FOCUS_Z_SCORE_VALUE == config.ABNORMALITY_THRESHOLD and \
    FOCUS_DATSCAN_REGIONS_TO_TEST == [config.TARGET_Z_SCORE_COL]):
    run_type_descriptor = "Run_SingleDefaultTarget" # More specific for the most basic run

log_filename_base = f"experiment_log_{run_type_descriptor}.log"
log_filename = os.path.join(run_output_data_folder, log_filename_base)

summary_filename_prefix = f"summary_{run_type_descriptor}_"
best_summary_filename_prefix = f"best_auc_summary_{run_type_descriptor}_"
plot_filename_prefix_iter = f"plot_iter_{run_type_descriptor}_" # Used for conditional plotting

logger = logging.getLogger('DatnikExperiment'); logger.setLevel(log_level)
if logger.hasHandlers(): logger.handlers.clear()
ch = logging.StreamHandler(sys.stdout); ch.setLevel(log_level); ch_formatter = logging.Formatter(log_format); ch.setFormatter(ch_formatter); logger.addHandler(ch)
try:
    fh = logging.FileHandler(log_filename, mode='w', encoding='utf-8'); fh.setLevel(log_level); fh_formatter = logging.Formatter(log_format); fh.setFormatter(fh_formatter); logger.addHandler(fh)
    logger.info(f"Logging configured. Log file: {log_filename}")
except Exception as e_log: logger.error(f"Warning: File logging to {log_filename} failed: {e_log}")

# --- Basic Setup ---
start_time_script = time.time(); utils.setup_warnings(); sns.set_theme(style="whitegrid")
logger.info(f"========== SCRIPT START: DATNIK PREDICTION ({run_type_descriptor}) ==========")
logger.info(f"Run Timestamp: {timestamp}")
logger.info(f"Output Data Subfolder: {run_output_data_folder}")
if config.GENERATE_PLOTS: logger.info(f"Output Plot Subfolder: {run_output_plot_folder}")
logger.info(f"Config: prediction.{config.__name__}.py")
logger.info(f"Run FT Models: {RUN_FT_MODELS}, Run HM Models: {RUN_HM_MODELS}")
logger.info(f"Iterate DaTscan Regions: {ITERATE_DATSCAN_REGIONS}, Target Type: {ITERATE_TARGET_TYPE}, Iterate Thresholds: {ITERATE_THRESHOLDS}")
if not ITERATE_DATSCAN_REGIONS: logger.info(f"Focus Regions: {FOCUS_DATSCAN_REGIONS_TO_TEST}")
if not ITERATE_THRESHOLDS: logger.info(f"Focus Z-Score: {FOCUS_Z_SCORE_VALUE}, Focus Raw Value: {FOCUS_RAW_VALUE} (if Raw target type used)")

# --- Define Threshold Value Lists ---
RAW_DATA_THRESHOLDS_TO_TEST = np.round(np.arange(0.75, 3.01, 0.25), 3) 
Z_SCORE_THRESHOLDS_TO_TEST = np.round(np.arange(-3.0, -0.49, 0.25), 3) 
MIN_SAMPLES_PER_CLASS = 10
if ITERATE_THRESHOLDS: logger.info(f"Min samples per class for threshold iteration: {MIN_SAMPLES_PER_CLASS}")

# --- Load Raw Data (ONCE) ---
logger.info("Loading initial raw data (df_raw)...")
try: df_raw = data_loader.load_data(config.INPUT_FOLDER, config.INPUT_CSV_NAME)
except Exception as e_data_raw: logger.exception("FATAL: Error loading raw data."); sys.exit(1)

# --- Filter config.CONFIGURATIONS_TO_RUN based on RUN_FT_MODELS and RUN_HM_MODELS ---
active_configurations_to_run = []
if RUN_FT_MODELS: active_configurations_to_run.extend([c for c in config.CONFIGURATIONS_TO_RUN if c['task_prefix_for_features'].lower() == 'ft'])
if RUN_HM_MODELS: active_configurations_to_run.extend([c for c in config.CONFIGURATIONS_TO_RUN if c['task_prefix_for_features'].lower() == 'hm'])
if not active_configurations_to_run: logger.error("FATAL: No model configurations selected."); sys.exit(1)
logger.info(f"Active model configurations: {[c['config_name'] for c in active_configurations_to_run]}")

# --- Identify Target DaTscan Columns to Process ---
datscan_target_columns_to_process = [] # List of dicts: {"name": col_name, "type": "Z_SCORE" or "RAW"}
if ITERATE_DATSCAN_REGIONS:
    if ITERATE_TARGET_TYPE == "Z_SCORES_ONLY" or ITERATE_TARGET_TYPE == "Z_SCORES_AND_RAW":
        for col in df_raw.columns:
            if (col.startswith("Contralateral_") or col.startswith("Ipsilateral_")) and col.endswith("_Z"):
                if col in df_raw.columns: datscan_target_columns_to_process.append({"name": col, "type": "Z_SCORE"})
    if ITERATE_TARGET_TYPE == "RAW_VALUES_ONLY" or ITERATE_TARGET_TYPE == "Z_SCORES_AND_RAW":
        for col in df_raw.columns:
            if (col.startswith("Contralateral_") or col.startswith("Ipsilateral_")) and col.endswith("_Raw"):
                if col in df_raw.columns: datscan_target_columns_to_process.append({"name": col, "type": "RAW"})
    if not datscan_target_columns_to_process:
        logger.warning("ITERATE_DATSCAN_REGIONS/TARGET_TYPE set, but no matching cols found. Defaulting to config target.")
        datscan_target_columns_to_process = [{"name": config.TARGET_Z_SCORE_COL, "type": "Z_SCORE"}]
else: # Not iterating regions, use FOCUS_DATSCAN_REGIONS_TO_TEST
    for region_name in FOCUS_DATSCAN_REGIONS_TO_TEST:
        if region_name not in df_raw.columns: logger.warning(f"Focus region '{region_name}' not in df_raw. Skipping."); continue
        target_type_inferred = "UNKNOWN"
        if region_name.endswith("_Z"): target_type_inferred = "Z_SCORE"
        elif region_name.endswith("_Raw"): target_type_inferred = "RAW"
        
        if target_type_inferred == "Z_SCORE" and (ITERATE_TARGET_TYPE == "Z_SCORES_ONLY" or ITERATE_TARGET_TYPE == "Z_SCORES_AND_RAW"):
            datscan_target_columns_to_process.append({"name": region_name, "type": "Z_SCORE"})
        elif target_type_inferred == "RAW" and (ITERATE_TARGET_TYPE == "RAW_VALUES_ONLY" or ITERATE_TARGET_TYPE == "Z_SCORES_AND_RAW"):
            datscan_target_columns_to_process.append({"name": region_name, "type": "RAW"})
        elif target_type_inferred == "UNKNOWN":
             logger.warning(f"Focus region '{region_name}' suffix not _Z or _Raw. Assuming Z-score if Z_SCORES_ONLY/Z_SCORES_AND_RAW.")
             if (ITERATE_TARGET_TYPE == "Z_SCORES_ONLY" or ITERATE_TARGET_TYPE == "Z_SCORES_AND_RAW"):
                 datscan_target_columns_to_process.append({"name": region_name, "type": "Z_SCORE"}) # Best guess

if not datscan_target_columns_to_process: logger.error("FATAL: No DaTscan target columns selected/found."); sys.exit(1)
logger.info(f"Effective DaTscan target columns for this run: {[item['name'] for item in datscan_target_columns_to_process]}")

# --- Initialize Results Storage ---
results_data_metrics = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(
    lambda: collections.defaultdict(lambda: collections.defaultdict(list)))))
results_data_roc = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(
    lambda: collections.defaultdict(lambda: collections.defaultdict(list)))))
results_data_importances = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(
    lambda: collections.defaultdict(lambda: collections.defaultdict(list)))))
results_data_rfe = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(
    lambda: collections.defaultdict(lambda: collections.defaultdict(list)))))

# --- Derive Task Prefixes (ONCE) ---
defined_task_prefixes_main_script = sorted(list(set(exp_conf.get('task_prefix_for_features') for exp_conf in active_configurations_to_run))) # Use active_configurations_to_run
if not defined_task_prefixes_main_script: logger.error("FATAL: No task prefixes from active_configurations_to_run."); sys.exit(1)
logger.info(f"Kinematic task prefixes from active configs: {defined_task_prefixes_main_script}")

# --- MAIN EXPERIMENT LOOPS ---
for target_info in datscan_target_columns_to_process:
    target_col_name_key = target_info["name"] # For dict key
    target_col_type_key = target_info["type"] # For dict key

    logger.info(f"\n\n\n========== PROCESSING TARGET DATSCAN COLUMN: {target_col_name_key} (Type: {target_col_type_key}) ==========")
    
    current_threshold_list_to_use = [] # Initialize as an empty list
    if ITERATE_THRESHOLDS:
        if target_col_type_key == "Z_SCORE": 
            current_threshold_list_to_use = Z_SCORE_THRESHOLDS_TO_TEST
        elif target_col_type_key == "RAW": 
            current_threshold_list_to_use = RAW_DATA_THRESHOLDS_TO_TEST
    else: # Corresponds to 'if ITERATE_THRESHOLDS:'
        if target_col_type_key == "Z_SCORE": 
            current_threshold_list_to_use = [FOCUS_Z_SCORE_VALUE]
        elif target_col_type_key == "RAW": 
            current_threshold_list_to_use = [FOCUS_RAW_VALUE]
    
    if not hasattr(current_threshold_list_to_use, '__len__') or len(current_threshold_list_to_use) == 0:
        logger.warning(f"No thresholds defined or list is empty for target {target_col_name_key} (type {target_col_type_key}). Skipping this target column.")
        continue 
    
    logger.info(f"    Thresholds for {target_col_name_key}: {current_threshold_list_to_use}")

    for current_threshold_val in current_threshold_list_to_use:
        current_threshold_val = float(current_threshold_val)
        thresh_val_key = round(current_threshold_val, 3)
        logger.info(f"\n\n  ======== THRESHOLD VALUE: {thresh_val_key:.3f} (Column: {target_col_name_key}) ========")
        
        try:
            X_full_glob, y_full_iter, groups_full_iter, task_feat_map_eng, _ = \
                data_loader.prepare_data(df_raw, config, 
                                         target_z_score_column_override=target_col_name_key,
                                         abnormality_threshold_override=current_threshold_val)
        except Exception as e_prep: 
            logger.exception(f"Err prep data for {target_col_name_key}, Thr {thresh_val_key:.3f}. Skip.")
            continue
        
        if y_full_iter.empty or X_full_glob.empty : 
            logger.warning(f"Prep for {target_col_name_key}, Thr {thresh_val_key:.3f} empty y/X. Skip.")
            continue
        if y_full_iter.nunique() < 2: 
            logger.warning(f"{target_col_name_key}, Thr {thresh_val_key:.3f} -> <2 classes. Skip.")
            continue
        
        class_counts_iter = y_full_iter.value_counts()
        logger.info(f"Class dist for {target_col_name_key}, Thr {thresh_val_key:.3f}: {class_counts_iter.to_dict()}")
        if (ITERATE_THRESHOLDS and hasattr(current_threshold_list_to_use, '__len__') and len(current_threshold_list_to_use) > 1) and \
           (class_counts_iter.min() < MIN_SAMPLES_PER_CLASS):
            logger.warning(f"{target_col_name_key}, Thr {thresh_val_key:.3f} has class < {MIN_SAMPLES_PER_CLASS}. Skip Thr.")
            continue
        
        task_feat_map_orig = {}
        for T_prefix in defined_task_prefixes_main_script:
            task_feat_map_orig[T_prefix] = sorted([f"{T_prefix}_{b}" for b in config.BASE_KINEMATIC_COLS if f"{T_prefix}_{b}" in X_full_glob.columns])

        for split_mode in config.SPLITTING_MODES_TO_RUN:
            logger.info(f"\n    Mode: {split_mode.upper()} (Target: {target_col_name_key} @ {thresh_val_key:.3f})")
            for i_rep in range(config.N_REPETITIONS):
                rand_state = config.BASE_RANDOM_STATE + i_rep
                logger.info(f"      Rep {i_rep+1}/{config.N_REPETITIONS} (Seed: {rand_state})")
                try:
                    unique_patients_iter = groups_full_iter.unique()
                    n_patients_iter = len(unique_patients_iter)
                    min_test_p, max_test_p = 1, n_patients_iter - config.N_SPLITS_CV
                    if max_test_p < min_test_p: max_test_p = min_test_p 
                    
                    if n_patients_iter < config.N_SPLITS_CV + min_test_p: 
                        logger.error(f"Rep {i_rep+1}: Not enough patients ({n_patients_iter}) for CV ({config.N_SPLITS_CV}) and min test ({min_test_p}). Skip rep.")
                        continue
                        
                    n_test_p_ideal = int(np.ceil(n_patients_iter * config.TEST_SET_SIZE))
                    n_test_p = max(min_test_p, min(n_test_p_ideal, max_test_p))
                    
                    rng_i = np.random.RandomState(rand_state)
                    shuffled_p_i = rng_i.permutation(unique_patients_iter)
                    test_p_ids_i, train_val_p_ids_i = set(shuffled_p_i[:n_test_p]), set(shuffled_p_i[n_test_p:])
                    
                    train_val_mask = groups_full_iter.isin(train_val_p_ids_i)
                    test_mask = groups_full_iter.isin(test_p_ids_i)
                    
                    X_train_val, y_train_val, groups_train_val = X_full_glob.loc[train_val_mask].copy(), y_full_iter.loc[train_val_mask].copy(), groups_full_iter.loc[train_val_mask].copy()
                    X_test, y_test = X_full_glob.loc[test_mask].copy(), y_full_iter.loc[test_mask].copy()
                    
                    n_unique_g_train = groups_train_val.nunique()
                    actual_n_splits_cv = min(config.N_SPLITS_CV, n_unique_g_train)
                    
                    inner_cv = None
                    if actual_n_splits_cv >= 2:
                        inner_cv = StratifiedGroupKFold(n_splits=actual_n_splits_cv, shuffle=True, random_state=rand_state)
                    
                    if X_train_val.empty or X_test.empty or y_train_val.nunique()<2 or y_test.nunique()<1: 
                        logger.warning("Outer split issue (empty data or single class). Skip rep.")
                        continue
                    
                    for exp_conf in active_configurations_to_run: 
                        cfg_name = exp_conf['config_name']
                        task_pfx = exp_conf['task_prefix_for_features']
                        apply_fe_cfg = exp_conf.get('apply_feature_engineering',False)
                        
                        current_cols_cfg = task_feat_map_eng.get(task_pfx, []) if apply_fe_cfg else task_feat_map_orig.get(task_pfx, [])
                        
                        if not current_cols_cfg: 
                            results_data_metrics[target_col_name_key][target_col_type_key][thresh_val_key][split_mode][cfg_name].append({})
                            continue
                            
                        valid_train_c = [c for c in current_cols_cfg if c in X_train_val.columns]
                        valid_test_c = [c for c in current_cols_cfg if c in X_test.columns]
                        
                        if not valid_train_c: 
                            results_data_metrics[target_col_name_key][target_col_type_key][thresh_val_key][split_mode][cfg_name].append({})
                            continue
                            
                        X_tr_task = X_train_val[valid_train_c].copy()
                        X_tst_task = X_test[valid_test_c].copy() if valid_test_c else pd.DataFrame()
                        y_tst_task = y_test.loc[X_tst_task.index] if not X_tst_task.empty else pd.Series(dtype=y_test.dtype)
                        
                        best_p, tune_s, fit_s = None, None, False
                        try:
                            pipe_cfg, srch_prms = pipeline_builder.build_pipeline_from_config(exp_conf, rand_state, config)
                            if not pipe_cfg.steps: raise ValueError("Empty pipeline")
                            
                            if inner_cv is None: 
                                best_p = clone(pipe_cfg)
                                best_p.fit(X_tr_task, y_train_val)
                                fit_s = True
                            else: 
                                if config.ENABLE_TUNING and srch_prms:
                                    SCV_cls = RandomizedSearchCV if exp_conf.get('search_type','random')=='random' else GridSearchCV
                                    scv_kw = {'estimator':pipe_cfg, 'scoring':config.TUNING_SCORING_METRIC, 
                                              'cv':inner_cv, 'n_jobs':-1, 'refit':True, 
                                              'error_score':np.nan} # MODIFIED for robustness
                                    if exp_conf.get('search_type','random')=='random': 
                                        scv_kw.update({'param_distributions':srch_prms, 'n_iter':config.N_ITER_RANDOM_SEARCH, 'random_state':rand_state})
                                    else: 
                                        scv_kw['param_grid'] = srch_prms
                                    
                                    scv_m = SCV_cls(**scv_kw)
                                    scv_m.fit(X_tr_task, y_train_val, groups=groups_train_val)
                                    best_p, tune_s, fit_s = scv_m.best_estimator_, scv_m.best_score_, True
                                else: 
                                    best_p = clone(pipe_cfg)
                                    best_p.fit(X_tr_task, y_train_val)
                                    fit_s=True
                                    
                            if fit_s:
                                if exp_conf.get('feature_selector')=='rfe' and best_p and 'feature_selector' in best_p.named_steps: # Added best_p check
                                    rfe_s = best_p.named_steps['feature_selector']
                                    if isinstance(rfe_s,RFE) and hasattr(rfe_s,'support_'):
                                        try: 
                                            rfe_n = X_tr_task.columns[rfe_s.support_].tolist()
                                            results_data_rfe[target_col_name_key][target_col_type_key][thresh_val_key][split_mode][cfg_name].append(rfe_n)
                                        except IndexError as ie:
                                            logger.warning(f"IndexError extracting RFE features for {cfg_name}. Support len: {len(rfe_s.support_)}, X_tr_task cols: {len(X_tr_task.columns)}. Error: {ie}")
                                        except Exception as e_rfe_extract:
                                            logger.warning(f"Error extracting RFE features for {cfg_name}: {e_rfe_extract}")
                                            
                                if X_tst_task.empty or y_tst_task.nunique()<1: 
                                    nan_m={m:np.nan for m in['roc_auc','f1_macro']}
                                    nan_m['best_cv_score']=tune_s if tune_s is not None else np.nan
                                    results_data_metrics[target_col_name_key][target_col_type_key][thresh_val_key][split_mode][cfg_name].append(nan_m)
                                else:
                                    preds_t = best_p.predict(X_tst_task)
                                    probs_t = best_p.predict_proba(X_tst_task)[:,1]
                                    t_met, t_roc = evaluation.evaluate_predictions(y_tst_task, preds_t, probs_t)
                                    t_met['best_cv_score']= tune_s if tune_s is not None else np.nan
                                    results_data_metrics[target_col_name_key][target_col_type_key][thresh_val_key][split_mode][cfg_name].append(t_met)
                                    if t_roc: 
                                        results_data_roc[target_col_name_key][target_col_type_key][thresh_val_key][split_mode][cfg_name].append(t_roc)
                                        
                                if best_p: # Ensure best_p exists before getting importances
                                    imps_s = utils.get_feature_importances(best_p, X_tr_task.columns)
                                    if imps_s is not None: 
                                        results_data_importances[target_col_name_key][target_col_type_key][thresh_val_key][split_mode][cfg_name].append(imps_s)
                            else: # fit_s is False, meaning tuning might have failed entirely or was skipped
                                logger.warning(f"Fit was not successful for {cfg_name} (Seed: {rand_state}, Target: {target_col_name_key} @ {thresh_val_key:.3f}). Skipping RFE/Importance/Metric storage for this run.")
                                results_data_metrics[target_col_name_key][target_col_type_key][thresh_val_key][split_mode][cfg_name].append({'roc_auc':np.nan, 'best_cv_score': np.nan})


                        except ValueError as ve:
                            if "only one class present" in str(ve).lower() or \
                               "must be >= 2" in str(ve).lower() or \
                               ("n_splits=" in str(ve).lower() and "cannot be greater than the number of members in each class." in str(ve).lower()) or \
                               ("Invalid parameter for estimator Pipeline" in str(ve) and "Check the list of available parameters with `estimator.get_params().keys()`" in str(ve)) : 
                                logger.warning(f"Handled ValueError for {cfg_name} (Seed: {rand_state}, Target: {target_col_name_key} @ {thresh_val_key:.3f}): {ve}")
                                results_data_metrics[target_col_name_key][target_col_type_key][thresh_val_key][split_mode][cfg_name].append({'roc_auc':np.nan, 'best_cv_score': np.nan})
                            else: 
                                logger.exception(f"Unhandled ValueError in config {cfg_name} (Seed: {rand_state}, Target: {target_col_name_key} @ {thresh_val_key:.3f}):")
                                results_data_metrics[target_col_name_key][target_col_type_key][thresh_val_key][split_mode][cfg_name].append({})
                        except Exception as e_mdl: 
                            logger.exception(f"Error during model processing for config {cfg_name} (Seed: {rand_state}, Target: {target_col_name_key} @ {thresh_val_key:.3f}):")
                            results_data_metrics[target_col_name_key][target_col_type_key][thresh_val_key][split_mode][cfg_name].append({})
                except Exception as e_r: 
                    logger.exception(f"Error in Repetition {i_rep+1} (Seed: {rand_state}, Target: {target_col_name_key} @ {thresh_val_key:.3f}):")
                    for exp_conf_err in active_configurations_to_run:
                        results_data_metrics[target_col_name_key][target_col_type_key][thresh_val_key][split_mode][exp_conf_err['config_name']].append({})
                    continue 
    logger.info(f"\n========== FINISHED TARGET: {target_col_name_key} ==========")


logger.info(f"\n--- All Processing Finished ({run_type_descriptor}) ---")

# --- Aggregate and Save Results ---
logger.info("\n--- Aggregating Final Results (Metrics) ---")
# ... (Metrics aggregation and saving - kept the same as your last provided version) ...
summary_list = []
for target_col_name, target_col_data in results_data_metrics.items():
    for target_type_val, type_data_val in target_col_data.items():
        for thresh_val, thresh_data_val in type_data_val.items():
            for mode_val, mode_data_val in thresh_data_val.items():
                for config_name_val, metric_lists_val in mode_data_val.items():
                    task_name_val = utils.get_task_from_config_name(config_name_val)
                    if not metric_lists_val: continue
                    valid_metrics = [m for m in metric_lists_val if isinstance(m, dict) and m]
                    if not valid_metrics: continue
                    df_metrics = pd.DataFrame(valid_metrics)
                    if df_metrics.empty: continue
                    
                    mean_roc_auc = df_metrics['roc_auc'].mean() if 'roc_auc' in df_metrics.columns else np.nan
                    std_roc_auc = df_metrics['roc_auc'].std() if 'roc_auc' in df_metrics.columns else np.nan
                    mean_f1_macro = df_metrics['f1_macro'].mean() if 'f1_macro' in df_metrics.columns else np.nan
                    std_f1_macro = df_metrics['f1_macro'].std() if 'f1_macro' in df_metrics.columns else np.nan
                    mean_best_cv_score = np.nan
                    if 'best_cv_score' in df_metrics.columns and df_metrics['best_cv_score'].notna().any():
                        mean_best_cv_score = df_metrics['best_cv_score'].mean()
                    
                    n_valid_runs = 0
                    if 'roc_auc' in df_metrics.columns:
                        n_valid_runs_metric = df_metrics['roc_auc'].notna().sum()
                        n_valid_runs = int(n_valid_runs_metric) 
                    elif not df_metrics.empty: 
                        n_valid_runs = len(df_metrics)

                    summary_list.append({
                        'Datscan_Target_Column': target_col_name,
                        'Datscan_Target_Type': target_type_val,
                        'Threshold_Value': thresh_val,
                        'Mode': mode_val,
                        'Config_Name': config_name_val, 
                        'Task_Name': task_name_val, 
                        'N_Valid_Runs': n_valid_runs,
                        'Mean_ROC_AUC': mean_roc_auc,
                        'Std_ROC_AUC': std_roc_auc,
                        'Mean_F1_Macro': mean_f1_macro,
                        'Std_F1_Macro': std_f1_macro,
                        'Mean_Best_CV_Score': mean_best_cv_score,
                    })
final_summary_df = pd.DataFrame(summary_list) if summary_list else pd.DataFrame()

if not final_summary_df.empty:
    s_fname = os.path.join(run_output_data_folder, f"{summary_filename_prefix}{timestamp}.csv")
    final_summary_df.to_csv(s_fname, index=False, sep=';', decimal='.', float_format='%.5f')
    logger.info(f"Final Summary saved: {s_fname}")

    if (ITERATE_DATSCAN_REGIONS or ITERATE_THRESHOLDS or ITERATE_TARGET_TYPE != "Z_SCORES_ONLY") and \
       best_summary_filename_prefix and 'Mean_ROC_AUC' in final_summary_df.columns:
        logger.info("\n--- Best Performing Settings (based on Mean ROC AUC) ---")
        df_for_best_agg = final_summary_df.dropna(subset=['Mean_ROC_AUC'])
        if not df_for_best_agg.empty:
            group_keys_base = ['Mode', 'Config_Name'] 
            
            conditional_group_keys = []
            if ITERATE_DATSCAN_REGIONS: 
                conditional_group_keys.append('Datscan_Target_Column')
            if ITERATE_TARGET_TYPE == "Z_SCORES_AND_RAW" or \
               ( (ITERATE_TARGET_TYPE == "RAW_VALUES_ONLY" or ITERATE_TARGET_TYPE == "Z_SCORES_ONLY") and \
                 (ITERATE_DATSCAN_REGIONS or (not ITERATE_DATSCAN_REGIONS and len(FOCUS_DATSCAN_REGIONS_TO_TEST) > 1)) ): # Check if type was a choice point
                 conditional_group_keys.append('Datscan_Target_Type')
            
            final_group_keys = conditional_group_keys + group_keys_base
            final_group_keys = sorted(list(set(final_group_keys)), key=final_group_keys.index)

            if not final_group_keys: 
                final_group_keys = group_keys_base 

            try:
                best_summary = df_for_best_agg.loc[df_for_best_agg.groupby(final_group_keys, dropna=False)['Mean_ROC_AUC'].idxmax()]
                
                log_cols_order = ['Datscan_Target_Column','Datscan_Target_Type', 'Threshold_Value', 'Mode', 
                                  'Config_Name', 'Task_Name', 'Mean_ROC_AUC', 'N_Valid_Runs']
                log_cols = [k for k in log_cols_order if k in best_summary.columns] 
                logger.info(f"\n{best_summary[log_cols].to_string()}") 
                
                bs_fname = os.path.join(run_output_data_folder, f"{best_summary_filename_prefix}{timestamp}.csv")
                best_summary.to_csv(bs_fname, index=False, sep=';', decimal='.', float_format='%.5f')
                logger.info(f"Best settings summary saved: {bs_fname}")
            except Exception as e_b_sum: 
                logger.error(f"Error creating best settings summary: {e_b_sum}", exc_info=True)
else: 
    logger.warning("No data for final summary.")


# --- Aggregate and Save RFE Features ---
logger.info("\n--- Aggregating RFE Selected Features ---")
if results_data_rfe:
    for target_col_name, target_data in results_data_rfe.items():
        for target_type, type_data in target_data.items():
            for thresh_val, thresh_data in type_data.items():
                for mode_val, mode_data in thresh_data.items():
                    for config_name_val, rfe_lists_val in mode_data.items():
                        if not rfe_lists_val:
                            continue
                        temp_rfe_data_for_processor = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
                        base_task_name = utils.get_task_from_config_name(config_name_val)
                        sanitized_target_col = target_col_name.replace('_Raw','R').replace('_Z','Z').replace('Contralateral_','C_').replace('Ipsilateral_','I_')
                        sanitized_thresh_val = str(thresh_val).replace('.','p')
                        specific_task_name_for_file = f"{base_task_name}_{sanitized_target_col}_{target_type}_Thr{sanitized_thresh_val}"
                        
                        temp_rfe_data_for_processor[mode_val][config_name_val][specific_task_name_for_file] = rfe_lists_val
                        
                        logger.info(f"Aggregating RFE for: {config_name_val} - {specific_task_name_for_file} (Mode: {mode_val})")
                        
                        # MODIFIED: Pass run_output_data_folder as output_dir_override
                        results_processor.aggregate_rfe_features(
                            temp_rfe_data_for_processor, 
                            config, 
                            output_dir_override=run_output_data_folder 
                        )
else:
    logger.info("No RFE data collected (results_data_rfe is empty).")


# --- Plotting ---
if config.GENERATE_PLOTS:
    logger.info("\n--- Generating Plots ---")
    if final_summary_df.empty: 
        logger.warning("No summary data to plot.")
    else:
        # Scenario 1: Iterating regions AND thresholds -> Heatmaps
        if ITERATE_DATSCAN_REGIONS and ITERATE_THRESHOLDS and plot_filename_prefix_iter:
            target_types_in_summary = final_summary_df['Datscan_Target_Type'].unique()
            for target_type_plot in target_types_in_summary:
                logger.info(f"Generating Heatmaps for {target_type_plot} targets...")
                df_type_subset = final_summary_df[final_summary_df['Datscan_Target_Type'] == target_type_plot]
                
                for cfg_name_plt in df_type_subset['Config_Name'].unique():
                    df_plot_c = df_type_subset[df_type_subset['Config_Name'] == cfg_name_plt]
                    if df_plot_c.empty or not all(c in df_plot_c.columns for c in ['Datscan_Target_Column', 'Threshold_Value', 'Mean_ROC_AUC']): 
                        logger.debug(f"Skipping heatmap for {cfg_name_plt} ({target_type_plot}): missing necessary columns or empty data.")
                        continue
                    try:
                        pivot = df_plot_c.pivot_table(index='Datscan_Target_Column', 
                                                      columns='Threshold_Value', 
                                                      values='Mean_ROC_AUC', 
                                                      dropna=False)
                        if pivot.empty: 
                            logger.debug(f"Skipping heatmap for {cfg_name_plt} ({target_type_plot}): pivot table is empty.")
                            continue
                            
                        plt.figure(figsize=(max(10,len(pivot.columns)*1.2), max(6,len(pivot.index)*0.7)))
                        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis_r", linewidths=.5, 
                                    cbar_kws={'label':'Mean ROC AUC'}, center=0.5, vmin=0.3, vmax=0.9)
                        plt.title(f"Mean ROC AUC: {cfg_name_plt} ({target_type_plot})\nTarget Region vs. Binarization Threshold", fontsize=14)
                        plt.ylabel("DaTscan Target Region",fontsize=12)
                        plt.xlabel(f"{target_type_plot} Binarization Threshold",fontsize=12)
                        plt.xticks(rotation=45,ha='right'); plt.yticks(rotation=0)
                        plt.tight_layout() 
                        
                        plot_f_name_safe = cfg_name_plt.replace('/','-').replace('\\','-') 
                        plot_f = os.path.join(run_output_plot_folder, f"{plot_filename_prefix_iter}heatmap_{plot_f_name_safe}_{target_type_plot}.png")
                        plt.savefig(plot_f,dpi=300,bbox_inches='tight'); plt.close()
                        logger.info(f"Saved heatmap: {plot_f}")
                    except Exception as e_hm: 
                        logger.error(f"Error generating heatmap for {cfg_name_plt} ({target_type_plot}): {e_hm}", exc_info=True)
        
        # MODIFIED: Scenario 2: Iterating ONLY thresholds (FOCUS_REGIONS is used) -> Line plots
        elif not ITERATE_DATSCAN_REGIONS and ITERATE_THRESHOLDS and plot_filename_prefix_iter:
            logger.info("Generating Line Plots (AUC vs. Threshold) for Focused Regions...")
            # Group by Datscan_Target_Column, Config_Name, and Datscan_Target_Type
            # Ensure these columns exist in final_summary_df before grouping
            grouping_cols = ['Datscan_Target_Column', 'Config_Name', 'Datscan_Target_Type']
            if not all(col in final_summary_df.columns for col in grouping_cols):
                logger.error(f"One or more grouping columns {grouping_cols} not in final_summary_df. Skipping line plots.")
            else:
                for (target_col, cfg_name, target_type_plot), group_df in final_summary_df.groupby(grouping_cols):
                    if group_df.empty or 'Mean_ROC_AUC' not in group_df.columns or 'Threshold_Value' not in group_df.columns:
                        continue
                    
                    plot_df = group_df.sort_values(by='Threshold_Value')
                    if plot_df['Mean_ROC_AUC'].notna().sum() < 2: 
                        logger.debug(f"Skipping line plot for {cfg_name}, {target_col} ({target_type_plot}): not enough data points with Mean_ROC_AUC.")
                        continue

                    plt.figure(figsize=(10, 6))
                    plt.plot(plot_df['Threshold_Value'], plot_df['Mean_ROC_AUC'], marker='o', linestyle='-', label=f"Mean ROC AUC")
                    
                    if 'Std_ROC_AUC' in plot_df.columns and 'N_Valid_Runs' in plot_df.columns:
                        valid_std_points = plot_df[(plot_df['N_Valid_Runs'] > 1) & plot_df['Std_ROC_AUC'].notna()]
                        if not valid_std_points.empty:
                            plt.fill_between(valid_std_points['Threshold_Value'], 
                                             valid_std_points['Mean_ROC_AUC'] - valid_std_points['Std_ROC_AUC'], 
                                             valid_std_points['Mean_ROC_AUC'] + valid_std_points['Std_ROC_AUC'], 
                                             alpha=0.2, label='±1 Std Dev (where N_Valid_Runs > 1)')

                    plt.title(f"Mean ROC AUC vs. Threshold\nConfig: {cfg_name}\nTarget: {target_col} ({target_type_plot})", fontsize=14)
                    plt.xlabel(f"{target_type_plot} Binarization Threshold", fontsize=12)
                    plt.ylabel("Mean ROC AUC", fontsize=12)
                    plt.ylim(0.2, 1.0) # Adjusted Y-axis for typical AUC range
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.legend()
                    plt.tight_layout()
                    
                    plot_f_name_safe_cfg = cfg_name.replace('/','-').replace('\\','-')
                    plot_f_name_safe_target = target_col.replace('/','-').replace('\\','-')
                    plot_f = os.path.join(run_output_plot_folder, f"{plot_filename_prefix_iter}lineplot_{plot_f_name_safe_cfg}_{plot_f_name_safe_target}_{target_type_plot}.png")
                    plt.savefig(plot_f, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"Saved line plot: {plot_f}")
        else:
            logger.info("Plotting conditions not met for detailed heatmaps or line plots. "
                        "Consider using generic plotting functions from plotting.py for other scenarios, "
                        "or check ITERATE_DATSCAN_REGIONS and ITERATE_THRESHOLDS flags.")


# --- Script End ---
end_time_script = time.time()
total_duration_seconds = end_time_script - start_time_script

logger.info(f"\n========== SCRIPT FINISHED ({run_type_descriptor}): Total execution time: {total_duration_seconds:.1f} seconds "
            f"({total_duration_seconds/60:.1f} minutes / {total_duration_seconds/3600:.2f} hours) ==========")