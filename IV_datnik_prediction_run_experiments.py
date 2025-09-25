# --- START OF FILE IV_datnik_prediction_run_experiments.py (FINAL PUBLICATION RUN WITH FIGURE 3 - COMPLETE) ---
# -*- coding: utf-8 -*-
"""
Main script to run DatScan prediction experiments for BINARY classification.
This script orchestrates data loading, preprocessing, model training,
tuning, evaluation, and results aggregation for various configurations
defined in 'prediction.config'.

NOTE: All analyses performed by this script are MANDATORILY AGE-CONTROLLED
as per the logic in data_loader.py.

--- MODIFICATION FOR PUBLICATION RUN ---
- Focuses only on the Finger Tapping task with original features.
- Sweeps Z-score thresholds from -1.5 to -2.0 to find the optimal cutoff.
- Generates a final, combined, multi-panel Figure 3 summarizing prediction results.
- Explicitly handles and reports cases where a threshold yields insufficient samples.
"""

import os
import sys
import pandas as pd
import numpy as np
import collections
import time
import logging
import datetime
import traceback 

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
import matplotlib.gridspec as gridspec

from sklearn.model_selection import (
    StratifiedGroupKFold, RandomizedSearchCV, GridSearchCV
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
    from prediction import results_processor
    from prediction import plotting # Used for individual plot functions if needed, Fig 3 is custom
    print("Successfully imported all required modules from 'prediction' package.")
except ImportError as e_pred_pkg:
    print(f"ERROR: Failed to import from 'prediction' package: {e_pred_pkg}")
    raise e_pred_pkg

# --- Experiment Control Flags ---
RUN_FT_MODELS = True
RUN_HM_MODELS = False
ITERATE_DATSCAN_REGIONS = False
ITERATE_TARGET_TYPE = "Z_SCORES_ONLY"
ITERATE_THRESHOLDS = True

# --- Focused Run Parameters ---
FOCUS_DATSCAN_REGIONS_TO_TEST = ["Contralateral_Putamen_Z"]

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
run_type_descriptor = "Run_Pub_FT_ThreshSweep_Fig3"
log_filename_base = f"experiment_log_{run_type_descriptor}.log"
log_filename = os.path.join(run_output_data_folder, log_filename_base)
summary_filename_prefix = f"summary_{run_type_descriptor}_"
best_summary_filename_prefix = f"best_auc_summary_{run_type_descriptor}_"

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
logger.info("NOTE: ALL PREDICTION ANALYSES ARE MANDATORILY AGE-CONTROLLED.")
logger.info(f"Run Timestamp: {timestamp}")
logger.info(f"Output Data Subfolder: {run_output_data_folder}")
if config.GENERATE_PLOTS: logger.info(f"Output Plot Subfolder: {run_output_plot_folder}")
logger.info(f"Config: prediction.{config.__name__}.py")
logger.info(f"Run FT Models: {RUN_FT_MODELS}, Run HM Models: {RUN_HM_MODELS}")
logger.info(f"Iterate DaTscan Regions: {ITERATE_DATSCAN_REGIONS}, Target Type: {ITERATE_TARGET_TYPE}, Iterate Thresholds: {ITERATE_THRESHOLDS}")

# --- Define Threshold Value Lists ---
Z_SCORE_THRESHOLDS_TO_TEST = np.round(np.arange(-2.0, -1.49, 0.1), 3) # Generates [-2.0, -1.9, ..., -1.5]
MIN_SAMPLES_PER_CLASS = 10

if ITERATE_THRESHOLDS: 
    logger.info(f"Threshold Z-Score Range: {min(Z_SCORE_THRESHOLDS_TO_TEST)} to {max(Z_SCORE_THRESHOLDS_TO_TEST)} (step 0.1)")
    logger.info(f"Min samples per class for threshold iteration: {MIN_SAMPLES_PER_CLASS}")
if not ITERATE_DATSCAN_REGIONS: logger.info(f"Focus Regions: {FOCUS_DATSCAN_REGIONS_TO_TEST}")

# --- Load Raw Data (ONCE) ---
logger.info("Loading initial raw data (df_raw)...")
try: df_raw = data_loader.load_data(config.INPUT_FOLDER, config.INPUT_CSV_NAME)
except Exception as e_data_raw: logger.exception("FATAL: Error loading raw data."); sys.exit(1)

# --- Filter config.CONFIGURATIONS_TO_RUN ---
active_configurations_to_run = []
if RUN_FT_MODELS: active_configurations_to_run.extend([c for c in config.CONFIGURATIONS_TO_RUN if c['task_prefix_for_features'].lower() == 'ft'])
if RUN_HM_MODELS: active_configurations_to_run.extend([c for c in config.CONFIGURATIONS_TO_RUN if c['task_prefix_for_features'].lower() == 'hm'])
active_configurations_to_run = [c for c in active_configurations_to_run if 'EngFeats' not in c['config_name']]
if not active_configurations_to_run: logger.error("FATAL: No model configurations selected."); sys.exit(1)
logger.info(f"Active model configurations: {[c['config_name'] for c in active_configurations_to_run]}")

# --- Initialize Results Storage ---
results_data_metrics = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(
    lambda: collections.defaultdict(lambda: collections.defaultdict(list)))))
results_data_roc = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(
    lambda: collections.defaultdict(lambda: collections.defaultdict(list)))))
results_data_rfe = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(
    lambda: collections.defaultdict(lambda: collections.defaultdict(list)))))
skipped_thresholds_info = collections.defaultdict(list)

# --- Derive Task Prefixes ---
defined_task_prefixes_main_script = sorted(list(set(exp_conf.get('task_prefix_for_features') for exp_conf in active_configurations_to_run)))
if not defined_task_prefixes_main_script: logger.error("FATAL: No task prefixes from active_configurations_to_run."); sys.exit(1)
logger.info(f"Kinematic task prefixes from active configs: {defined_task_prefixes_main_script}")

# --- MAIN EXPERIMENT LOOP ---
datscan_target_columns_to_process = [{"name": FOCUS_DATSCAN_REGIONS_TO_TEST[0], "type": "Z_SCORE"}]
for target_info in datscan_target_columns_to_process:
    results_data_metrics.clear()
    results_data_roc.clear()
    results_data_rfe.clear()
    skipped_thresholds_info.clear()
    
    target_col_name_key = target_info["name"]
    target_col_type_key = target_info["type"]

    logger.info(f"\n\n\n========== PROCESSING TARGET DATSCAN COLUMN: {target_col_name_key} (Type: {target_col_type_key}) ==========")
    
    current_threshold_list_to_use = Z_SCORE_THRESHOLDS_TO_TEST
    logger.info(f"    Thresholds for {target_col_name_key}: {current_threshold_list_to_use}")

    for current_threshold_val in current_threshold_list_to_use:
        thresh_val_key = round(float(current_threshold_val), 3)
        logger.info(f"\n\n  ======== THRESHOLD VALUE: {thresh_val_key:.3f} (Column: {target_col_name_key}) ========")
        
        try:
            X_full_glob, y_full_iter, groups_full_iter, task_feat_map_eng, task_feat_map_orig = \
                data_loader.prepare_data(df_raw, config, 
                                         target_z_score_column_override=target_col_name_key,
                                         abnormality_threshold_override=current_threshold_val)
            
            if y_full_iter.empty or X_full_glob.empty : 
                logger.warning(f"Data for {target_col_name_key}, Thr {thresh_val_key:.3f} is empty. Skipping.")
                skipped_thresholds_info[target_col_name_key].append({'Threshold_Value': thresh_val_key, 'Reason': 'Empty data after filtering'})
                continue
            
            class_counts_iter = y_full_iter.value_counts()
            logger.info(f"Class dist for {target_col_name_key}, Thr {thresh_val_key:.3f}: {class_counts_iter.to_dict()}")
            if (ITERATE_THRESHOLDS and class_counts_iter.min() < MIN_SAMPLES_PER_CLASS):
                logger.warning(f"{target_col_name_key}, Thr {thresh_val_key:.3f} has class < {MIN_SAMPLES_PER_CLASS}. Skipping this threshold.")
                skipped_thresholds_info[target_col_name_key].append({'Threshold_Value': thresh_val_key, 'Reason': f'Insufficient samples (< {MIN_SAMPLES_PER_CLASS})'})
                continue
                
        except (ValueError, KeyError) as e_prep: 
            logger.warning(f"Data for {target_col_name_key}, Thr {thresh_val_key:.3f} is invalid. Skipping. Reason: {e_prep}")
            skipped_thresholds_info[target_col_name_key].append({'Threshold_Value': thresh_val_key, 'Reason': f'Data prep error: {e_prep}'})
            continue

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
                        logger.warning(f"Rep {i_rep+1}: Not enough patients ({n_patients_iter}) for CV ({config.N_SPLITS_CV}) and min test ({min_test_p}). Skip rep.")
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
                        logger.warning("Outer split resulted in empty data or single class. Skip rep.")
                        continue
                    
                    for exp_conf in active_configurations_to_run: 
                        cfg_name = exp_conf['config_name']
                        task_pfx = exp_conf['task_prefix_for_features']
                        
                        # 1. Get the list of original base feature names from the config file.
                        original_base_names = config.BASE_KINEMATIC_COLS
                        
                        # 2. Construct the full feature names for the current task (e.g., 'ft_meanamplitude').
                        original_cols_for_task = [f"{task_pfx}_{base}" for base in original_base_names]
                        
                        # 3. Filter this list to ensure we only use columns that actually exist in our dataset.
                        current_cols_cfg = [col for col in original_cols_for_task if col in X_train_val.columns]
                        
                        if not current_cols_cfg: 
                            results_data_metrics[target_col_name_key][target_col_type_key][thresh_val_key][split_mode][cfg_name].append({})
                            continue
                            
                        X_tr_task = X_train_val[current_cols_cfg].copy()
                        X_tst_task = X_test[current_cols_cfg].copy()
                        y_tst_task = y_test.loc[X_tst_task.index]
                        
                        best_p, tune_s, fit_s = None, None, False
                        try:
                            pipe_cfg, srch_prms = pipeline_builder.build_pipeline_from_config(exp_conf, rand_state, config)
                            if not pipe_cfg.steps: raise ValueError("Empty pipeline")
                            
                            if inner_cv is None: 
                                best_p = clone(pipe_cfg); best_p.fit(X_tr_task, y_train_val); fit_s = True
                            else: 
                                if config.ENABLE_TUNING and srch_prms:
                                    scv_m = RandomizedSearchCV(estimator=pipe_cfg, param_distributions=srch_prms, n_iter=config.N_ITER_RANDOM_SEARCH,
                                                               scoring=config.TUNING_SCORING_METRIC, cv=inner_cv, n_jobs=-1, refit=True,
                                                               random_state=rand_state, error_score=np.nan)
                                    scv_m.fit(X_tr_task, y_train_val, groups=groups_train_val)
                                    best_p, tune_s, fit_s = scv_m.best_estimator_, scv_m.best_score_, True
                                else: 
                                    best_p = clone(pipe_cfg); best_p.fit(X_tr_task, y_train_val); fit_s=True
                                    
                            if fit_s:
                                if exp_conf.get('feature_selector')=='rfe' and best_p and hasattr(best_p, 'named_steps'):
                                    rfe_step = best_p.named_steps.get('feature_selector')
                                    if rfe_step and isinstance(rfe_step, RFE) and hasattr(rfe_step,'support_'):
                                        rfe_n = X_tr_task.columns[rfe_step.support_].tolist()
                                        results_data_rfe[target_col_name_key][target_col_type_key][thresh_val_key][split_mode][cfg_name].append(rfe_n)
                                            
                                if X_tst_task.empty or y_tst_task.nunique()<1: 
                                    nan_m={'roc_auc':np.nan, 'f1_macro':np.nan, 'best_cv_score':tune_s if tune_s is not None else np.nan}
                                    results_data_metrics[target_col_name_key][target_col_type_key][thresh_val_key][split_mode][cfg_name].append(nan_m)
                                else:
                                    preds_t = best_p.predict(X_tst_task)
                                    probs_t = best_p.predict_proba(X_tst_task)[:,1]
                                    t_met, t_roc = evaluation.evaluate_predictions(y_tst_task, preds_t, probs_t)
                                    t_met['best_cv_score']= tune_s if tune_s is not None else np.nan
                                    results_data_metrics[target_col_name_key][target_col_type_key][thresh_val_key][split_mode][cfg_name].append(t_met)
                                    if t_roc: results_data_roc[target_col_name_key][target_col_type_key][thresh_val_key][split_mode][cfg_name].append(t_roc)
                            else:
                                results_data_metrics[target_col_name_key][target_col_type_key][thresh_val_key][split_mode][cfg_name].append({'roc_auc':np.nan, 'best_cv_score': np.nan})
                        except ValueError as ve:
                            logger.warning(f"Handled ValueError for {cfg_name} (Seed: {rand_state}): {ve}")
                            results_data_metrics[target_col_name_key][target_col_type_key][thresh_val_key][split_mode][cfg_name].append({'roc_auc':np.nan, 'best_cv_score': np.nan})
                        except Exception as e_mdl: 
                            logger.exception(f"Error during model processing for config {cfg_name} (Seed: {rand_state}):")
                            results_data_metrics[target_col_name_key][target_col_type_key][thresh_val_key][split_mode][cfg_name].append({})
                except Exception as e_r: 
                    logger.exception(f"Error in Repetition {i_rep+1} (Seed: {rand_state}):")
    logger.info(f"\n========== FINISHED TARGET: {target_col_name_key} ==========")


logger.info(f"\n--- All Processing Finished ({run_type_descriptor}) ---")

# --- Aggregate and Save Results ---
logger.info("\n--- Aggregating Final Results (Metrics) ---")
summary_list = []
target_col_name = FOCUS_DATSCAN_REGIONS_TO_TEST[0]
if target_col_name in skipped_thresholds_info:
    for skipped_info in skipped_thresholds_info[target_col_name]:
        summary_list.append({
            'Datscan_Target_Column': target_col_name, 'Datscan_Target_Type': 'Z_SCORE',
            'Threshold_Value': skipped_info['Threshold_Value'], 'Mode': 'group',
            'Config_Name': active_configurations_to_run[0]['config_name'], 
            'Task_Name': active_configurations_to_run[0]['task_prefix_for_features'], 
            'N_Valid_Runs': 0, 'Mean_ROC_AUC': np.nan, 'Std_ROC_AUC': np.nan,
            'Mean_F1_Macro': np.nan, 'Std_F1_Macro': np.nan, 'Mean_Best_CV_Score': np.nan,
            'Skipped_Reason': skipped_info.get('Reason', 'Unknown')
        })

for target_col_name, target_col_data in results_data_metrics.items():
    for target_type_val, type_data_val in target_col_data.items():
        for thresh_val, thresh_data_val in type_data_val.items():
            for mode_val, mode_data_val in thresh_data_val.items():
                for config_name_val, metric_lists_val in mode_data_val.items():
                    if not metric_lists_val: continue
                    df_metrics = pd.DataFrame([m for m in metric_lists_val if m])
                    if df_metrics.empty: continue
                    
                    summary_list.append({
                        'Datscan_Target_Column': target_col_name, 'Datscan_Target_Type': target_type_val,
                        'Threshold_Value': thresh_val, 'Mode': mode_val, 'Config_Name': config_name_val, 
                        'Task_Name': utils.get_task_from_config_name(config_name_val), 
                        'N_Valid_Runs': int(df_metrics['roc_auc'].notna().sum()),
                        'Mean_ROC_AUC': df_metrics['roc_auc'].mean(), 'Std_ROC_AUC': df_metrics['roc_auc'].std(),
                        'Mean_F1_Macro': df_metrics['f1_macro'].mean(), 'Std_F1_Macro': df_metrics['f1_macro'].std(),
                        'Mean_Best_CV_Score': df_metrics['best_cv_score'].mean() if 'best_cv_score' in df_metrics.columns and df_metrics['best_cv_score'].notna().any() else np.nan,
                    })
final_summary_df = pd.DataFrame(summary_list).sort_values(by='Threshold_Value')

if not final_summary_df.empty:
    s_fname = os.path.join(run_output_data_folder, f"{summary_filename_prefix}{timestamp}.csv")
    final_summary_df.to_csv(s_fname, index=False, sep=';', decimal='.', float_format='%.5f')
    logger.info(f"Final Summary saved: {s_fname}")
    
    best_summary = final_summary_df.loc[[final_summary_df.dropna(subset=['Mean_ROC_AUC'])['Mean_ROC_AUC'].idxmax()]]
    bs_fname = os.path.join(run_output_data_folder, f"{best_summary_filename_prefix}{timestamp}.csv")
    best_summary.to_csv(bs_fname, index=False, sep=';', decimal='.', float_format='%.5f')
    logger.info(f"Best settings summary saved: {bs_fname}")


# ==============================================================================
# --- GENERATE PUBLICATION-READY FIGURE 3 ---
# ==============================================================================
if config.GENERATE_PLOTS and not final_summary_df.empty:
    logger.info("\n--- Generating Figure 3: Prediction Results Summary ---")
    try:
        # --- Setup Figure Layout ---
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.25)
        ax_A = fig.add_subplot(gs[0, 0])
        ax_B = fig.add_subplot(gs[0, 1])
        ax_C = fig.add_subplot(gs[1, :]) # Panel C spans the bottom row
        
        # --- Find Best Performing Threshold ---
        best_row = final_summary_df.loc[final_summary_df.dropna(subset=['Mean_ROC_AUC'])['Mean_ROC_AUC'].idxmax()]
        best_threshold = best_row['Threshold_Value']
        best_config = best_row['Config_Name']
        best_task = best_row['Task_Name']
        target_col_name = best_row['Datscan_Target_Column']
        
        # --- Panel A: Model Performance Optimization vs. Threshold ---
        plot_df_A = final_summary_df.dropna(subset=['Mean_ROC_AUC'])
        sns.lineplot(data=plot_df_A, x='Threshold_Value', y='Mean_ROC_AUC', marker='o', ax=ax_A, color='indigo', zorder=5)
        ax_A.fill_between(plot_df_A['Threshold_Value'], 
                          plot_df_A['Mean_ROC_AUC'] - plot_df_A['Std_ROC_AUC'],
                          plot_df_A['Mean_ROC_AUC'] + plot_df_A['Std_ROC_AUC'],
                          color='indigo', alpha=0.15, zorder=4)
        ax_A.axhline(0.5, color='k', linestyle='--', linewidth=1)
        ax_A.axvline(best_threshold, color='crimson', linestyle=':', linewidth=1.5, zorder=6, label=f'Optimal Threshold = {best_threshold:.2f}')
        ax_A.invert_xaxis()
        ax_A.set_title('A) Model Performance Optimization', fontsize=13, weight='bold', loc='left')
        ax_A.set_xlabel('Z-Score Abnormality Threshold')
        ax_A.set_ylabel('Mean ROC AUC')
        ax_A.legend(loc='lower left')
        
        # --- Panel B: Final Model Performance (ROC Curve) ---
        roc_runs = results_data_roc[target_col_name]['Z_SCORE'][best_threshold]['group'][best_config]
        if roc_runs:
            base_fpr = np.linspace(0, 1, 101)
            tprs_interp = []
            for fpr_run, tpr_run in roc_runs:
                if len(fpr_run) > 1 and len(tpr_run) > 1: # Basic check for valid curve
                    tprs_interp.append(np.interp(base_fpr, fpr_run, tpr_run))

            if tprs_interp:
                mean_tprs = np.mean(tprs_interp, axis=0); std_tprs = np.std(tprs_interp, axis=0)
                tprs_upper = np.minimum(mean_tprs + std_tprs, 1); tprs_lower = np.maximum(mean_tprs - std_tprs, 0)
                
                auc_label = f"Finger Tapping (AUC = {best_row['Mean_ROC_AUC']:.3f} ± {best_row['Std_ROC_AUC']:.3f})"
                ax_B.plot(base_fpr, mean_tprs, label=auc_label, color='indigo', lw=2)
                ax_B.fill_between(base_fpr, tprs_lower, tprs_upper, color='indigo', alpha=0.15)
        
        ax_B.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
        ax_B.set_title(f'B) Final Model Performance (at Z < {best_threshold:.2f})', fontsize=13, weight='bold', loc='left')
        ax_B.set_xlabel('False Positive Rate (1 - Specificity)'); ax_B.set_ylabel('True Positive Rate (Sensitivity)')
        ax_B.legend(loc='lower right'); ax_B.set_aspect('equal', adjustable='box')

        # --- Panel C: Predictive Kinematic Signature (Feature Importance) ---
        rfe_runs = results_data_rfe[target_col_name]['Z_SCORE'][best_threshold]['group'][best_config]
        if rfe_runs:
            feature_counts = collections.Counter(f for run in rfe_runs for f in run)
            n_runs = len(rfe_runs)
            rfe_df = pd.DataFrame(feature_counts.items(), columns=['Feature', 'Selection_Count'])
            rfe_df['Selection_Frequency'] = rfe_df['Selection_Count'] / n_runs
            rfe_df = rfe_df.sort_values('Selection_Frequency', ascending=False).head(config.PLOT_TOP_N_FEATURES)
            
            try: rfe_df['Readable_Feature'] = rfe_df['Feature'].apply(utils.get_readable_name)
            except Exception: rfe_df['Readable_Feature'] = rfe_df['Feature']

            sns.barplot(data=rfe_df, x='Selection_Frequency', y='Readable_Feature', color='indigo', ax=ax_C, alpha=0.8)
            ax_C.set_title('C) Predictive Kinematic Signature (Feature Importance)', fontsize=13, weight='bold', loc='left')
            ax_C.set_xlabel('RFE Selection Frequency (across all repetitions)'); ax_C.set_ylabel('Kinematic Feature')
            ax_C.set_xlim(0, 1)

        # --- Final Touches and Save ---
        fig.suptitle('Figure 3: Prediction of Dopaminergic Deficit from Finger-Tapping Kinematics', fontsize=16, weight='bold')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        figure_3_filename = os.path.join(run_output_plot_folder, "Figure3_Prediction_Summary.png")
        plt.savefig(figure_3_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"\n--- SUCCESS! Combined Figure 3 saved to: {os.path.abspath(figure_3_filename)} ---")
        
    except Exception as e:
        logger.error("\n" + "!"*60)
        logger.error("!!! AN UNEXPECTED ERROR OCCURRED DURING FIGURE 3 GENERATION !!!")
        logger.error(f"!!! Error Type: {type(e).__name__} at line {e.__traceback__.tb_lineno}")
        logger.error(f"!!! Error Message: {e}")
        logger.error("!!! Printing detailed traceback:")
        traceback.print_exc()
        logger.error("!"*60 + "\n")

# --- Script End ---
end_time_script = time.time()
total_duration_seconds = end_time_script - start_time_script
logger.info(f"\n========== SCRIPT FINISHED ({run_type_descriptor}): Total execution time: {total_duration_seconds:.1f} seconds "
            f"({total_duration_seconds/60:.1f} minutes) ==========")
# %%
# --- END OF FILE IV_datnik_prediction_run_experiments.py (FINAL PUBLICATION RUN WITH FIGURE 3 - COMPLETE) ---