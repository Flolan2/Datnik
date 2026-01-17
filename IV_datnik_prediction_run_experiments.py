# --- START OF FILE IV_datnik_prediction_run_experiments.py (MODIFIED FOR SENSITIVITY ANALYSIS) ---
# -*- coding: utf-8 -*-
"""
Main script to run DatScan prediction experiments for BINARY classification.

--- METHODOLOGY ---
This script performs two distinct, methodologically sound analyses:

1.  PRIMARY ANALYSIS (Fixed Threshold):
    - Evaluates model performance at a pre-defined, clinically relevant DaTscan
      threshold (Z < -1.96). This provides the primary, unbiased performance estimate.
    - Uses an adaptive repetitions framework to ensure a stable estimate of the
      mean ROC AUC.
    - Generates Figure 2 (ROC curves and feature signature).

2.  SENSITIVITY ANALYSIS (Threshold Sweep):
    - Characterizes the model's robustness by evaluating its performance across a
      range of abnormality thresholds.
    - This is NOT for selecting a new "best" threshold but to understand the model's
      performance profile from borderline to severe deficits.
    - The output is a plot of AUC vs. Z-Score Threshold.
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
from statsmodels.formula.api import ols
import scipy.stats as stats


# --- Path Setup & Module Imports ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
if current_script_dir not in sys.path: sys.path.insert(0, current_script_dir)
try:
    from prediction import config, utils, data_loader, pipeline_builder, evaluation, results_processor, plotting
    print("Successfully imported all required modules from 'prediction' package.")
except ImportError as e_pred_pkg:
    print(f"ERROR: Failed to import from 'prediction' package: {e_pred_pkg}"); raise e_pred_pkg
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV

plt.rcParams.update({'savefig.format': 'pdf', 'savefig.dpi': 300})
    

# --- Experiment Control Flags ---
PERFORM_FIXED_THRESHOLD_RUN = True   # Run the primary analysis at Z < -1.96
PERFORM_THRESHOLD_SWEEP_ANALYSIS = False # Run the secondary sensitivity analysis

# --- Threshold Sweep Configuration (only used if sweep is active) ---
THRESHOLDS_TO_SWEEP = np.round(np.arange(-3.0, -1.5, 0.1), 2) # e.g., -3.0, -2.9, ..., -1.6

# --- Adaptive Repetition Control ---
USE_AUTOMATIC_STOPPING_RULE = True
MIN_REPETITIONS = 50
MAX_REPETITIONS = 1000
DESIRED_HALF_WIDTH = 0.025
CONFIDENCE_LEVEL = 0.95

# --- HELPER FUNCTION FOR LEAK-FREE AGE CONTROL (Unchanged) ---
def apply_age_control_split(X_train, y_train_cont, age_train, X_test, y_test_cont, age_test, threshold):
    X_train_resid = pd.DataFrame(index=X_train.index); X_test_resid = pd.DataFrame(index=X_test.index)
    train_df_for_ols = pd.concat([X_train, age_train], axis=1)
    for col in X_train.columns:
        model = ols(f"Q('{col}') ~ Q('{config.AGE_COL}')", data=train_df_for_ols).fit()
        X_train_resid[col] = X_train[col] - model.predict(age_train)
        X_test_resid[col] = X_test[col] - model.predict(age_test)
    y_train_df_for_ols = pd.DataFrame({'target': y_train_cont, config.AGE_COL: age_train})
    y_model = ols(f"target ~ Q('{config.AGE_COL}')", data=y_train_df_for_ols).fit()
    y_train_resid = y_train_cont - y_model.predict(age_train); y_train_bin = (y_train_resid <= threshold).astype(int)
    y_test_resid = y_test_cont - y_model.predict(age_test); y_test_bin = (y_test_resid <= threshold).astype(int)
    return X_train_resid, y_train_bin, X_test_resid, y_test_bin

# --- Model Configurations to Run ---
RUN_FT_MODELS = True
RUN_HM_MODELS = True
FOCUS_DATSCAN_REGIONS_TO_TEST = ["Contralateral_Putamen_Z"]

# --- Setup Paths, Logging, etc. ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

run_output_data_folder = os.path.join(config.DATA_OUTPUT_FOLDER, timestamp) 
run_output_plot_folder = config.PLOT_OUTPUT_FOLDER

os.makedirs(run_output_plot_folder, exist_ok=True)

os.makedirs(run_output_data_folder, exist_ok=True)
if config.GENERATE_PLOTS: os.makedirs(run_output_plot_folder, exist_ok=True)

os.makedirs(run_output_data_folder, exist_ok=True)
if config.GENERATE_PLOTS: os.makedirs(run_output_plot_folder, exist_ok=True)

log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(funcName)s:%(lineno)d] - %(message)s'
log_level = logging.INFO
run_type_descriptor = "Run_Pub_FT_vs_HM_WithSensitivityAnalysis"
log_filename = os.path.join(run_output_data_folder, f"experiment_log_{run_type_descriptor}.log")
logger = logging.getLogger('DatnikExperiment'); logger.setLevel(log_level)
if logger.hasHandlers(): logger.handlers.clear()
ch = logging.StreamHandler(sys.stdout); ch.setLevel(log_level); ch_formatter = logging.Formatter(log_format); ch.setFormatter(ch_formatter); logger.addHandler(ch)
try: fh = logging.FileHandler(log_filename, mode='w', encoding='utf-8'); fh.setLevel(log_level); fh_formatter = logging.Formatter(log_format); fh.setFormatter(fh_formatter); logger.addHandler(fh)
except Exception as e_log: logger.error(f"Warning: File logging to {log_filename} failed: {e_log}")


start_time_script = time.time(); utils.setup_warnings(); sns.set_theme(style="whitegrid")
logger.info(f"========== SCRIPT START: DATNIK PREDICTION ({run_type_descriptor}) ==========")
logger.info(f"Run Timestamp: {timestamp}")

# --- Load and Prepare Data ---
logger.info("Loading initial raw data...");
df_raw = data_loader.load_data(config.INPUT_FOLDER, config.INPUT_CSV_NAME)
active_configurations_to_run = [c for c in config.CONFIGURATIONS_TO_RUN if (RUN_FT_MODELS and 'FT' in c['config_name']) or (RUN_HM_MODELS and 'HM' in c['config_name'])]
logger.info(f"Active model configurations: {[c['config_name'] for c in active_configurations_to_run]}")
target_col_name_key = FOCUS_DATSCAN_REGIONS_TO_TEST[0]
logger.info(f"Preparing data for target: {target_col_name_key}...")
X_full_raw, y_full_cont, groups_full, age_full, _, _ = data_loader.prepare_data_pre_split(df_raw, config, target_z_score_column_override=target_col_name_key)
if y_full_cont.empty or X_full_raw.empty:
    logger.error("Initial data preparation failed. Exiting."); sys.exit(1)


# ==============================================================================
# --- ANALYSIS 1: PRIMARY FIXED-THRESHOLD EXPERIMENT (Z < -1.96) ---
# ==============================================================================
if PERFORM_FIXED_THRESHOLD_RUN:
    logger.info(f"\n\n{'='*25}\n--- STARTING PRIMARY ANALYSIS: FIXED THRESHOLD (Z < {config.ABNORMALITY_THRESHOLD}) ---\n{'='*25}")
    
    # --- Data containers for this specific analysis ---
    results_data_metrics_fixed = collections.defaultdict(list)
    results_data_roc_fixed = collections.defaultdict(list)
    auc_tracker_fixed = collections.defaultdict(list)
    completed_configs_fixed = set()
    num_reps_to_run = MAX_REPETITIONS if USE_AUTOMATIC_STOPPING_RULE else config.N_REPETITIONS
    final_rep_count_fixed = 0

    for i_rep in range(num_reps_to_run):
        final_rep_count_fixed = i_rep + 1
        rand_state = config.BASE_RANDOM_STATE + i_rep
        logger.info(f"\n--- [Fixed Threshold] Repetition: {i_rep+1}/{num_reps_to_run} ---")

        if USE_AUTOMATIC_STOPPING_RULE and len(completed_configs_fixed) == len(active_configurations_to_run):
            logger.info(f"All model configurations have met precision. Stopping fixed-threshold run at repetition {i_rep}.")
            final_rep_count_fixed = i_rep; break

        try:
            # 1. Split and Binarize using the FIXED threshold
            unique_patients_iter = groups_full.unique(); n_test_p = int(np.ceil(len(unique_patients_iter) * config.TEST_SET_SIZE)); rng_i = np.random.RandomState(rand_state); shuffled_p_i = rng_i.permutation(unique_patients_iter); test_p_ids_i, train_val_p_ids_i = set(shuffled_p_i[:n_test_p]), set(shuffled_p_i[n_test_p:]); train_val_mask = groups_full.isin(train_val_p_ids_i); test_mask = groups_full.isin(test_p_ids_i)
            X_train_val_raw, y_train_val_cont, groups_train_val = X_full_raw.loc[train_val_mask], y_full_cont.loc[train_val_mask], groups_full.loc[train_val_mask]; X_test_raw, y_test_cont = X_full_raw.loc[test_mask], y_full_cont.loc[test_mask]; age_train_val, age_test = age_full.loc[train_val_mask], age_full.loc[test_mask]
            X_train_val, y_train_val, X_test, y_test = apply_age_control_split(X_train_val_raw, y_train_val_cont, age_train_val, X_test_raw, y_test_cont, age_test, config.ABNORMALITY_THRESHOLD)

            if y_train_val.value_counts().min() < config.N_SPLITS_CV: continue
            inner_cv = StratifiedGroupKFold(n_splits=config.N_SPLITS_CV, shuffle=True, random_state=rand_state)
            
            # 2. Train and Evaluate for each model config
            for exp_conf in active_configurations_to_run:
                cfg_name = exp_conf['config_name']
                if USE_AUTOMATIC_STOPPING_RULE and cfg_name in completed_configs_fixed: continue
                
                task_pfx = exp_conf['task_prefix_for_features']; original_cols_for_task = [f"{task_pfx}_{base}" for base in config.BASE_KINEMATIC_COLS]; current_cols_cfg = [col for col in original_cols_for_task if col in X_train_val.columns]; 
                if not current_cols_cfg: continue
                X_tr_task, X_tst_task = X_train_val[current_cols_cfg], X_test[current_cols_cfg]
                pipe_cfg, srch_prms = pipeline_builder.build_pipeline_from_config(exp_conf, rand_state, config); scv_m = RandomizedSearchCV(estimator=pipe_cfg, param_distributions=srch_prms, n_iter=config.N_ITER_RANDOM_SEARCH, scoring=config.TUNING_SCORING_METRIC, cv=inner_cv, n_jobs=-1, refit=True, random_state=rand_state, error_score=np.nan)
                scv_m.fit(X_tr_task, y_train_val, groups=groups_train_val); best_p = scv_m.best_estimator_
                preds_t = best_p.predict(X_tst_task); probs_t = best_p.predict_proba(X_tst_task)[:,1]
                t_met, t_roc = evaluation.evaluate_predictions(y_test, preds_t, probs_t)
                
                results_data_metrics_fixed[cfg_name].append(t_met)
                if t_roc: results_data_roc_fixed[cfg_name].append(t_roc)
                
                if USE_AUTOMATIC_STOPPING_RULE:
                    auc_tracker_fixed[cfg_name].append(t_met.get('roc_auc'))
                    current_n = len(auc_tracker_fixed[cfg_name])
                    if current_n >= MIN_REPETITIONS:
                        valid_aucs = [auc for auc in auc_tracker_fixed[cfg_name] if pd.notna(auc)]; current_n = len(valid_aucs)
                        if current_n < 2: continue
                        current_std = np.std(valid_aucs, ddof=1)
                        if current_std > 0:
                            t_value = stats.t.ppf(1 - (1 - CONFIDENCE_LEVEL) / 2, df=current_n - 1)
                            current_half_width = t_value * current_std / np.sqrt(current_n)
                            if current_half_width <= DESIRED_HALF_WIDTH:
                                logger.info(f"--- Config '{cfg_name}' met precision for fixed threshold run. ---")
                                completed_configs_fixed.add(cfg_name)
                        else: completed_configs_fixed.add(cfg_name)
        except Exception as e_rep:
            logger.exception(f"Error in Repetition {i_rep+1} of fixed-threshold run:")

    # --- Aggregate, Save, and Plot Results for the FIXED Threshold Run ---
    logger.info("\n--- Aggregating Final Results for FIXED THRESHOLD Run ---")
    summary_list_fixed = []
    for cfg_name, metrics_list in results_data_metrics_fixed.items():
        if not metrics_list: continue
        df_metrics = pd.DataFrame(metrics_list); n_reps = len(df_metrics)
        mean_auc = df_metrics['roc_auc'].mean(); std_auc = df_metrics['roc_auc'].std()
        ci_lower, ci_upper, ci_half_width = np.nan, np.nan, np.nan
        if n_reps > 1 and pd.notna(std_auc) and std_auc > 0:
            t_val = stats.t.ppf(1 - (1 - CONFIDENCE_LEVEL) / 2, df=n_reps - 1)
            ci_half_width = t_val * std_auc / np.sqrt(n_reps)
            ci_lower, ci_upper = mean_auc - ci_half_width, mean_auc + ci_half_width
        summary_list_fixed.append({
            'Config_Name': cfg_name, 'Task_Name': utils.get_task_from_config_name(cfg_name),
            'Mean_ROC_AUC': mean_auc, 'Std_ROC_AUC': std_auc, 'N_Repetitions': n_reps,
            'Mean_Sensitivity': df_metrics['sensitivity'].mean(), 'Std_Sensitivity': df_metrics['sensitivity'].std(),
            'Mean_Specificity': df_metrics['specificity'].mean(), 'Std_Specificity': df_metrics['specificity'].std(),
            'AUC_CI95_Lower': ci_lower, 'AUC_CI95_Upper': ci_upper, 'AUC_CI95_HalfWidth': ci_half_width
        })
    final_summary_df_fixed = pd.DataFrame(summary_list_fixed).sort_values(by=['Mean_ROC_AUC'], ascending=False)
    summary_path_fixed = os.path.join(run_output_data_folder, f"prediction_summary_FIXED_THRESHOLD.csv")
    final_summary_df_fixed.to_csv(summary_path_fixed, index=False, sep=';', decimal='.')
    logger.info(f"[SUCCESS] Saved fixed-threshold prediction summary to: {summary_path_fixed}")

    # --- Coefficient Collection and Figure 2 Plotting ---
    if config.GENERATE_PLOTS and not final_summary_df_fixed.empty:
        # This re-run loop is for generating a stable feature signature for visualization.
        results_data_importances_fixed = collections.defaultdict(lambda: collections.defaultdict(list))
        best_config_name = final_summary_df_fixed.iloc[0]['Config_Name']
        logger.info(f"\n--- Re-running '{best_config_name}' on fixed-threshold data to collect coefficients ---")
        num_reps_for_coeffs = final_summary_df_fixed.loc[final_summary_df_fixed['Config_Name'] == best_config_name, 'N_Repetitions'].iloc[0]
        try:
            for i_rep in range(int(num_reps_for_coeffs)):
                rand_state = config.BASE_RANDOM_STATE + i_rep
                unique_patients_iter = groups_full.unique(); n_test_p = int(np.ceil(len(unique_patients_iter) * config.TEST_SET_SIZE)); rng_i = np.random.RandomState(rand_state); shuffled_p_i = rng_i.permutation(unique_patients_iter); test_p_ids_i, train_val_p_ids_i = set(shuffled_p_i[:n_test_p]), set(shuffled_p_i[n_test_p:]); train_val_mask = groups_full.isin(train_val_p_ids_i)
                X_train_val_raw, y_train_val_cont, groups_train_val = X_full_raw.loc[train_val_mask], y_full_cont.loc[train_val_mask], groups_full.loc[train_val_mask]; age_train_val = age_full.loc[train_val_mask]
                X_train_val, y_train_val, _, _ = apply_age_control_split(X_train_val_raw, y_train_val_cont, age_train_val, X_train_val_raw.head(1), y_train_val_cont.head(1), age_train_val.head(1), config.ABNORMALITY_THRESHOLD)
                inner_cv = StratifiedGroupKFold(n_splits=config.N_SPLITS_CV, shuffle=True, random_state=rand_state)
                exp_conf = [c for c in active_configurations_to_run if c['config_name'] == best_config_name][0]
                cfg_name = exp_conf['config_name']; task_pfx = exp_conf['task_prefix_for_features']; original_cols_for_task = [f"{task_pfx}_{base}" for base in config.BASE_KINEMATIC_COLS]; current_cols_cfg = [col for col in original_cols_for_task if col in X_train_val.columns]; X_tr_task = X_train_val[current_cols_cfg]
                pipe_cfg, srch_prms = pipeline_builder.build_pipeline_from_config(exp_conf, rand_state, config); scv_m = RandomizedSearchCV(estimator=pipe_cfg, param_distributions=srch_prms, n_iter=config.N_ITER_RANDOM_SEARCH, scoring=config.TUNING_SCORING_METRIC, cv=inner_cv, n_jobs=-1, refit=True, random_state=rand_state, error_score=np.nan); scv_m.fit(X_tr_task, y_train_val, groups=groups_train_val); best_pipeline = scv_m.best_estimator_
                importances = utils.get_feature_importances(best_pipeline, X_tr_task.columns)
                if importances is not None: results_data_importances_fixed[cfg_name][task_pfx].append(importances)
        except Exception as e_rerun: logger.error(f"Error during coefficient re-run. Panel B may be blank. Error: {e_rerun}")
        
        aggregated_importances = results_processor.aggregate_importances({'group': results_data_importances_fixed}, config, output_dir_override=run_output_data_folder)
        
        # --- Generate Figure 2 ---
        logger.info("\n--- Generating Figure 2: Prediction Results Summary (Fixed Threshold) ---")
        plotting.plot_figure2(
            summary_df=final_summary_df_fixed,
            roc_data=results_data_roc_fixed,
            importances_data=aggregated_importances,
            output_folder=run_output_plot_folder,
            config=config
        )

# ==============================================================================
# --- ANALYSIS 2: SENSITIVITY ANALYSIS (Z-SCORE THRESHOLD SWEEP) ---
# ==============================================================================
if PERFORM_THRESHOLD_SWEEP_ANALYSIS:
    logger.info(f"\n\n{'='*25}\n--- STARTING SENSITIVITY ANALYSIS: THRESHOLD SWEEP ---\n{'='*25}")
    all_sweep_summaries = []

    for current_threshold in THRESHOLDS_TO_SWEEP:
        logger.info(f"\n\n{'#'*20} TESTING THRESHOLD: Z < {current_threshold} {'#'*20}\n")
        
        results_data_metrics_sweep = collections.defaultdict(list)
        num_reps_for_sweep = MIN_REPETITIONS
        
        for i_rep in range(num_reps_for_sweep):
            rand_state = config.BASE_RANDOM_STATE + i_rep
            logger.info(f"--- [Sweep Threshold {current_threshold}] Repetition: {i_rep+1}/{num_reps_for_sweep} ---")
            
            try:
                # 1. Split and Binarize using the CURRENT threshold from the sweep
                unique_patients_iter = groups_full.unique(); n_test_p = int(np.ceil(len(unique_patients_iter) * config.TEST_SET_SIZE)); rng_i = np.random.RandomState(rand_state); shuffled_p_i = rng_i.permutation(unique_patients_iter); test_p_ids_i, train_val_p_ids_i = set(shuffled_p_i[:n_test_p]), set(shuffled_p_i[n_test_p:]); train_val_mask = groups_full.isin(train_val_p_ids_i); test_mask = groups_full.isin(test_p_ids_i)
                X_train_val_raw, y_train_val_cont, groups_train_val = X_full_raw.loc[train_val_mask], y_full_cont.loc[train_val_mask], groups_full.loc[train_val_mask]; X_test_raw, y_test_cont = X_full_raw.loc[test_mask], y_full_cont.loc[test_mask]; age_train_val, age_test = age_full.loc[train_val_mask], age_full.loc[test_mask]
                
                X_train_val, y_train_val, X_test, y_test = apply_age_control_split(X_train_val_raw, y_train_val_cont, age_train_val, X_test_raw, y_test_cont, age_test, current_threshold)

                if y_train_val.value_counts().min() < config.N_SPLITS_CV:
                    logger.warning(f"Skipping rep for threshold {current_threshold}: too few members in a class for CV.")
                    continue
                inner_cv = StratifiedGroupKFold(n_splits=config.N_SPLITS_CV, shuffle=True, random_state=rand_state)

                # 2. Train and Evaluate
                for exp_conf in active_configurations_to_run:
                    cfg_name = exp_conf['config_name']
                    task_pfx = exp_conf['task_prefix_for_features']; original_cols_for_task = [f"{task_pfx}_{base}" for base in config.BASE_KINEMATIC_COLS]; current_cols_cfg = [col for col in original_cols_for_task if col in X_train_val.columns]; 
                    if not current_cols_cfg: continue
                    X_tr_task, X_tst_task = X_train_val[current_cols_cfg], X_test[current_cols_cfg]
                    pipe_cfg, srch_prms = pipeline_builder.build_pipeline_from_config(exp_conf, rand_state, config); scv_m = RandomizedSearchCV(estimator=pipe_cfg, param_distributions=srch_prms, n_iter=config.N_ITER_RANDOM_SEARCH, scoring=config.TUNING_SCORING_METRIC, cv=inner_cv, n_jobs=-1, refit=True, random_state=rand_state, error_score=np.nan)
                    scv_m.fit(X_tr_task, y_train_val, groups=groups_train_val); best_p = scv_m.best_estimator_
                    probs_t = best_p.predict_proba(X_tst_task)[:,1]
                    t_met, _ = evaluation.evaluate_predictions(y_test, None, probs_t)
                    results_data_metrics_sweep[cfg_name].append(t_met)
            except Exception as e_rep_sweep:
                logger.exception(f"Error in Repetition {i_rep+1} of threshold sweep for Z < {current_threshold}:")

        # --- Aggregate results for THIS ONE threshold and append ---
        for cfg_name, metrics_list in results_data_metrics_sweep.items():
            if not metrics_list: continue
            df_metrics = pd.DataFrame(metrics_list)
            all_sweep_summaries.append({
                'Threshold_Value': current_threshold,
                'Config_Name': cfg_name,
                'Task_Name': utils.get_task_from_config_name(cfg_name),
                'Mean_ROC_AUC': df_metrics['roc_auc'].mean(),
                'Std_ROC_AUC': df_metrics['roc_auc'].std(),
                'N_Repetitions': len(df_metrics)
            })

    # --- After the entire sweep, save and plot the results ---
    if all_sweep_summaries:
        final_sweep_summary_df = pd.DataFrame(all_sweep_summaries)
        sweep_summary_path = os.path.join(run_output_data_folder, "prediction_summary_THRESHOLD_SWEEP.csv")
        final_sweep_summary_df.to_csv(sweep_summary_path, index=False, sep=';', decimal='.')
        logger.info(f"\n[SUCCESS] Saved full threshold sweep summary to: {sweep_summary_path}")

        if config.GENERATE_PLOTS:
            logger.info("\n--- Generating Sensitivity Analysis Plot (AUC vs. Threshold) ---")
            sweep_plot_filename = os.path.join(run_output_plot_folder, "Supplementary Figure.pdf") 
            plotting.plot_performance_vs_threshold(
                summary_df=final_sweep_summary_df,
                x_col='Threshold_Value',
                y_col='Mean_ROC_AUC',
                filename=sweep_plot_filename,
                config=config
            )
        logger.warning("Threshold sweep analysis completed with no results to summarize or plot.")


# --- Script End ---
end_time_script = time.time()
total_duration_seconds = end_time_script - start_time_script
logger.info(f"\n========== SCRIPT FINISHED ({run_type_descriptor}): Total execution time: {total_duration_seconds:.1f} seconds ({total_duration_seconds/60:.1f} minutes) ==========")