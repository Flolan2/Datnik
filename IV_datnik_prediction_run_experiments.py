# --- START OF FILE IV_datnik_prediction_run_experiments.py (FINAL, COMPLETE, AND CORRECTED) ---
# -*- coding: utf-8 -*-
"""
Main script to run DatScan prediction experiments for BINARY classification.
This script orchestrates data loading, preprocessing, model training,
tuning, evaluation, and results aggregation for various configurations
defined in 'prediction.config'.

NOTE: All analyses performed by this script are MANDATORILY AGE-CONTROLLED
as per the logic in data_loader.py.

--- MODIFICATION FOR PUBLICATION RUN (FT vs HM with Directional Importance) ---
- Compares the predictive performance of Finger Tapping (FT) vs. Hand Movement (HM).
- Sweeps Z-score thresholds to find the optimal cutoff based on the best performing task.
- Generates a final, 3-panel Figure 3 showing:
    A) Threshold optimization for both tasks.
    B) Final ROC curves for both tasks.
    C) The predictive feature signature for the FT task WITH EFFECT DIRECTION.
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
    StratifiedGroupKFold, RandomizedSearchCV
)
from sklearn.base import clone
from sklearn.feature_selection import RFE

# --- Import from 'prediction' Package ---
try:
    from prediction import utils
    from prediction import data_loader
    from prediction import pipeline_builder
    from prediction import evaluation
    from prediction import results_processor
    from prediction import plotting
    print("Successfully imported all required modules from 'prediction' package.")
except ImportError as e_pred_pkg:
    print(f"ERROR: Failed to import from 'prediction' package: {e_pred_pkg}")
    raise e_pred_pkg

# --- Experiment Control Flags ---
RUN_FT_MODELS = True
RUN_HM_MODELS = True
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
run_type_descriptor = "Run_Pub_FT_vs_HM_Directional_Fig3" # Updated descriptor
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

# --- Define Threshold Value Lists ---
Z_SCORE_THRESHOLDS_TO_TEST = np.round(np.arange(-2.0, -1.49, 0.1), 3)
MIN_SAMPLES_PER_CLASS = 10

logger.info(f"Threshold Z-Score Range: {min(Z_SCORE_THRESHOLDS_TO_TEST)} to {max(Z_SCORE_THRESHOLDS_TO_TEST)} (step 0.1)")

# --- Load Raw Data (ONCE) ---
logger.info("Loading initial raw data (df_raw)...")
try: df_raw = data_loader.load_data(config.INPUT_FOLDER, config.INPUT_CSV_NAME)
except Exception as e_data_raw: logger.exception("FATAL: Error loading raw data."); sys.exit(1)

# --- Filter config.CONFIGURATIONS_TO_RUN ---
active_configurations_to_run = []
if RUN_FT_MODELS: active_configurations_to_run.extend([c for c in config.CONFIGURATIONS_TO_RUN if 'LR_RFE15_FT_OriginalFeats' in c['config_name']])
if RUN_HM_MODELS: active_configurations_to_run.extend([c for c in config.CONFIGURATIONS_TO_RUN if 'LR_RFE15_HM_OriginalFeats' in c['config_name']])

if not active_configurations_to_run: 
    logger.error("FATAL: No 'OriginalFeats' model configurations found in config.py."); sys.exit(1)
logger.info(f"Active model configurations: {[c['config_name'] for c in active_configurations_to_run]}")

# --- Initialize Results Storage ---
results_data_metrics = collections.defaultdict(list)
results_data_roc = collections.defaultdict(list)
skipped_thresholds_info = collections.defaultdict(list)

# --- MAIN EXPERIMENT LOOP ---
target_col_name_key = FOCUS_DATSCAN_REGIONS_TO_TEST[0]
logger.info(f"\n\n\n========== PROCESSING TARGET: {target_col_name_key} ==========")

for current_threshold_val in Z_SCORE_THRESHOLDS_TO_TEST:
    thresh_val_key = round(float(current_threshold_val), 3)
    logger.info(f"\n\n  ======== THRESHOLD VALUE: {thresh_val_key:.3f} ========")
    
    try:
        X_full_glob, y_full_iter, groups_full_iter, _, _ = \
            data_loader.prepare_data(df_raw, config, 
                                     target_z_score_column_override=target_col_name_key,
                                     abnormality_threshold_override=current_threshold_val)
        
        if y_full_iter.empty or X_full_glob.empty : 
            logger.warning(f"Data for Thr {thresh_val_key:.3f} is empty. Skipping.")
            skipped_thresholds_info[thresh_val_key].append({'Reason': 'Empty data after filtering'})
            continue
        
        class_counts_iter = y_full_iter.value_counts()
        if (class_counts_iter.min() < MIN_SAMPLES_PER_CLASS):
            logger.warning(f"Thr {thresh_val_key:.3f} has class < {MIN_SAMPLES_PER_CLASS}. Skipping.")
            skipped_thresholds_info[thresh_val_key].append({'Reason': f'Insufficient samples (< {MIN_SAMPLES_PER_CLASS})'})
            continue
            
    except Exception as e_prep: 
        logger.warning(f"Data prep for Thr {thresh_val_key:.3f} failed. Skipping. Reason: {e_prep}")
        skipped_thresholds_info[thresh_val_key].append({'Reason': f'Data prep error: {e_prep}'})
        continue

    for i_rep in range(config.N_REPETITIONS):
        rand_state = config.BASE_RANDOM_STATE + i_rep
        
        try:
            unique_patients_iter = groups_full_iter.unique()
            n_test_p = int(np.ceil(len(unique_patients_iter) * config.TEST_SET_SIZE))
            rng_i = np.random.RandomState(rand_state)
            shuffled_p_i = rng_i.permutation(unique_patients_iter)
            test_p_ids_i, train_val_p_ids_i = set(shuffled_p_i[:n_test_p]), set(shuffled_p_i[n_test_p:])
            train_val_mask = groups_full_iter.isin(train_val_p_ids_i)
            test_mask = groups_full_iter.isin(test_p_ids_i)
            X_train_val, y_train_val, groups_train_val = X_full_glob.loc[train_val_mask], y_full_iter.loc[train_val_mask], groups_full_iter.loc[train_val_mask]
            X_test, y_test = X_full_glob.loc[test_mask], y_full_iter.loc[test_mask]
            
            inner_cv = StratifiedGroupKFold(n_splits=config.N_SPLITS_CV, shuffle=True, random_state=rand_state)
            
            for exp_conf in active_configurations_to_run: 
                cfg_name = exp_conf['config_name']
                task_pfx = exp_conf['task_prefix_for_features']
                
                original_cols_for_task = [f"{task_pfx}_{base}" for base in config.BASE_KINEMATIC_COLS]
                current_cols_cfg = [col for col in original_cols_for_task if col in X_train_val.columns]
                
                if not current_cols_cfg: continue
                    
                X_tr_task, X_tst_task = X_train_val[current_cols_cfg], X_test[current_cols_cfg]
                
                pipe_cfg, srch_prms = pipeline_builder.build_pipeline_from_config(exp_conf, rand_state, config)
                
                scv_m = RandomizedSearchCV(estimator=pipe_cfg, param_distributions=srch_prms, n_iter=config.N_ITER_RANDOM_SEARCH,
                                           scoring=config.TUNING_SCORING_METRIC, cv=inner_cv, n_jobs=-1, refit=True,
                                           random_state=rand_state, error_score=np.nan)
                scv_m.fit(X_tr_task, y_train_val, groups=groups_train_val)
                best_p = scv_m.best_estimator_
                                
                preds_t = best_p.predict(X_tst_task)
                probs_t = best_p.predict_proba(X_tst_task)[:,1]
                t_met, t_roc = evaluation.evaluate_predictions(y_test, preds_t, probs_t)
                
                results_data_metrics[thresh_val_key, cfg_name].append(t_met)
                if t_roc: results_data_roc[thresh_val_key, cfg_name].append(t_roc)
        except Exception as e_rep: 
            logger.exception(f"Error in Repetition {i_rep+1} at Thr {thresh_val_key:.3f}:")

logger.info(f"\n========== FINISHED ALL THRESHOLDS ==========")

# --- Aggregate and Save Results ---
logger.info("\n--- Aggregating Final Results (Metrics) ---")
summary_list = []
for (thresh, cfg), metrics in results_data_metrics.items():
    df_metrics = pd.DataFrame(metrics)
    summary_list.append({
        'Threshold_Value': thresh, 'Config_Name': cfg, 
        'Task_Name': utils.get_task_from_config_name(cfg),
        'Mean_ROC_AUC': df_metrics['roc_auc'].mean(), 'Std_ROC_AUC': df_metrics['roc_auc'].std()
    })
final_summary_df = pd.DataFrame(summary_list).sort_values(by=['Config_Name', 'Threshold_Value'])

# --- Collect Coefficients for the Best Model at the Optimal Threshold ---
results_data_importances = collections.defaultdict(lambda: collections.defaultdict(list))
if not final_summary_df.empty:
    best_row = final_summary_df.loc[final_summary_df.dropna(subset=['Mean_ROC_AUC'])['Mean_ROC_AUC'].idxmax()]
    best_threshold = best_row['Threshold_Value']
    logger.info(f"\n--- Re-running at optimal threshold {best_threshold:.2f} to collect final model coefficients ---")

    try:
        X_full_glob, y_full_iter, groups_full_iter, _, _ = \
            data_loader.prepare_data(df_raw, config, 
                                     target_z_score_column_override=target_col_name_key,
                                     abnormality_threshold_override=best_threshold)

        for i_rep in range(config.N_REPETITIONS):
            rand_state = config.BASE_RANDOM_STATE + i_rep
            unique_patients_iter = groups_full_iter.unique()
            n_test_p = int(np.ceil(len(unique_patients_iter) * config.TEST_SET_SIZE))
            rng_i = np.random.RandomState(rand_state)
            shuffled_p_i = rng_i.permutation(unique_patients_iter)
            test_p_ids_i, train_val_p_ids_i = set(shuffled_p_i[:n_test_p]), set(shuffled_p_i[n_test_p:])
            train_val_mask = groups_full_iter.isin(train_val_p_ids_i)
            X_train_val, y_train_val, groups_train_val = X_full_glob.loc[train_val_mask], y_full_iter.loc[train_val_mask], groups_full_iter.loc[train_val_mask]
            
            inner_cv = StratifiedGroupKFold(n_splits=config.N_SPLITS_CV, shuffle=True, random_state=rand_state)

            exp_conf = [c for c in active_configurations_to_run if 'FT' in c['config_name']][0]
            cfg_name = exp_conf['config_name']
            task_pfx = exp_conf['task_prefix_for_features']
            original_cols_for_task = [f"{task_pfx}_{base}" for base in config.BASE_KINEMATIC_COLS]
            current_cols_cfg = [col for col in original_cols_for_task if col in X_train_val.columns]
            X_tr_task = X_train_val[current_cols_cfg]

            pipe_cfg, srch_prms = pipeline_builder.build_pipeline_from_config(exp_conf, rand_state, config)
            scv_m = RandomizedSearchCV(estimator=pipe_cfg, param_distributions=srch_prms, n_iter=config.N_ITER_RANDOM_SEARCH,
                                       scoring=config.TUNING_SCORING_METRIC, cv=inner_cv, n_jobs=-1, refit=True,
                                       random_state=rand_state, error_score=np.nan)
            scv_m.fit(X_tr_task, y_train_val, groups=groups_train_val)
            best_pipeline = scv_m.best_estimator_
            
            importances = utils.get_feature_importances(best_pipeline, X_tr_task.columns)
            if importances is not None:
                results_data_importances[cfg_name][task_pfx].append(importances)
    except Exception as e_rerun:
        logger.error(f"Error during coefficient re-run. Panel C may be blank. Error: {e_rerun}")

aggregated_importances = results_processor.aggregate_importances(
    {'group': results_data_importances}, config, output_dir_override=run_output_data_folder
)

# ==============================================================================
# --- GENERATE PUBLICATION-READY FIGURE 3 ---
# ==============================================================================
if config.GENERATE_PLOTS and not final_summary_df.empty:
    logger.info("\n--- Generating Figure 3: Prediction Results Summary (FT vs HM, Directional) ---")
    try:
        CONFIG_FT = 'LR_RFE15_FT_OriginalFeats'
        CONFIG_HM = 'LR_RFE15_HM_OriginalFeats'
        COLOR_FT = 'indigo'
        COLOR_HM = 'mediumseagreen'

        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
        ax_A = fig.add_subplot(gs[0, 0])
        ax_B = fig.add_subplot(gs[0, 1])
        ax_C = fig.add_subplot(gs[1, :])
        
        # Panel A
        plot_df_A_ft = final_summary_df[final_summary_df['Config_Name'] == CONFIG_FT].dropna(subset=['Mean_ROC_AUC'])
        plot_df_A_hm = final_summary_df[final_summary_df['Config_Name'] == CONFIG_HM].dropna(subset=['Mean_ROC_AUC'])
        sns.lineplot(data=plot_df_A_ft, x='Threshold_Value', y='Mean_ROC_AUC', marker='o', ax=ax_A, color=COLOR_FT, zorder=5, label='Finger Tapping')
        ax_A.fill_between(plot_df_A_ft['Threshold_Value'], plot_df_A_ft['Mean_ROC_AUC']-plot_df_A_ft['Std_ROC_AUC'], plot_df_A_ft['Mean_ROC_AUC']+plot_df_A_ft['Std_ROC_AUC'], color=COLOR_FT, alpha=0.15, zorder=4)
        sns.lineplot(data=plot_df_A_hm, x='Threshold_Value', y='Mean_ROC_AUC', marker='s', ax=ax_A, color=COLOR_HM, zorder=5, label='Hand Movement')
        ax_A.fill_between(plot_df_A_hm['Threshold_Value'], plot_df_A_hm['Mean_ROC_AUC']-plot_df_A_hm['Std_ROC_AUC'], plot_df_A_hm['Mean_ROC_AUC']+plot_df_A_hm['Std_ROC_AUC'], color=COLOR_HM, alpha=0.15, zorder=4)
        ax_A.axhline(0.5, color='k', linestyle='--', linewidth=1)
        ax_A.axvline(best_threshold, color='crimson', linestyle=':', linewidth=1.5, zorder=6, label=f'Optimal Threshold = {best_threshold:.2f}')
        ax_A.invert_xaxis(); ax_A.set_title('A) Model Performance Optimization', fontsize=13, weight='bold', loc='left'); ax_A.set_xlabel('Z-Score Abnormality Threshold'); ax_A.set_ylabel('Mean ROC AUC'); ax_A.legend(loc='lower left')
        
        # Panel B
        ax_B.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
        task_plot_configs = [{'name': CONFIG_FT, 'label': 'Finger Tapping', 'color': COLOR_FT}, {'name': CONFIG_HM, 'label': 'Hand Movement', 'color': COLOR_HM}]
        for plot_cfg in task_plot_configs:
            cfg_name = plot_cfg['name']; roc_runs = results_data_roc.get((best_threshold, cfg_name), []); metrics_row = final_summary_df[(final_summary_df['Config_Name'] == cfg_name) & (final_summary_df['Threshold_Value'] == best_threshold)].iloc[0]
            if roc_runs:
                base_fpr = np.linspace(0, 1, 101); tprs_interp = [np.interp(base_fpr, fpr, tpr) for fpr, tpr in roc_runs if len(fpr) > 1]
                if tprs_interp:
                    mean_tprs = np.mean(tprs_interp, axis=0); std_tprs = np.std(tprs_interp, axis=0); tprs_upper = np.minimum(mean_tprs + std_tprs, 1); tprs_lower = np.maximum(mean_tprs - std_tprs, 0)
                    auc_label = f"{plot_cfg['label']} (AUC = {metrics_row['Mean_ROC_AUC']:.3f} ± {metrics_row['Std_ROC_AUC']:.3f})"
                    ax_B.plot(base_fpr, mean_tprs, label=auc_label, color=plot_cfg['color'], lw=2); ax_B.fill_between(base_fpr, tprs_lower, tprs_upper, color=plot_cfg['color'], alpha=0.15)
        ax_B.set_title(f'B) Final Model Performance (at Z < {best_threshold:.2f})', fontsize=13, weight='bold', loc='left'); ax_B.set_xlabel('False Positive Rate (1 - Specificity)'); ax_B.set_ylabel('True Positive Rate (Sensitivity)'); ax_B.legend(loc='lower right'); ax_B.set_aspect('equal', adjustable='box')

        # Panel C
        ft_coeffs_df = aggregated_importances.get('group', {}).get(CONFIG_FT, {}).get('ft')
        
        if ft_coeffs_df is not None and not ft_coeffs_df.empty:
            plot_df = ft_coeffs_df.reindex(ft_coeffs_df['Mean_Importance'].abs().sort_values(ascending=False).index).head(config.PLOT_TOP_N_FEATURES)
            plot_df = plot_df.sort_values('Mean_Importance', ascending=False)
            plot_df['Readable_Feature'] = plot_df.index.str.replace('ft_', '').str.replace('_', ' ').str.title()
            colors = ['crimson' if c < 0 else 'royalblue' for c in plot_df['Mean_Importance']]
            sns.barplot(x=plot_df['Mean_Importance'], y=plot_df['Readable_Feature'], palette=colors, ax=ax_C)
            ax_C.axvline(0, color='k', linewidth=0.8, linestyle='--')
            ax_C.set_title('C) Predictive Kinematic Signature & Effect Direction (Finger Tapping)', fontsize=13, weight='bold', loc='left')
            ax_C.set_xlabel('Mean Coefficient (Log-Odds Change)\n<-- Lower Risk of Deficit | Higher Risk of Deficit -->')
            ax_C.set_ylabel('Kinematic Feature')
        else:
            ax_C.text(0.5, 0.5, "Coefficient data could not be generated.", ha='center', va='center', transform=ax_C.transAxes)
            ax_C.set_title('C) Predictive Kinematic Signature (Finger Tapping)', fontsize=13, weight='bold', loc='left')

        # --- Final Touches and Save ---
        fig.suptitle('Figure 3: Prediction of Dopaminergic Deficit from Tapping vs. Hand Movement Kinematics', fontsize=16, weight='bold')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        figure_3_filename = os.path.join(run_output_plot_folder, "Figure3_Prediction_Summary_FT_vs_HM_Directional.png")
        plt.savefig(figure_3_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"\n--- SUCCESS! FT vs HM Directional Figure 3 saved to: {os.path.abspath(figure_3_filename)} ---")
        
    except Exception as e:
        logger.error("\n" + "!"*60); logger.error("!!! AN UNEXPECTED ERROR OCCURRED DURING FIGURE 3 GENERATION !!!");
        logger.error(f"!!! Error Type: {type(e).__name__} at line {e.__traceback__.tb_lineno}"); logger.error(f"!!! Error Message: {e}");
        traceback.print_exc(); logger.error("!"*60 + "\n")

# --- Script End ---
end_time_script = time.time()
total_duration_seconds = end_time_script - start_time_script
logger.info(f"\n========== SCRIPT FINISHED ({run_type_descriptor}): Total execution time: {total_duration_seconds:.1f} seconds ({total_duration_seconds/60:.1f} minutes) ==========")