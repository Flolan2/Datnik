# --- START OF FILE IV_datnik_prediction_run_experiments.py (FINAL, ADAPTIVE REPETITIONS) ---
# -*- coding: utf-8 -*-
"""
Main script to run DatScan prediction experiments for BINARY classification.

--- METHODOLOGY: FIXED THRESHOLD & ADAPTIVE REPETITIONS ---
This version evaluates the ability of kinematic features to predict a pre-defined,
clinically relevant DaTscan status (Z < -1.96). To ensure a stable and precise
performance estimate, the number of cross-validation repetitions is determined
adaptively. The script runs repetitions until the 95% confidence interval for the
primary outcome (mean ROC AUC) narrows to a pre-defined precision, governed by an
automatic stopping rule. This provides a highly robust and computationally efficient
validation framework.

--- PUBLICATION RUN (FT vs HM with Directional Importance) ---
- Compares FT vs. HM predictive performance.
- Generates a final, 2-panel Figure 3 showing:
    A) Final unbiased ROC curves.
    B) Predictive feature signature for the best-performing task.
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
import scipy.stats as stats # For confidence interval calculation

# --- Path Setup for 'prediction' Package ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
if current_script_dir not in sys.path: sys.path.insert(0, current_script_dir)

# --- Configuration Import ---
try:
    from prediction import config
    print(f"INFO: Using BINARY configuration from prediction.{config.__name__}.py")
except ImportError as e_config:
    print(f"FATAL ERROR: Could not import 'prediction.config'. Error: {e_config}"); sys.exit(1)

# --- Standard Library & Third-Party Imports ---
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV

# --- Import from 'prediction' Package ---
try:
    from prediction import utils, data_loader, pipeline_builder, evaluation, results_processor, plotting
    print("Successfully imported all required modules from 'prediction' package.")
except ImportError as e_pred_pkg:
    print(f"ERROR: Failed to import from 'prediction' package: {e_pred_pkg}"); raise e_pred_pkg

# --- Automatic Repetition Control Parameters ---
USE_AUTOMATIC_STOPPING_RULE = True
MIN_REPETITIONS = 30
MAX_REPETITIONS = 1100  # Set your manual upper limit here
DESIRED_HALF_WIDTH = 0.025  # Target precision for the 95% CI of the mean ROC AUC
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

# --- Experiment Control Flags ---
RUN_FT_MODELS = True
RUN_HM_MODELS = True
FOCUS_DATSCAN_REGIONS_TO_TEST = ["Contralateral_Putamen_Z"]

# --- Setup Paths, Logging, etc. ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_output_data_folder = os.path.join(config.DATA_OUTPUT_FOLDER, timestamp)
run_output_plot_folder = os.path.join(config.PLOT_OUTPUT_FOLDER, timestamp)
os.makedirs(run_output_data_folder, exist_ok=True)
if config.GENERATE_PLOTS: os.makedirs(run_output_plot_folder, exist_ok=True)

log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(funcName)s:%(lineno)d] - %(message)s'
log_level = logging.INFO
run_type_descriptor = "Run_Pub_FT_vs_HM_AdaptiveReps_LeakFree"
log_filename = os.path.join(run_output_data_folder, f"experiment_log_{run_type_descriptor}.log")
logger = logging.getLogger('DatnikExperiment'); logger.setLevel(log_level)
if logger.hasHandlers(): logger.handlers.clear()
ch = logging.StreamHandler(sys.stdout); ch.setLevel(log_level); ch_formatter = logging.Formatter(log_format); ch.setFormatter(ch_formatter); logger.addHandler(ch)
try: fh = logging.FileHandler(log_filename, mode='w', encoding='utf-8'); fh.setLevel(log_level); fh_formatter = logging.Formatter(log_format); fh.setFormatter(fh_formatter); logger.addHandler(fh)
except Exception as e_log: logger.error(f"Warning: File logging to {log_filename} failed: {e_log}")

start_time_script = time.time(); utils.setup_warnings(); sns.set_theme(style="whitegrid")
logger.info(f"========== SCRIPT START: DATNIK PREDICTION ({run_type_descriptor}) ==========")
if USE_AUTOMATIC_STOPPING_RULE:
    logger.info(f"METHODOLOGY: Adaptive repetitions with a FIXED clinical threshold (Z < {config.ABNORMALITY_THRESHOLD}).")
    logger.info(f"Target 95% CI half-width for AUC: {DESIRED_HALF_WIDTH}")
else:
    # Fallback to the N_REPETITIONS from config.py if the automatic rule is disabled
    logger.info(f"METHODOLOGY: Fixed {config.N_REPETITIONS} repetitions with a FIXED clinical threshold (Z < {config.ABNORMALITY_THRESHOLD}).")
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

# --- MAIN EXPERIMENT LOOP (WITH AUTOMATIC STOPPING RULE) ---
results_data_metrics = collections.defaultdict(list)
results_data_roc = collections.defaultdict(list)
auc_tracker = collections.defaultdict(list)
completed_configs = set()
num_reps_to_run = MAX_REPETITIONS if USE_AUTOMATIC_STOPPING_RULE else config.N_REPETITIONS
final_rep_count = 0

for i_rep in range(num_reps_to_run):
    final_rep_count = i_rep + 1
    rand_state = config.BASE_RANDOM_STATE + i_rep
    logger.info(f"\n\n{'='*20} REPETITION: {i_rep+1}/{num_reps_to_run} {'='*20}")

    if USE_AUTOMATIC_STOPPING_RULE and len(completed_configs) == len(active_configurations_to_run):
        logger.info(f"All model configurations have met the desired precision. Stopping at repetition {i_rep}.")
        final_rep_count = i_rep # Adjust final count as this loop didn't run
        break

    try:
        # 1. Create Train/Test Split
        unique_patients_iter = groups_full.unique(); n_test_p = int(np.ceil(len(unique_patients_iter) * config.TEST_SET_SIZE)); rng_i = np.random.RandomState(rand_state); shuffled_p_i = rng_i.permutation(unique_patients_iter); test_p_ids_i, train_val_p_ids_i = set(shuffled_p_i[:n_test_p]), set(shuffled_p_i[n_test_p:]); train_val_mask = groups_full.isin(train_val_p_ids_i); test_mask = groups_full.isin(test_p_ids_i)
        X_train_val_raw, y_train_val_cont, groups_train_val = X_full_raw.loc[train_val_mask], y_full_cont.loc[train_val_mask], groups_full.loc[train_val_mask]; X_test_raw, y_test_cont = X_full_raw.loc[test_mask], y_full_cont.loc[test_mask]; age_train_val, age_test = age_full.loc[train_val_mask], age_full.loc[test_mask]
        
        # 2. Apply Age Control & Binarize using the FIXED Threshold
        X_train_val, y_train_val, X_test, y_test = apply_age_control_split(X_train_val_raw, y_train_val_cont, age_train_val, X_test_raw, y_test_cont, age_test, config.ABNORMALITY_THRESHOLD)

        if y_train_val.value_counts().min() < config.N_SPLITS_CV:
            logger.warning(f"Rep {i_rep+1}: Training data has a class with fewer members than CV splits. Skipping rep."); continue

        inner_cv = StratifiedGroupKFold(n_splits=config.N_SPLITS_CV, shuffle=True, random_state=rand_state)
        
        # 3. Train and Evaluate for each model config
        for exp_conf in active_configurations_to_run: 
            cfg_name = exp_conf['config_name']
            if USE_AUTOMATIC_STOPPING_RULE and cfg_name in completed_configs: continue
            
            task_pfx = exp_conf['task_prefix_for_features']; original_cols_for_task = [f"{task_pfx}_{base}" for base in config.BASE_KINEMATIC_COLS]; current_cols_cfg = [col for col in original_cols_for_task if col in X_train_val.columns]; 
            if not current_cols_cfg: continue
            X_tr_task, X_tst_task = X_train_val[current_cols_cfg], X_test[current_cols_cfg]
            pipe_cfg, srch_prms = pipeline_builder.build_pipeline_from_config(exp_conf, rand_state, config); scv_m = RandomizedSearchCV(estimator=pipe_cfg, param_distributions=srch_prms, n_iter=config.N_ITER_RANDOM_SEARCH, scoring=config.TUNING_SCORING_METRIC, cv=inner_cv, n_jobs=-1, refit=True, random_state=rand_state, error_score=np.nan)
            scv_m.fit(X_tr_task, y_train_val, groups=groups_train_val); best_p = scv_m.best_estimator_
            preds_t = best_p.predict(X_tst_task); probs_t = best_p.predict_proba(X_tst_task)[:,1]
            t_met, t_roc = evaluation.evaluate_predictions(y_test, preds_t, probs_t)
            
            results_data_metrics[cfg_name].append(t_met)
            if t_roc: results_data_roc[cfg_name].append(t_roc)
            
            if USE_AUTOMATIC_STOPPING_RULE:
                auc_tracker[cfg_name].append(t_met.get('roc_auc'))
                current_n = len(auc_tracker[cfg_name])
                if current_n >= MIN_REPETITIONS:
                    valid_aucs = [auc for auc in auc_tracker[cfg_name] if pd.notna(auc)]; current_n = len(valid_aucs)
                    if current_n < 2: continue
                    current_std = np.std(valid_aucs, ddof=1)
                    if current_std > 0:
                        t_value = stats.t.ppf(1 - (1 - CONFIDENCE_LEVEL) / 2, df=current_n - 1)
                        current_half_width = t_value * current_std / np.sqrt(current_n)
                        logger.info(f"--- [Rep {i_rep+1}] Config '{cfg_name}': Current Half-Width = {current_half_width:.4f} (Target <= {DESIRED_HALF_WIDTH}) ---")
                        if current_half_width <= DESIRED_HALF_WIDTH:
                            logger.info(f"--- Config '{cfg_name}' has met precision target. It will no longer be run. ---")
                            completed_configs.add(cfg_name)
                    else: # If std is 0, precision is perfect
                        logger.info(f"--- Config '{cfg_name}' has met precision target (zero variance). ---")
                        completed_configs.add(cfg_name)

    except Exception as e_rep: 
        logger.exception(f"Error in Repetition {i_rep+1}:")

if USE_AUTOMATIC_STOPPING_RULE and final_rep_count == MAX_REPETITIONS:
    logger.warning(f"Reached MAX_REPETITIONS ({MAX_REPETITIONS}). Some models may not have met the desired precision.")
elif not USE_AUTOMATIC_STOPPING_RULE:
    logger.info(f"Completed all {config.N_REPETITIONS} fixed repetitions.")

# --- Aggregate and Save Results ---
logger.info("\n--- Aggregating Final Results Across All Repetitions ---")
summary_list = []
for cfg_name, metrics_list in results_data_metrics.items():
    if not metrics_list: continue
    df_metrics = pd.DataFrame(metrics_list)
    n_reps = len(df_metrics)
    mean_auc = df_metrics['roc_auc'].mean(); std_auc = df_metrics['roc_auc'].std()
    
    ci_half_width, ci_lower, ci_upper = np.nan, np.nan, np.nan
    if n_reps > 1 and pd.notna(std_auc) and std_auc > 0:
        t_val = stats.t.ppf(1 - (1 - CONFIDENCE_LEVEL) / 2, df=n_reps - 1)
        ci_half_width = t_val * std_auc / np.sqrt(n_reps)
        ci_lower, ci_upper = mean_auc - ci_half_width, mean_auc + ci_half_width

    summary_list.append({
        'Config_Name': cfg_name, 'Task_Name': utils.get_task_from_config_name(cfg_name),
        'Mean_ROC_AUC': mean_auc, 'Std_ROC_AUC': std_auc,
        'Mean_Sensitivity': df_metrics['sensitivity'].mean(), 'Std_Sensitivity': df_metrics['sensitivity'].std(),
        'Mean_Specificity': df_metrics['specificity'].mean(), 'Std_Specificity': df_metrics['specificity'].std(),
        'N_Repetitions': n_reps, 'AUC_CI95_Lower': ci_lower, 'AUC_CI95_Upper': ci_upper, 'AUC_CI95_HalfWidth': ci_half_width
    })
final_summary_df = pd.DataFrame(summary_list).sort_values(by=['Mean_ROC_AUC'], ascending=False)
try:
    summary_path = os.path.join(run_output_data_folder, f"prediction_summary_{run_type_descriptor}.csv")
    final_summary_df.to_csv(summary_path, index=False, sep=';', decimal='.')
    logger.info(f"[SUCCESS] Saved final prediction summary to: {summary_path}")
except Exception as e: logger.error(f"[ERROR] Failed to save final prediction summary. Reason: {e}")

# --- Collect Coefficients for the Best Model ---
results_data_importances = collections.defaultdict(lambda: collections.defaultdict(list))
if not final_summary_df.empty:
    best_config_name = final_summary_df.iloc[0]['Config_Name']
    logger.info(f"\n--- Re-running '{best_config_name}' to collect final model coefficients ---")
    # This re-run loop is for generating a stable feature signature for visualization.
    # We re-run for the number of repetitions that the BEST model actually completed.
    num_reps_for_coeffs = final_summary_df.iloc[0]['N_Repetitions']
    try:
        for i_rep in range(int(num_reps_for_coeffs)):
            rand_state = config.BASE_RANDOM_STATE + i_rep
            # ... (split data as before) ...
            unique_patients_iter = groups_full.unique(); n_test_p = int(np.ceil(len(unique_patients_iter) * config.TEST_SET_SIZE)); rng_i = np.random.RandomState(rand_state); shuffled_p_i = rng_i.permutation(unique_patients_iter); test_p_ids_i, train_val_p_ids_i = set(shuffled_p_i[:n_test_p]), set(shuffled_p_i[n_test_p:]); train_val_mask = groups_full.isin(train_val_p_ids_i)
            X_train_val_raw, y_train_val_cont, groups_train_val = X_full_raw.loc[train_val_mask], y_full_cont.loc[train_val_mask], groups_full.loc[train_val_mask]; age_train_val = age_full.loc[train_val_mask]
            
            X_train_val, y_train_val, _, _ = apply_age_control_split(X_train_val_raw, y_train_val_cont, age_train_val, X_train_val_raw.head(1), y_train_val_cont.head(1), age_train_val.head(1), config.ABNORMALITY_THRESHOLD)
            inner_cv = StratifiedGroupKFold(n_splits=config.N_SPLITS_CV, shuffle=True, random_state=rand_state)
            exp_conf = [c for c in active_configurations_to_run if c['config_name'] == best_config_name][0]
            cfg_name = exp_conf['config_name']; task_pfx = exp_conf['task_prefix_for_features']; original_cols_for_task = [f"{task_pfx}_{base}" for base in config.BASE_KINEMATIC_COLS]; current_cols_cfg = [col for col in original_cols_for_task if col in X_train_val.columns]; X_tr_task = X_train_val[current_cols_cfg]
            pipe_cfg, srch_prms = pipeline_builder.build_pipeline_from_config(exp_conf, rand_state, config); scv_m = RandomizedSearchCV(estimator=pipe_cfg, param_distributions=srch_prms, n_iter=config.N_ITER_RANDOM_SEARCH, scoring=config.TUNING_SCORING_METRIC, cv=inner_cv, n_jobs=-1, refit=True, random_state=rand_state, error_score=np.nan); scv_m.fit(X_tr_task, y_train_val, groups=groups_train_val); best_pipeline = scv_m.best_estimator_
            importances = utils.get_feature_importances(best_pipeline, X_tr_task.columns)
            if importances is not None: results_data_importances[cfg_name][task_pfx].append(importances)
    except Exception as e_rerun: logger.error(f"Error during coefficient re-run. Panel B may be blank. Error: {e_rerun}")

aggregated_importances = results_processor.aggregate_importances({'group': results_data_importances}, config, output_dir_override=run_output_data_folder)
try:
    ft_coeffs_df = aggregated_importances.get('group', {}).get('LR_RFE15_FT_OriginalFeats', {}).get('ft')
    if ft_coeffs_df is not None and not ft_coeffs_df.empty:
        coeffs_path = os.path.join(run_output_data_folder, f"prediction_ft_coefficients_{run_type_descriptor}.csv")
        ft_coeffs_df.to_csv(coeffs_path, sep=';', decimal='.')
        logger.info(f"[SUCCESS] Saved FT model coefficients to: {coeffs_path}")
except Exception as e: logger.error(f"[ERROR] Failed to save FT coefficients CSV. Reason: {e}")

# ==============================================================================
# --- GENERATE PUBLICATION-READY FIGURE 3 ---
# ==============================================================================
if config.GENERATE_PLOTS and not final_summary_df.empty:
    logger.info("\n--- Generating Figure 3: Prediction Results Summary (Adaptive Repetitions) ---")
    
    VARIABLE_NAMES = { 'meanamplitude': 'Mean Amplitude', 'stdamplitude': 'SD of Amplitude', 'meanrmsvelocity': 'Mean RMS Velocity', 'stdrmsvelocity': 'SD of RMS Velocity', 'meanopeningspeed': 'Mean Opening Speed', 'stdopeningspeed': 'SD of Opening Speed', 'meanclosingspeed': 'Mean Closing Speed', 'stdclosingspeed': 'SD of Closing Speed', 'meancycleduration': 'Mean Cycle Duration', 'stdcycleduration': 'SD of Cycle Duration', 'rangecycleduration': 'Range of Cycle Duration', 'amplitudedecay': 'Amplitude Decay', 'velocitydecay': 'Velocity Decay', 'ratedecay': 'Rate Decay', 'cvamplitude': 'CV of Amplitude', 'cvcycleduration': 'CV of Cycle Duration', 'cvrmsvelocity': 'CV of RMS Velocity', 'cvopeningspeed': 'CV of Opening Speed', 'cvclosingspeed': 'CV of Closing Speed', 'rate': 'Frequency', 'meanspeed': 'Mean Speed', 'stdspeed': 'SD of Speed', 'cvspeed': 'CV of Speed' }
    
    try:
        CONFIG_FT = 'LR_RFE15_FT_OriginalFeats'
        CONFIG_HM = 'LR_RFE15_HM_OriginalFeats'
        COLOR_FT = 'indigo'
        COLOR_HM = 'mediumseagreen'

        fig = plt.figure(figsize=(18, 8))
        gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.4, wspace=0.3)
        ax_A = fig.add_subplot(gs[0, 0])
        ax_B = fig.add_subplot(gs[0, 1])
        
        # Panel A: Final Model Performance (ROC Curves)
        ax_A.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
        for plot_cfg in [{'name': CONFIG_FT, 'label': 'Finger Tapping', 'color': COLOR_FT}, {'name': CONFIG_HM, 'label': 'Hand Movement', 'color': COLOR_HM}]:
            cfg_name = plot_cfg['name']
            roc_runs = results_data_roc.get(cfg_name, [])
            metrics_row = final_summary_df[final_summary_df['Config_Name'] == cfg_name]
            if roc_runs and not metrics_row.empty:
                metrics_row = metrics_row.iloc[0]
                base_fpr = np.linspace(0, 1, 101); tprs_interp = [np.interp(base_fpr, fpr, tpr) for fpr, tpr in roc_runs if len(fpr) > 1]
                if tprs_interp:
                    mean_tprs = np.mean(tprs_interp, axis=0); std_tprs = np.std(tprs_interp, axis=0)
                    tprs_upper = np.minimum(mean_tprs + std_tprs, 1); tprs_lower = np.maximum(mean_tprs - std_tprs, 0)
                    auc_label = f"{plot_cfg['label']} (AUC = {metrics_row['Mean_ROC_AUC']:.3f} ± {metrics_row['Std_ROC_AUC']:.3f})"
                    ax_A.plot(base_fpr, mean_tprs, label=auc_label, color=plot_cfg['color'], lw=2)
                    ax_A.fill_between(base_fpr, tprs_lower, tprs_upper, color=plot_cfg['color'], alpha=0.15)
        ax_A.set_title(f'A) Final Model Performance (at Z < {config.ABNORMALITY_THRESHOLD})', fontsize=13, weight='bold', loc='left')
        ax_A.set_xlabel('False Positive Rate (1 - Specificity)'); ax_A.set_ylabel('True Positive Rate (Sensitivity)')
        ax_A.legend(loc='lower right'); ax_A.set_aspect('equal', adjustable='box')

        # Panel B: Predictive Kinematic Signature
        ft_coeffs_df = aggregated_importances.get('group', {}).get(CONFIG_FT, {}).get('ft')
        if ft_coeffs_df is not None and not ft_coeffs_df.empty:
            plot_df = ft_coeffs_df.reindex(ft_coeffs_df['Mean_Importance'].abs().sort_values(ascending=False).index).head(config.PLOT_TOP_N_FEATURES).sort_values('Mean_Importance', ascending=False)
            base_feature_names = plot_df.index.str.replace('ft_', ''); plot_df['Readable_Feature'] = base_feature_names.map(VARIABLE_NAMES)
            colors = ['crimson' if c < 0 else 'royalblue' for c in plot_df['Mean_Importance']]
            sns.barplot(x=plot_df['Mean_Importance'], y=plot_df['Readable_Feature'], palette=colors, ax=ax_B)
            ax_B.axvline(0, color='k', linewidth=0.8, linestyle='--')
            ax_B.set_title('B) Predictive Kinematic Signature (Finger Tapping)', fontsize=13, weight='bold', loc='left')
            ax_B.set_xlabel('Mean Coefficient (Log-Odds Change)\n<-- Lower Risk of Deficit | Higher Risk of Deficit -->')
            ax_B.set_ylabel('Kinematic Feature')
        else:
            ax_B.text(0.5, 0.5, "Coefficient data could not be generated.", ha='center', va='center'); ax_B.set_title('B) Predictive Kinematic Signature (Finger Tapping)', fontsize=13, weight='bold', loc='left')

        fig.suptitle(f'Figure 3: Prediction of Dopaminergic Deficit (Fixed Threshold Z < {config.ABNORMALITY_THRESHOLD})', fontsize=16, weight='bold')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        figure_3_filename = os.path.join(run_output_plot_folder, f"Figure3_Prediction_Summary_{run_type_descriptor}.png")
        plt.savefig(figure_3_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"\n--- SUCCESS! Final Figure 3 saved to: {os.path.abspath(figure_3_filename)} ---")
        
    except Exception as e:
        logger.error("\n" + "!"*60); logger.error("!!! AN UNEXPECTED ERROR OCCURRED DURING FIGURE 3 GENERATION !!!");
        logger.error(f"!!! Error Type: {type(e).__name__} at line {e.__traceback__.tb_lineno}"); logger.error(f"!!! Error Message: {e}");
        traceback.print_exc(); logger.error("!"*60 + "\n")

# --- Script End ---
end_time_script = time.time()
total_duration_seconds = end_time_script - start_time_script
logger.info(f"\n========== SCRIPT FINISHED ({run_type_descriptor}): Total execution time: {total_duration_seconds:.1f} seconds ({total_duration_seconds/60:.1f} minutes) ==========")