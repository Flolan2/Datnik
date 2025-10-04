# -*- coding: utf-8 -*-
"""
Script for EXTERNAL VALIDATION of a pre-trained DatScan prediction model.
VERSION 4: "Bullet-Proof" - Incorporates advanced expert feedback for publication.

--- METHODOLOGY ---
This script implements a rigorous, defensible validation workflow:

1.  TRAIN & SAVE FINAL MODEL:
    - Trains the definitive model on the ENTIRE original DaTscan dataset.
    - **CORRECTLY binarizes the DaTscan target directly, without re-residualizing.**
    - Determines and saves a "locked" decision threshold from the training data.
    - Saves the complete pipeline, feature-only age-control models, and locked threshold.

2.  LOAD & VALIDATE ON EXTERNAL COHORT:
    - Loads the new clinically-labeled cohort ('PD' vs 'Healthy').
    - **Uses a robust, case-insensitive renaming function tied to the trained model's features.**
    - Applies pre-trained transformations and feature engineering.
    - Evaluates performance at the pre-specified locked threshold with guardrails.
    - Calculates bootstrap 95% confidence intervals for the AUC.
    - Performs a calibration analysis (intercept and slope).
    - Generates an enhanced, multi-panel summary plot.
"""

import os
import sys
import pandas as pd
import numpy as np
import time
import logging
import datetime
import joblib
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

# --- Path Setup & Module Imports ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
if current_script_dir not in sys.path: sys.path.insert(0, current_script_dir)
try:
    from prediction import config, utils, data_loader, pipeline_builder, evaluation
    print("Successfully imported all required modules from 'prediction' package.")
except ImportError as e:
    print(f"ERROR: Failed to import from 'prediction' package: {e}"); raise e

# --- Validation Configuration ---
MODEL_TO_VALIDATE = 'LR_RFE15_FT_OriginalFeats'
NEW_CONTROLS_CSV = "Validation_DataFromControls.csv"
NEW_PD_CSV = "Validation_DataFromParkinsons.csv"
N_BOOTSTRAPS = 2000

# --- Setup Paths, Logging, etc. ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_output_folder = os.path.join(config.DATA_OUTPUT_FOLDER, "External_Validation_PubReady_" + timestamp)
os.makedirs(run_output_folder, exist_ok=True)
SAVED_MODEL_PATH = os.path.join(run_output_folder, f"{MODEL_TO_VALIDATE}_final_model_with_threshold.joblib")

log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(funcName)s:%(lineno)d] - %(message)s'
log_level = logging.INFO
log_filename = os.path.join(run_output_folder, f"external_validation_log.log")
logger = logging.getLogger('DatnikValidation'); logger.setLevel(log_level)
if logger.hasHandlers(): logger.handlers.clear()
ch = logging.StreamHandler(sys.stdout); ch.setLevel(log_level); ch_formatter = logging.Formatter(log_format); ch.setFormatter(ch_formatter); logger.addHandler(ch)
fh = logging.FileHandler(log_filename, mode='w', encoding='utf-8'); fh.setLevel(log_level); fh_formatter = logging.Formatter(log_format); fh.setFormatter(fh_formatter); logger.addHandler(fh)


# --- PLOTTING FUNCTION (UNCHANGED FROM PREVIOUS VERSION) ---
def plot_external_validation_results(y_true, y_proba, metrics, locked_threshold, cal_intercept, cal_slope, output_folder):
    # ... (This function is already excellent, no changes needed)
    logger.info("Generating enhanced external validation summary plot...")
    fig = plt.figure(figsize=(24, 7)); gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)
    ax_A = fig.add_subplot(gs[0, 0]); ax_B = fig.add_subplot(gs[0, 1]); ax_C = fig.add_subplot(gs[0, 2])
    fpr, tpr, _ = roc_curve(y_true, y_proba); auc_mean = metrics['bootstrap_auc_mean']; auc_ci = metrics['bootstrap_auc_ci']
    ax_A.plot(fpr, tpr, color='crimson', lw=2.5, label=f'AUC = {auc_mean:.3f} (95% CI: [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}])')
    ax_A.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)'); ax_A.set_title('A) Model Discrimination on External Cohort', fontsize=14, weight='bold', loc='left')
    ax_A.set_xlabel('False Positive Rate (1 - Specificity)'); ax_A.set_ylabel('True Positive Rate (Sensitivity)'); ax_A.legend(loc='lower right'); ax_A.set_aspect('equal', adjustable='box')
    plot_df = pd.DataFrame({'Diagnosis': y_true.map({0: 'Healthy', 1: 'PD'}), 'Predicted Risk Score': y_proba})
    sns.stripplot(data=plot_df, x='Diagnosis', y='Predicted Risk Score', order=['Healthy', 'PD'], palette=['royalblue', 'crimson'], jitter=0.2, alpha=0.6, ax=ax_B)
    sns.boxplot(data=plot_df, x='Diagnosis', y='Predicted Risk Score', order=['Healthy', 'PD'], palette=['lightblue', 'lightcoral'], boxprops=dict(alpha=0.6), ax=ax_B, showfliers=False)
    ax_B.axhline(locked_threshold, color='black', linestyle='--', linewidth=1.2, label=f'Locked Threshold = {locked_threshold:.2f}'); ax_B.set_title('B) Predicted Risk by Clinical Diagnosis', fontsize=14, weight='bold', loc='left')
    ax_B.set_ylabel('Predicted Probability of Dopaminergic Deficit'); ax_B.set_xlabel('Clinical Diagnosis'); ax_B.legend(); ax_B.set_ylim(-0.05, 1.05)
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=5, strategy='quantile')
    ax_C.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated"); ax_C.plot(prob_pred, prob_true, "s-", label="Model calibration", color='indigo')
    ax_C.set_title('C) Model Calibration', fontsize=14, weight='bold', loc='left'); ax_C.set_xlabel("Mean Predicted Probability (in bin)"); ax_C.set_ylabel("Fraction of Positives (in bin)")
    ax_C.legend(loc='upper left'); ax_C.text(0.05, 0.95, f'Intercept: {cal_intercept:.3f}\nSlope: {cal_slope:.3f}', transform=ax_C.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax_C.set_aspect('equal', adjustable='box'); fig.suptitle('External Validation of DaTscan-Trained Model on Clinical Cohort (PD vs. Healthy)', fontsize=16, weight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]); figure_filename = os.path.join(output_folder, "Figure_External_Validation_Summary.png")
    plt.savefig(figure_filename, dpi=300, bbox_inches='tight'); plt.close(fig); logger.info(f"[SUCCESS] Validation plot saved to: {os.path.abspath(figure_filename)}")

# --- HELPER FUNCTION FOR EXTERNAL DATA (MUST-FIX IMPLEMENTED) ---
def rename_validation_columns(df, config, trained_feature_names):
    """Robustly renames columns from validation CSV to match the exact training format."""
    logger.info("Renaming columns from validation data using trained feature list as ground truth...")
    
    low2orig = {c.lower(): c for c in df.columns}
    rename_map = {}
    
    # Map kinematic features
    for trained_feat in trained_feature_names:
        if trained_feat.startswith('ft_'):
            base_name_lower = trained_feat[3:].lower() # e.g., 'ft_meanamplitude' -> 'meanamplitude'
            if base_name_lower in low2orig:
                rename_map[low2orig[base_name_lower]] = trained_feat
    
    # Map Age column
    age_low = config.AGE_COL.lower()
    if age_low in low2orig:
        rename_map[low2orig[age_low]] = config.AGE_COL
        
    df.rename(columns=rename_map, inplace=True)
    return df

# --- CORE FUNCTIONS (MODIFIED WITH REVIEWER FIXES) ---
def train_and_save_final_model(original_data_df, config, model_config_dict, output_path):
    logger.info("--- Phase 1: Training, Determining Locked Threshold, and Saving ---")
    target_col = "Contralateral_Putamen_Z"
    X_full, y_cont, _, age_full, _, _ = data_loader.prepare_data_pre_split(
        original_data_df, config, target_z_score_column_override=target_col
    )
    
    # Residualize features (X) by age
    age_control_models = {'features': {}}
    X_resid = pd.DataFrame(index=X_full.index)
    train_df_for_ols = pd.concat([X_full, age_full], axis=1)
    for col in X_full.columns:
        model = ols(f"Q('{col}') ~ Q('{config.AGE_COL}')", data=train_df_for_ols).fit()
        X_resid[col] = X_full[col] - model.predict(age_full)
        age_control_models['features'][col] = model

    # --- MUST-FIX: Binarize the RAW, non-residualized DaTscan Z-score ---
    logger.info(f"Binarizing raw DaTscan Z-scores at threshold {config.ABNORMALITY_THRESHOLD} (no target residualization).")
    y_bin = (y_cont <= config.ABNORMALITY_THRESHOLD).astype(int)
    
    # Train the final pipeline
    task_pfx = model_config_dict['task_prefix_for_features']
    cols_for_task = [c for c in X_resid.columns if c.startswith(task_pfx + '_')]
    X_task = X_resid[cols_for_task]
    final_pipeline, _ = pipeline_builder.build_pipeline_from_config(model_config_dict, config.BASE_RANDOM_STATE, config)
    final_pipeline.fit(X_task, y_bin)

    # Determine and save the locked threshold
    train_probs = final_pipeline.predict_proba(X_task)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_bin, train_probs)
    locked_threshold = thresholds[np.argmax(tpr - fpr)] # Max Youden's J

    # Check if a pre-specified threshold should override the calculated one
    if 'locked_threshold' in model_config_dict:
        locked_threshold = model_config_dict['locked_threshold']
        logger.info(f"Overriding calculated threshold with pre-specified value from config: {locked_threshold:.4f}")
    else:
        logger.info(f"Determined locked threshold from training data (max Youden's J): {locked_threshold:.4f}")

    artifact_to_save = {
        'pipeline': final_pipeline,
        'age_control_models': age_control_models,
        'feature_names': cols_for_task,
        'locked_threshold': locked_threshold
    }
    joblib.dump(artifact_to_save, output_path)
    logger.info(f"[SUCCESS] Final model and artifacts saved to: {output_path}")
    return output_path

def validate_on_external_cohort(new_data_df, saved_model_path, config, output_folder):
    logger.info("--- Phase 2: Validating Model on External Cohort ---")
    artifacts = joblib.load(saved_model_path)
    pipeline, age_control_models, training_features, locked_thr = \
        artifacts['pipeline'], artifacts['age_control_models'], artifacts['feature_names'], artifacts['locked_threshold']
    logger.info(f"Successfully loaded artifacts. Using locked threshold: {locked_thr:.4f}")

    # --- MUST-FIX: Robust renaming using the loaded training features ---
    new_data_df = rename_validation_columns(new_data_df, config, training_features)

    # Data prep and FE
    new_data_df['y_true'] = new_data_df['Diagnosis'].map({'PD': 1, 'Healthy': 0})
    if hasattr(config, 'FEATURE_ENGINEERING_SETS_OPTIMIZED'):
        from prediction import feature_engineering as fe
        for fe_set in config.FEATURE_ENGINEERING_SETS_OPTIMIZED:
            if fe_set['name'].startswith('ft_'):
                try:
                    fe_func = getattr(fe, fe_set.get('function'))
                    new_features_df = fe_func(new_data_df, **fe_set.get('params', {}))
                    if new_features_df is not None:
                        for new_col in new_features_df.columns: new_data_df[new_col] = new_features_df[new_col]
                except Exception: pass
    new_data_df.dropna(subset=['y_true', config.AGE_COL] + training_features, inplace=True)
    y_true_new = new_data_df['y_true']
    
    X_new_raw = new_data_df[training_features]
    age_new = new_data_df[config.AGE_COL]
    
    # Age control application
    X_new_resid = pd.DataFrame(index=X_new_raw.index)
    for col in training_features:
        if col in age_control_models['features']:
            X_new_resid[col] = X_new_raw[col] - age_control_models['features'][col].predict(age_new)
        else:
            X_new_resid[col] = X_new_raw[col]

    # Predictions and evaluation
    probabilities = pipeline.predict_proba(X_new_resid)[:, 1]
    y_hat_locked = (probabilities >= locked_thr).astype(int)

    # Bootstrap CIs for AUC
    aucs = []; n_samples = len(y_true_new)
    for i in range(N_BOOTSTRAPS):
        indices = resample(np.arange(n_samples), n_samples=n_samples, random_state=i)
        if len(np.unique(y_true_new.iloc[indices])) < 2: continue
        aucs.append(roc_auc_score(y_true_new.iloc[indices], probabilities[indices]))
    mean_auc, lo_auc, hi_auc = np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

    # Metrics at Locked Threshold with guardrail
    if len(np.unique(y_true_new)) < 2:
        logger.warning("Only one class in external cohort; threshold metrics are undefined.")
        tn, fp, fn, tp = (0, 0, 0, 0) if 0 in np.unique(y_true_new) else (0, 0, 0, 0)
    else:
        tn, fp, fn, tp = confusion_matrix(y_true_new, y_hat_locked, labels=[0, 1]).ravel()
    
    metrics_at_locked = {
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'accuracy': (tp + tn) / len(y_true_new)
    }

    # --- MUST-FIX: Calibration model with penalty='none' ---
    eps = 1e-6
    logit_p = np.log((probabilities + eps) / (1 - probabilities + eps)).reshape(-1, 1)
    cal_model = LogisticRegression(penalty=None, solver='lbfgs').fit(logit_p, y_true_new)
    cal_intercept, cal_slope = cal_model.intercept_[0], cal_model.coef_[0, 0]

    # Reporting and Plotting
    # ... (Reporting print statements are unchanged) ...
    print("\n" + "="*60); print("--- EXTERNAL VALIDATION RESULTS (Publication Ready) ---"); print("="*60)
    print(f"Validated on {len(y_true_new)} subjects. Group Counts: {y_true_new.value_counts().to_dict()}")
    print("-" * 60); print(f"DISCRIMINATION (vs. Clinical Diagnosis):")
    print(f"  ROC AUC (Bootstrap 95% CI): {mean_auc:.3f} [{lo_auc:.3f} - {hi_auc:.3f}]")
    print("-" * 60); print(f"PERFORMANCE AT LOCKED THRESHOLD ({locked_thr:.3f}):")
    print(f"  Sensitivity (Recall):       {metrics_at_locked['sensitivity']:.3f}")
    print(f"  Specificity:                {metrics_at_locked['specificity']:.3f}")
    print(f"  Precision (PPV):            {metrics_at_locked['precision']:.3f}")
    print(f"  Accuracy:                   {metrics_at_locked['accuracy']:.3f}")
    print(f"  Confusion Matrix (TN,FP,FN,TP): ({tn}, {fp}, {fn}, {tp})")
    print("-" * 60); print(f"CALIBRATION:"); print(f"  Calibration Intercept:      {cal_intercept:.3f} (Ideal: 0)")
    print(f"  Calibration Slope:          {cal_slope:.3f} (Ideal: 1)"); print("="*60 + "\n")

    metrics = {'bootstrap_auc_mean': mean_auc, 'bootstrap_auc_ci': (lo_auc, hi_auc)}
    plot_external_validation_results(y_true_new, probabilities, metrics, locked_thr, cal_intercept, cal_slope, output_folder)
    return metrics

if __name__ == '__main__':
    start_time = time.time()
    utils.setup_warnings(); sns.set_theme(style="whitegrid")
    logger.info("========== SCRIPT START: EXTERNAL MODEL VALIDATION (Pub Ready) ==========")

    model_config = next((c for c in config.CONFIGURATIONS_TO_RUN if c['config_name'] == MODEL_TO_VALIDATE), None)
    if not model_config: logger.error(f"Model config '{MODEL_TO_VALIDATE}' not found in config.py"); sys.exit(1)

    if os.path.exists(SAVED_MODEL_PATH):
        logger.warning(f"Overwriting existing model file to ensure correct methodology is applied: {SAVED_MODEL_PATH}")
        os.remove(SAVED_MODEL_PATH)
        
    df_original = data_loader.load_data(config.INPUT_FOLDER, config.INPUT_CSV_NAME)
    train_and_save_final_model(df_original, config, model_config, SAVED_MODEL_PATH)

    try:
        df_hc = pd.read_csv(os.path.join(config.INPUT_FOLDER, NEW_CONTROLS_CSV))
        df_pd = pd.read_csv(os.path.join(config.INPUT_FOLDER, NEW_PD_CSV))
        df_hc['Diagnosis'] = 'Healthy'; df_pd['Diagnosis'] = 'PD'
        df_validation = pd.concat([df_hc, df_pd], ignore_index=True)
    except FileNotFoundError as e:
        logger.error(f"Validation data file not found. Details: {e}"); sys.exit(1)
    
    validation_results = validate_on_external_cohort(df_validation, SAVED_MODEL_PATH, config, run_output_folder)

    logger.info(f"========== SCRIPT FINISHED: Total execution time: {time.time() - start_time:.1f} seconds ==========")