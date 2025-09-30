# --- START OF FILE III_datnik_multivariate_models.py (CORRECTED PLOTTING) ---

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loads the hand-centric 'final_merged_data.csv', filters for OFF state,
CONTROLS FOR AGE by calculating residuals, and then calls separate pipeline
scripts to perform PLS and ElasticNet analyses.
Generates Figure 2 summarizing multivariate findings (PLS & ElasticNet).
"""

import os
import sys
import pandas as pd
import numpy as np
import traceback
from statsmodels.formula.api import ols

try:
    from datnik_plotting import plot_task_comparison_scatter, get_readable_name
    from datnik_run_pls import run_pls_pipeline
    from datnik_run_elasticnet import run_elasticnet_pipeline
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    PLOT_AVAILABLE = True
    print("[INFO] Successfully loaded analysis, plotting, and pipeline modules.")
except ImportError as e:
    print(f"CRITICAL WARNING: Error importing analysis/plotting/pipeline functions: {e}")
    PLOT_AVAILABLE = False
    if 'get_readable_name' not in locals():
        def get_readable_name(name, **kwargs): return name
    if 'run_pls_pipeline' not in locals():
        def run_pls_pipeline(*args, **kwargs): print("ERROR: run_pls_pipeline not imported."); return None
    if 'run_elasticnet_pipeline' not in locals():
        def run_elasticnet_pipeline(*args, **kwargs): print("ERROR: run_elasticnet_pipeline not imported."); return None


# -------------------------
# 1. Load and Process Data
# (This section is unchanged)
# -------------------------
script_dir = os.path.dirname(os.path.abspath(__file__)); project_root_dir = os.path.dirname(script_dir); processed_data_dir = os.path.join(project_root_dir, "Output", "Data_Processed"); merged_csv_file = os.path.join(processed_data_dir, "final_merged_data.csv")
print(f"--- Datnik Multivariate Models (Age Controlled) ---"); print(f"Loading data from: {merged_csv_file}")
try:
    try: df_full = pd.read_csv(merged_csv_file, sep=';', decimal='.')
    except: df_full = pd.read_csv(merged_csv_file, sep=',', decimal='.')
    print(f"Original data loaded successfully. Shape: {df_full.shape}")
    if 'Medication Condition' not in df_full.columns or 'Age' not in df_full.columns: print("CRITICAL ERROR: 'Medication Condition' or 'Age' column is missing."); sys.exit(1)
except FileNotFoundError: print(f"FATAL ERROR: Input file not found at {merged_csv_file}"); sys.exit(1)
except Exception as e: print(f"FATAL ERROR: Error loading or processing data: {e}"); sys.exit(1)
df_full['Medication Condition'] = df_full['Medication Condition'].astype(str).str.strip().str.lower(); df = df_full[df_full['Medication Condition'] == 'off'].copy()
if df.empty: print("FATAL ERROR: No data remaining after filtering for 'OFF' state. Exiting."); sys.exit(0)
else: print(f"Data filtered for 'OFF' state. New shape: {df.shape}")
base_kinematic_cols = ["meanamplitude","stdamplitude","meanspeed","stdspeed","meanrmsvelocity","stdrmsvelocity","meanopeningspeed","stdopeningspeed","meanclosingspeed","stdclosingspeed","meancycleduration","stdcycleduration","rangecycleduration","rate","amplitudedecay","velocitydecay","ratedecay","cvamplitude","cvcycleduration","cvspeed","cvrmsvelocity","cvopeningspeed","cvclosingspeed"]; TARGET_IMAGING_COL = "Contralateral_Putamen_Z"; AGE_COL = 'Age'; tasks = ['ft', 'hm']
output_base_folder = os.path.join(project_root_dir, "Output"); data_output_folder = os.path.join(output_base_folder, "Data"); plots_folder = os.path.join(output_base_folder, "Plots"); os.makedirs(data_output_folder, exist_ok=True); os.makedirs(plots_folder, exist_ok=True)
PLS_BSR_THRESHOLD = 1.8; pls_config = {'PLS_BSR_THRESHOLD': PLS_BSR_THRESHOLD, 'data_output_folder': data_output_folder, 'plots_folder': os.path.join(plots_folder, "PLS"), 'PLS_MAX_COMPONENTS': 5, 'PLS_N_PERMUTATIONS': 1000, 'PLS_N_BOOTSTRAPS': 1000, 'PLS_ALPHA': 0.05}; os.makedirs(pls_config['plots_folder'], exist_ok=True)
enet_config = {'PLS_BSR_THRESHOLD': PLS_BSR_THRESHOLD, 'data_output_folder': data_output_folder, 'plots_folder': os.path.join(plots_folder, "ElasticNet"), 'ENET_L1_RATIOS': np.linspace(0.1, 1.0, 10), 'ENET_CV_FOLDS': 5, 'ENET_MAX_ITER': 10000, 'ENET_RANDOM_STATE': 42, 'PLOT_TOP_N_ENET': 20}; os.makedirs(enet_config['plots_folder'], exist_ok=True)

# ----------------------------------------------------------------------
# 2. Prepare Data and Call Pipelines
# (This section is unchanged)
# ----------------------------------------------------------------------
if TARGET_IMAGING_COL not in df.columns: print(f"CRITICAL ERROR: '{TARGET_IMAGING_COL}' not found. Cannot run models."); sys.exit(1)
all_task_pls_results = {}; all_task_enet_results = {}
print("\n=== Starting Age-Controlled Multivariate Analysis Workflow ===")
for task in tasks:
    print(f"\n--- Processing Task: {task.upper()} ---")
    kinematic_cols = [f"{task}_{base}" for base in base_kinematic_cols if f"{task}_{base}" in df.columns]; cols_for_analysis = [TARGET_IMAGING_COL, AGE_COL] + kinematic_cols
    task_df = df[cols_for_analysis].dropna().copy()
    if len(task_df) < 15: print(f"WARNING: Insufficient data for {task.upper()} (N={len(task_df)}). Skipping models."); continue
    print(f"Clean data ready for {task.upper()} analysis (N={len(task_df)})")
    print("Calculating residuals to control for age...")
    y_model = ols(f"Q('{TARGET_IMAGING_COL}') ~ {AGE_COL}", data=task_df).fit(); y_resid = y_model.resid
    X_resid_df = pd.DataFrame(index=task_df.index)
    for col in kinematic_cols: x_model = ols(f"Q('{col}') ~ {AGE_COL}", data=task_df).fit(); X_resid_df[col] = x_model.resid
    print("Calling PLS pipeline with residualized data...")
    pls_config['task_prefix'] = task
    pls_results = run_pls_pipeline(X=X_resid_df, y=y_resid, config=pls_config)
    if pls_results: all_task_pls_results[task] = pls_results; print("PLS pipeline finished successfully.")
    else: print("PLS pipeline did not return results.")
    # --- NEW: Save PLS results to CSV ---
    if pls_results:
        try:
            # Save Bootstrap Ratios
            first_sig_lv_data = next((lv for lv in pls_results['lv_results'].values() if lv.get('significant')), pls_results['lv_results'].get('LV1'))
            if first_sig_lv_data and first_sig_lv_data.get('bootstrap_ratios') is not None:
                bsr_series = first_sig_lv_data['bootstrap_ratios']
                bsr_path = os.path.join(data_output_folder, f"pls_bootstrap_ratios_{task}.csv")
                bsr_series.to_csv(bsr_path, sep=';', decimal='.', header=['BSR'])
                print(f"[SUCCESS] Saved PLS BSRs for {task.upper()} to: {bsr_path}")
    
            # Save Summary Stats
            summary_stats = {
                'Task': task,
                'LV1_Correlation': first_sig_lv_data.get('correlation'),
                'LV1_P_Value': first_sig_lv_data.get('p_value'),
                'LV1_Significant': first_sig_lv_data.get('significant')
            }
            summary_df = pd.DataFrame([summary_stats])
            summary_path = os.path.join(data_output_folder, f"pls_summary_stats_{task}.csv")
            summary_df.to_csv(summary_path, index=False, sep=';', decimal='.')
            print(f"[SUCCESS] Saved PLS summary for {task.upper()} to: {summary_path}")
    
        except Exception as e:
            print(f"[ERROR] Could not save PLS results for {task.upper()}. Reason: {e}")
    # --- END NEW SECTION ---
    
    print("Calling ElasticNet pipeline with residualized data...")
    enet_config['task_prefix'] = task
    enet_results_single_task = run_elasticnet_pipeline(X=X_resid_df, y=y_resid, config=enet_config)
    if enet_results_single_task: all_task_enet_results[task] = enet_results_single_task; print("ElasticNet pipeline finished successfully.")
    else: print("ElasticNet pipeline did not return results.")
    
    # --- NEW: Save ElasticNet results to CSV ---
    if enet_results_single_task:
        try:
            # Save Coefficients
            if enet_results_single_task.get('coefficients') is not None:
                coeffs_series = enet_results_single_task['coefficients']
                coeffs_path = os.path.join(data_output_folder, f"enet_coefficients_{task}.csv")
                coeffs_series.to_csv(coeffs_path, sep=';', decimal='.', header=['Coefficient'])
                print(f"[SUCCESS] Saved ENet coefficients for {task.upper()} to: {coeffs_path}")
    
            # Save Summary Stats
            perf = enet_results_single_task.get('performance', {})
            summary_stats = {
                'Task': task,
                'CV_R2': perf.get('R2'),
                'Optimal_Alpha': perf.get('alpha'),
                'Optimal_L1_Ratio': perf.get('l1_ratio')
            }
            summary_df = pd.DataFrame([summary_stats])
            summary_path = os.path.join(data_output_folder, f"enet_summary_stats_{task}.csv")
            summary_df.to_csv(summary_path, index=False, sep=';', decimal='.')
            print(f"[SUCCESS] Saved ENet summary for {task.upper()} to: {summary_path}")
    
        except Exception as e:
            print(f"[ERROR] Could not save ENet results for {task.upper()}. Reason: {e}")
    # --- END NEW SECTION ---
    
    
pls_results_raw_dict = all_task_pls_results; enet_results = all_task_enet_results

# ==============================================================================
# --- 3. Generate Figure 2 (MODIFIED PLOTTING LOGIC) ---
# ==============================================================================
print("\n--- [INFO] Checking prerequisites for generating Figure 2 ---")
if not PLOT_AVAILABLE:
    print("[INFO] RESULT: SKIPPING Figure 2 because PLOT_AVAILABLE is False.")
elif not pls_results_raw_dict and not enet_results:
    print("[INFO] RESULT: SKIPPING Figure 2 because both PLS and ElasticNet results are missing.")
else:
    print("[INFO] RESULT: Prerequisites MET. Proceeding with Figure 2 generation.")
    try:
        print("\n--- Generating Figure 2: Multivariate Kinematic Signatures (Age-Controlled) ---")

        fig = plt.figure(figsize=(18, 16))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.6, wspace=0.4)
        ax_A = fig.add_subplot(gs[0, 0]); ax_B = fig.add_subplot(gs[0, 1])
        ax_C = fig.add_subplot(gs[1, 0]); ax_D = fig.add_subplot(gs[1, 1])
        axes = {'ft': {'pls': ax_A, 'enet': ax_C}, 'hm': {'pls': ax_B, 'enet': ax_D}}
        
        significant_pls_features = {}

        # --- Panels A & B: PLS Results ---
        for task in tasks:
            ax = axes[task]['pls']
            task_results = pls_results_raw_dict.get(task)
            task_name_full = "Finger Tapping" if task == 'ft' else "Hand Movements"
            panel_letter = "A" if task == 'ft' else "B"
            ax.set_title(f"{panel_letter}) PLS Results: {task_name_full}", fontsize=14, weight='bold', loc='left')
            
            if not task_results or not task_results.get('lv_results'):
                ax.text(0.5, 0.5, "No PLS results available.", ha='center', va='center'); continue

            first_sig_lv_data = next((lv for lv in task_results['lv_results'].values() if lv.get('significant')), None)
            
            if first_sig_lv_data and first_sig_lv_data.get('bootstrap_ratios') is not None:
                bsr = first_sig_lv_data['bootstrap_ratios'].dropna().sort_values()
                significant_pls_features[task] = bsr[bsr.abs() >= PLS_BSR_THRESHOLD].index.tolist()
                colors = ['#e85f5f' if x < 0 else '#007acc' for x in bsr.values]
                y_labels = [get_readable_name(name) for name in bsr.index]
                
                sns.barplot(x=bsr.values, y=y_labels, palette=colors, ax=ax)
                ax.axvline(x=PLS_BSR_THRESHOLD, color='k', linestyle='--'); ax.axvline(x=-PLS_BSR_THRESHOLD, color='k', linestyle='--')
                ax.set_xlabel("Bootstrap Ratio (BSR)", fontsize=11)
                
                p_val = first_sig_lv_data.get('p_value', np.nan); corr_val = first_sig_lv_data.get('correlation', np.nan)
                p_text = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
                stats_text = f"LV1 Correlation: r = {corr_val:.2f}\n{p_text}"
                # --- CHANGE 1: Move text box to a clean corner ---
                # Determine horizontal alignment based on data range
                ha = 'right' if ax.get_xlim()[0] < -1 else 'left'
                x_pos = 0.95 if ha == 'right' else 0.05
                ax.text(x_pos, 0.95, stats_text, transform=ax.transAxes, fontsize=10, va='top', ha=ha,
                        bbox=dict(boxstyle='round,pad=0.3', fc='whitesmoke', alpha=0.8))
            else:
                ax.text(0.5, 0.5, "No significant LV found.", ha='center', va='center')

        # --- Panels C & D: ElasticNet Results ---
        for task in tasks:
            ax = axes[task]['enet']
            task_results = enet_results.get(task)
            task_name_full = "Finger Tapping" if task == 'ft' else "Hand Movements"
            panel_letter = "C" if task == 'ft' else "D"
            ax.set_title(f"{panel_letter}) ElasticNet Results: {task_name_full}", fontsize=14, weight='bold', loc='left')

            if not task_results or task_results.get('coefficients') is None:
                ax.text(0.5, 0.5, "No ElasticNet results available.", ha='center', va='center'); continue

            coeffs_nonzero = task_results['coefficients'][lambda c: c.abs() > 1e-6].sort_values()

            if not coeffs_nonzero.empty:
                # --- CHANGE 2: Use new, high-contrast colors ---
                color_significant =  "#F08335" #Vibrant Orange
                color_not_significant = "#29686B" #teal

                pls_sig_for_task = significant_pls_features.get(task, [])
                colors = [color_significant if name in pls_sig_for_task else color_not_significant for name in coeffs_nonzero.index]
                y_labels = [get_readable_name(name) for name in coeffs_nonzero.index]
                
                sns.barplot(x=coeffs_nonzero.values, y=y_labels, palette=colors, ax=ax)
                ax.set_xlabel("Standardized Coefficient", fontsize=11)
                
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor=color_significant, label=f'Significant in PLS (BSR > |{PLS_BSR_THRESHOLD}|)'),
                                   Patch(facecolor=color_not_significant, label='Not significant in PLS')]
                ax.legend(handles=legend_elements, loc='best', fontsize=9)
                
                perf = task_results.get('performance', {}); r2 = perf.get('R2', np.nan); alpha = perf.get('alpha', np.nan); l1_ratio = perf.get('l1_ratio', np.nan)
                stats_text = f"CV R² = {r2:.2f}\nα = {alpha:.3f}, L1 ratio = {l1_ratio:.2f}"
                # --- CHANGE 1: Move text box to a clean corner ---
                ha = 'right' if ax.get_xlim()[0] < -0.1 else 'left'
                x_pos = 0.95 if ha == 'right' else 0.05
                ax.text(x_pos, 0.95, stats_text, transform=ax.transAxes, fontsize=10, va='top', ha=ha,
                        bbox=dict(boxstyle='round,pad=0.3', fc='whitesmoke', alpha=0.8))
            else:
                ax.text(0.5, 0.5, "Model selected no features.", ha='center', va='center')

        # --- Final adjustments and saving ---
        fig.suptitle('Figure 2: Age-Controlled Multivariate Kinematic Signatures of Dopamine Deficit (OFF State)', fontsize=18, weight='bold')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        figure_2_filename = os.path.join(plots_folder, "Figure2_Multivariate_Findings_Summary_AgeControlled_v2.png")
        plt.savefig(figure_2_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\n--- SUCCESS! Age-Controlled Figure 2 (v2) saved to: {os.path.abspath(figure_2_filename)} ---")

    except Exception as e:
        print("\n" + "!"*60)
        print("!!! AN UNEXPECTED ERROR OCCURRED DURING FIGURE 2 GENERATION !!!")
        traceback.print_exc()
        print("!"*60 + "\n")

print("\n--- Datnik Multivariate Models Script Finished ---")