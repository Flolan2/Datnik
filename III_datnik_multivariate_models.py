# --- START OF FILE datnik_multivariate_models.py (Trimmed for PLS & ElasticNet Focus, Saves Full PLS Details) ---

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loads the hand-centric 'final_merged_data.csv', filters for OFF state,
and calls separate pipeline scripts to perform PLS and ElasticNet analyses.
Saves comprehensive PLS details for each task.
Also generates Bivariate Task Comparison plots.
"""

import os
import sys
import pandas as pd
import numpy as np

try:
    from datnik_plotting import plot_task_comparison_scatter
    from datnik_run_pls import run_pls_pipeline
    from datnik_run_elasticnet import run_elasticnet_pipeline
    PLOT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Error importing analysis/plotting/pipeline functions: {e}")
    PLOT_AVAILABLE = False
    if 'plot_task_comparison_scatter' not in locals():
        def plot_task_comparison_scatter(*args, **kwargs): print("WARNING: plot_task_comparison_scatter not imported.")
    if 'run_pls_pipeline' not in locals():
        def run_pls_pipeline(*args, **kwargs): print("ERROR: run_pls_pipeline not imported."); return {}
    if 'run_elasticnet_pipeline' not in locals():
        def run_elasticnet_pipeline(*args, **kwargs): print("ERROR: run_elasticnet_pipeline not imported."); return {}

# -------------------------
# 1. Load and Process Data
# -------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(script_dir)
processed_data_dir = os.path.join(project_root_dir, "Output", "Data_Processed")
merged_csv_file = os.path.join(processed_data_dir, "final_merged_data.csv")

print(f"--- Datnik Multivariate Models (PLS & ElasticNet Focus, Full PLS Save) ---") # Updated script title
print(f"Loading data from: {merged_csv_file}")
try:
    try:
        df_full = pd.read_csv(merged_csv_file, sep=';', decimal='.')
        print("Read merged CSV with ';' separator.")
    except (FileNotFoundError, pd.errors.ParserError, UnicodeDecodeError) as e_semi:
        print(f"Warning: Failed to parse/find with ';': {e_semi}. Trying ',' separator...")
        df_full = pd.read_csv(merged_csv_file, sep=',', decimal='.')
        print("Read merged CSV with ',' separator.")
    except Exception as read_err:
        print(f"Error reading {merged_csv_file}: {read_err}")
        sys.exit(1)

    print(f"Original data loaded successfully. Shape: {df_full.shape}")
    if 'Medication Condition' not in df_full.columns:
        print("CRITICAL ERROR: 'Medication Condition' column is missing."); sys.exit(1)
    if 'Hand_Performed' not in df_full.columns:
        print("CRITICAL ERROR: 'Hand_Performed' column is missing."); sys.exit(1)
except FileNotFoundError:
    print(f"Error: Input file not found at {merged_csv_file}"); sys.exit(1)
except Exception as e:
    print(f"Error loading or processing data: {e}"); sys.exit(1)

print("\nFiltering data for Medication Condition == 'off'...")
df_full['Medication Condition'] = df_full['Medication Condition'].astype(str).str.strip().str.lower()
df = df_full[df_full['Medication Condition'] == 'off'].copy()

if df.empty:
    print("Error: No data remaining after filtering for 'OFF' medication state. Exiting."); sys.exit(0)
else:
    print(f"Data filtered for 'OFF' state. New shape: {df.shape}")
    print(f"Unique Patient-Hand combinations in OFF state data: {len(df.groupby(['Patient ID', 'Hand_Performed']))}")

# --- Setup for Multivariate Models & Task Comparison Plot ---
base_kinematic_cols = [
    "meanamplitude", "stdamplitude", "meanspeed", "stdspeed", "meanrmsvelocity",
    "stdrmsvelocity", "meanopeningspeed", "stdopeningspeed", "meanclosingspeed",
    "stdclosingspeed", "meancycleduration", "stdcycleduration", "rangecycleduration",
    "rate", "amplitudedecay", "velocitydecay", "ratedecay", "cvamplitude",
    "cvcycleduration", "cvspeed", "cvrmsvelocity", "cvopeningspeed", "cvclosingspeed"
]
TARGET_IMAGING_BASE = "Contralateral_Striatum"
TARGET_IMAGING_COL = f"{TARGET_IMAGING_BASE}_Z"
tasks = ['ft', 'hm']

output_base_folder = os.path.join(project_root_dir, "Output")
data_output_folder = os.path.join(output_base_folder, "Data") # For model CSV outputs
plots_folder = os.path.join(output_base_folder, "Plots")     # For model plots & comparison
os.makedirs(data_output_folder, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)
print(f"Output folders created/checked at: {output_base_folder}")

# --- Configs for Pipelines ---
PLS_MAX_COMPONENTS = 5; PLS_N_PERMUTATIONS = 1000; PLS_N_BOOTSTRAPS = 1000
PLS_ALPHA = 0.05; PLS_BSR_THRESHOLD = 1.8 # Using 1.8 as per your plot
# PLS_RESULTS_FILE_PATH is still used by datnik_run_elasticnet.py if it needs to load significant PLS features
# The datnik_run_pls.py script will still save its own version of significant results if criteria are met.
PLS_RESULTS_FILE_PATH_FOR_ENET = os.path.join(data_output_folder, "pls_significant_features_for_enet_highlighting.csv")

ENET_L1_RATIOS = np.linspace(0.1, 1.0, 10); ENET_CV_FOLDS = 5
ENET_MAX_ITER = 10000; ENET_RANDOM_STATE = 42; PLOT_TOP_N_ENET = 20

pls_config = {
    'tasks': tasks, 'base_kinematic_cols': base_kinematic_cols, 'TARGET_IMAGING_COL': TARGET_IMAGING_COL,
    'PLS_MAX_COMPONENTS': PLS_MAX_COMPONENTS, 'PLS_N_PERMUTATIONS': PLS_N_PERMUTATIONS,
    'PLS_N_BOOTSTRAPS': PLS_N_BOOTSTRAPS, 'PLS_ALPHA': PLS_ALPHA, 'PLS_BSR_THRESHOLD': PLS_BSR_THRESHOLD,
    'data_output_folder': data_output_folder, # datnik_run_pls saves its own summary here
    'plots_folder': os.path.join(plots_folder, "PLS")
}
os.makedirs(pls_config['plots_folder'], exist_ok=True)

enet_config = {
    'tasks': tasks, 'base_kinematic_cols': base_kinematic_cols, 'TARGET_IMAGING_COL': TARGET_IMAGING_COL,
    'ENET_L1_RATIOS': ENET_L1_RATIOS, 'ENET_CV_FOLDS': ENET_CV_FOLDS,
    'ENET_MAX_ITER': ENET_MAX_ITER, 'ENET_RANDOM_STATE': ENET_RANDOM_STATE,
    'data_output_folder': data_output_folder,
    'plots_folder': os.path.join(plots_folder, "ElasticNet"),
    'PLOT_TOP_N_ENET': PLOT_TOP_N_ENET,
    'PLS_RESULTS_FILE': PLS_RESULTS_FILE_PATH_FOR_ENET, # ElasticNet uses this for highlighting
    'PLS_BSR_THRESHOLD': PLS_BSR_THRESHOLD # The threshold used for highlighting in ENet plots
}
os.makedirs(enet_config['plots_folder'], exist_ok=True)

# --- Load Bivariate Data ---
raw_bivar_file = os.path.join(data_output_folder, f"ALL_raw_correlations_{TARGET_IMAGING_COL}_OFF.csv")
sig_bivar_file_combined = os.path.join(data_output_folder, f"ALL_significant_correlations_{TARGET_IMAGING_COL}_combined_OFF.csv")
all_raw_bivariate_results_df = pd.DataFrame()
combined_significant_bivariate_df = pd.DataFrame()
significant_in_both_tasks_bases = []
try:
    if os.path.exists(raw_bivar_file):
        all_raw_bivariate_results_df = pd.read_csv(raw_bivar_file, sep=';', decimal='.')
    if os.path.exists(sig_bivar_file_combined):
        combined_significant_bivariate_df = pd.read_csv(sig_bivar_file_combined, sep=';', decimal='.')
        if not combined_significant_bivariate_df.empty and 'Base Kinematic' in combined_significant_bivariate_df.columns and 'Task' in combined_significant_bivariate_df.columns:
            significance_counts = combined_significant_bivariate_df.groupby('Base Kinematic')['Task'].nunique()
            significant_in_both_tasks_bases = significance_counts[significance_counts == len(tasks)].index.tolist()
except Exception as e_load_bivar: print(f"Error loading pre-calculated bivariate results: {e_load_bivar}")

# ---------------------------------------------
# 2. Call Multivariate Model Pipelines
# ---------------------------------------------
if TARGET_IMAGING_COL not in df.columns:
    print(f"CRITICAL ERROR: '{TARGET_IMAGING_COL}' not found. Cannot run models."); sys.exit(1)

print("\n=== Calling PLS Correlation Pipeline ===")
pls_results_raw_dict = run_pls_pipeline(df=df, config=pls_config) # Store the raw dict
print("=== PLS Correlation Pipeline Finished ===")

# --- NEW: Save Full PLS Details ---
if pls_results_raw_dict:
    print("\n--- Saving Full PLS Details ---")
    all_tasks_pls_details_list = []
    for task_name_pls, task_data_pls in pls_results_raw_dict.items():
        lv_results_detail = task_data_pls.get('lv_results', {})
        kinematic_vars_pls = task_data_pls.get('kinematic_variables', [])
        
        for lv_num, lv_data_detail in lv_results_detail.items(): # Iterate through all LVs tested
            if lv_data_detail.get('x_loadings') is not None:
                loadings_series = lv_data_detail['x_loadings']
                # Ensure loadings_series uses the correct kinematic_vars_pls as index if it's just an array
                if not isinstance(loadings_series, pd.Series) and kinematic_vars_pls and len(loadings_series) == len(kinematic_vars_pls):
                    loadings_series = pd.Series(loadings_series, index=kinematic_vars_pls)
                elif not isinstance(loadings_series, pd.Series):
                    print(f"  Skipping LV{lv_num} for task {task_name_pls}: x_loadings not a Series or kinematic_vars mismatch.")
                    continue

                df_loadings_lv = loadings_series.reset_index()
                df_loadings_lv.columns = ['Kinematic_Variable', 'X_Loading']
                df_loadings_lv['LV'] = lv_num
                df_loadings_lv['Task'] = task_name_pls
                df_loadings_lv['LV_Significant'] = lv_data_detail.get('significant', False) # Was this LV significant?
                df_loadings_lv['LV_p_value'] = lv_data_detail.get('p_value')
                df_loadings_lv['LV_correlation_XYscores'] = lv_data_detail.get('correlation')
                df_loadings_lv['Y_Loading'] = lv_data_detail.get('y_loadings') # Usually a single value if Y is 1D

                if lv_data_detail.get('bootstrap_ratios') is not None:
                    bsr_series = lv_data_detail['bootstrap_ratios']
                    if not isinstance(bsr_series, pd.Series) and kinematic_vars_pls and len(bsr_series) == len(kinematic_vars_pls):
                         bsr_series = pd.Series(bsr_series, index=kinematic_vars_pls)
                    elif not isinstance(bsr_series, pd.Series):
                        print(f"  Note for LV{lv_num} task {task_name_pls}: bootstrap_ratios not a Series or kinematic_vars mismatch.")
                    
                    if isinstance(bsr_series, pd.Series):
                         df_loadings_lv['BSR'] = df_loadings_lv['Kinematic_Variable'].map(bsr_series).fillna(np.nan)
                    else:
                         df_loadings_lv['BSR'] = np.nan
                else:
                    df_loadings_lv['BSR'] = np.nan
                
                all_tasks_pls_details_list.append(df_loadings_lv)
            else:
                print(f"  No x_loadings found for Task {task_name_pls}, LV {lv_num}")
            
    if all_tasks_pls_details_list:
        full_pls_details_df = pd.concat(all_tasks_pls_details_list, ignore_index=True)
        # Order columns for readability
        cols_order_pls_full = [
            'Task', 'LV', 'Kinematic_Variable', 'X_Loading', 'BSR', 
            'LV_Significant', 'LV_p_value', 'LV_correlation_XYscores', 'Y_Loading'
        ]
        # Ensure all expected columns are present before reordering
        present_cols_pls_full = [col for col in cols_order_pls_full if col in full_pls_details_df.columns]
        full_pls_details_df = full_pls_details_df[present_cols_pls_full]

        full_pls_output_filename = os.path.join(data_output_folder, "pls_all_LVs_loadings_bsr_details.csv")
        try:
            full_pls_details_df.sort_values(by=['Task', 'LV', 'X_Loading'], ascending=[True, True, False], inplace=True)
            full_pls_details_df.to_csv(full_pls_output_filename, index=False, sep=';', decimal='.')
            print(f"Saved full PLS details (all LVs, loadings, BSRs) to: {full_pls_output_filename}")

            # Create the specific file ElasticNet expects for highlighting,
            # containing only features from significant LVs that meet BSR threshold for highlighting
            if PLS_RESULTS_FILE_PATH_FOR_ENET:
                enet_highlight_df = full_pls_details_df[
                    (full_pls_details_df['LV_Significant'] == True) &
                    (full_pls_details_df['BSR'].abs() >= PLS_BSR_THRESHOLD) # Use the defined BSR threshold
                ].copy()
                if not enet_highlight_df.empty:
                    # datnik_run_elasticnet.py expects columns like 'Task', 'LV', 'Kinematic_Variable', 'Loading', 'Bootstrap_Ratio'
                    # Ensure our columns match this expectation or adapt datnik_run_elasticnet.py
                    enet_highlight_df_renamed = enet_highlight_df.rename(columns={'BSR': 'Bootstrap_Ratio', 'X_Loading': 'Loading'})
                    cols_for_enet_file = ['Task', 'LV', 'Kinematic_Variable', 'Loading', 'Bootstrap_Ratio']
                    enet_highlight_df_final = enet_highlight_df_renamed[[col for col in cols_for_enet_file if col in enet_highlight_df_renamed.columns]]

                    enet_highlight_df_final.to_csv(PLS_RESULTS_FILE_PATH_FOR_ENET, index=False, sep=';', decimal='.')
                    print(f"Saved PLS features for ElasticNet highlighting (BSR >= {PLS_BSR_THRESHOLD}) to: {PLS_RESULTS_FILE_PATH_FOR_ENET}")
                else:
                    print(f"No PLS features met criteria for ElasticNet highlighting file (LV_Significant=True, |BSR|>={PLS_BSR_THRESHOLD}). File not created: {PLS_RESULTS_FILE_PATH_FOR_ENET}")


        except Exception as e_save_full_pls:
            print(f"Error saving full PLS details: {e_save_full_pls}")
    else:
        print("No PLS details to save (all_tasks_pls_details_list is empty).")
else:
    print("PLS analysis did not return results. Skipping full PLS detail saving.")
# --- END NEW ---

print("\n=== Calling ElasticNet Regression Pipeline ===")
enet_results = run_elasticnet_pipeline(df=df, config=enet_config)
print("=== ElasticNet Regression Pipeline Finished ===")

# ---------------------------------------------
# 3. Plotting (Bivariate TASK COMPARISON Plots)
# ---------------------------------------------
# ... (Rest of the script for Task Comparison Plots remains the same) ...
# ... (Ensure the end of the script is correctly pasted here) ...
if PLOT_AVAILABLE and significant_in_both_tasks_bases:
    if TARGET_IMAGING_COL not in df.columns:
        print("Skipping Bivariate Task Comparison plots: Target imaging column missing.")
    elif all_raw_bivariate_results_df.empty:
        print("Skipping Bivariate Task Comparison plots: Raw bivariate results data (for stats) missing or empty.")
    else:
        print(f"Plotting Bivariate Task Comparisons for {len(significant_in_both_tasks_bases)} base variables (OFF Data).")
        for base_k_col in significant_in_both_tasks_bases:
            ft_col_name = f"ft_{base_k_col}"
            hm_col_name = f"hm_{base_k_col}"
            if ft_col_name not in df.columns or hm_col_name not in df.columns:
                print(f"  Skipping comparison for {base_k_col}: Columns missing."); continue
            
            plot_data_for_comparison = df[[ft_col_name, hm_col_name, TARGET_IMAGING_COL]].copy()
            for col_proc in [ft_col_name, hm_col_name, TARGET_IMAGING_COL]:
                plot_data_for_comparison[col_proc] = pd.to_numeric(plot_data_for_comparison[col_proc], errors='coerce')
            plot_data_for_comparison.dropna(subset=[ft_col_name, hm_col_name, TARGET_IMAGING_COL], how='any', inplace=True)

            if plot_data_for_comparison.empty:
                print(f"  Skipping comparison for {base_k_col}: No valid data."); continue

            ft_stats_row = all_raw_bivariate_results_df[
                (all_raw_bivariate_results_df['Task'] == 'ft') &
                (all_raw_bivariate_results_df['Kinematic Variable'] == ft_col_name)]
            hm_stats_row = all_raw_bivariate_results_df[
                (all_raw_bivariate_results_df['Task'] == 'hm') &
                (all_raw_bivariate_results_df['Kinematic Variable'] == hm_col_name)]
            ft_stats_dict, hm_stats_dict = {}, {}
            if not ft_stats_row.empty:
                row = ft_stats_row.iloc[0]; ft_r = row.get('Pearson Correlation (r)', np.nan)
                ft_stats_dict = {'r': ft_r, 'p': row.get('P-value (uncorrected)', np.nan),
                                 'r2': ft_r**2 if pd.notna(ft_r) else np.nan, 'N': int(row.get('N', 0))}
            if not hm_stats_row.empty:
                row = hm_stats_row.iloc[0]; hm_r = row.get('Pearson Correlation (r)', np.nan)
                hm_stats_dict = {'r': hm_r, 'p': row.get('P-value (uncorrected)', np.nan),
                                 'r2': hm_r**2 if pd.notna(hm_r) else np.nan, 'N': int(row.get('N', 0))}

            comparison_plot_filename = f"task_comparison_{base_k_col}_vs_{TARGET_IMAGING_BASE}_OFF.png"
            print(f"  Plotting Bivariate Comparison: {base_k_col} (N plot points={len(plot_data_for_comparison)})")
            try:
                plot_task_comparison_scatter(
                    data=plot_data_for_comparison, ft_kinematic_col=ft_col_name, hm_kinematic_col=hm_col_name,
                    imaging_col=TARGET_IMAGING_COL, ft_stats=ft_stats_dict, hm_stats=hm_stats_dict,
                    output_folder=task_comparison_plots_folder, file_name=comparison_plot_filename)
            except Exception as e_plot_comp:
                 print(f"    ERROR generating task comparison plot for {base_k_col}: {e_plot_comp}")
elif not PLOT_AVAILABLE:
    print("Bivariate Task Comparison plotting skipped: Plotting functions unavailable.")
elif not significant_in_both_tasks_bases:
    print("Bivariate Task Comparison plotting skipped: No base variables found significant in both tasks.")
else:
    print("Bivariate Task Comparison plotting skipped due to missing prerequisites.")

print("--- Bivariate Task Comparison Plotting Finished ---")
print("\n--- Datnik Multivariate Models Script (PLS & ElasticNet, Full PLS Save) Finished ---")
# --- END OF FILE datnik_multivariate_models.py (Trimmed, Full PLS Save) ---