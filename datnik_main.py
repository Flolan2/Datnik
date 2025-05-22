# --- START OF FULL datnik_main.py ---

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performs Bivariate Correlation analysis between OFF-state kinematics
(FT & HM tasks) and contralateral Striatum Z-scores. Calls separate
scripts to perform Ridge, PLS, and ElasticNet analyses and plots.
Generates individual plots for significant bivariate findings.
"""

import os
import sys
import pandas as pd
import numpy as np
import json # For saving patient IDs

# --- Import analysis and plotting functions ---
try:
    from datnik_analysis import run_correlation_analysis
    from datnik_plotting import plot_task_comparison_scatter, plot_single_bivariate_scatter
    from datnik_run_ridge import run_ridge_pipeline
    from datnik_run_pls import run_pls_pipeline
    from datnik_run_elasticnet import run_elasticnet_pipeline
    PLOT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Error importing analysis/plotting/pipeline functions: {e}")
    print("Some functionality might be unavailable.")
    PLOT_AVAILABLE = False
    if 'run_correlation_analysis' not in locals():
        def run_correlation_analysis(*args, **kwargs): print("ERROR: run_correlation_analysis not imported."); return pd.DataFrame(), {} # Return tuple
    if 'plot_task_comparison_scatter' not in locals():
        def plot_task_comparison_scatter(*args, **kwargs): print("WARNING: plot_task_comparison_scatter not imported.")
    if 'plot_single_bivariate_scatter' not in locals():
        def plot_single_bivariate_scatter(*args, **kwargs): print("WARNING: plot_single_bivariate_scatter not imported.")
    if 'run_ridge_pipeline' not in locals():
        def run_ridge_pipeline(*args, **kwargs): print("ERROR: run_ridge_pipeline not imported."); return {}
    if 'run_pls_pipeline' not in locals():
        def run_pls_pipeline(*args, **kwargs): print("ERROR: run_pls_pipeline not imported."); return {}
    if 'run_elasticnet_pipeline' not in locals():
        def run_elasticnet_pipeline(*args, **kwargs): print("ERROR: run_elasticnet_pipeline not imported."); return {}

# -------------------------
# 1. Load and Process Data
# -------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
script_parent_dir = os.path.dirname(script_dir)
input_dir = os.path.join(script_parent_dir, "Input")
merged_csv_file = os.path.join(input_dir, "merged_summary_with_medon.csv")

print(f"Loading data from: {merged_csv_file}")
try:
    try: df_full = pd.read_csv(merged_csv_file, sep=';', decimal='.'); print("Read merged_summary CSV with ';' separator.")
    except (FileNotFoundError, pd.errors.ParserError, UnicodeDecodeError):
        print(f"Warning: Failed to parse/find with ';'. Trying ',' separator..."); df_full = pd.read_csv(merged_csv_file, sep=',', decimal='.'); print("Read merged_summary CSV with ',' separator.")
    except Exception as read_err: print(f"Error reading {merged_csv_file}: {read_err}"); sys.exit(1)
    print(f"Original data loaded successfully. Shape: {df_full.shape}")
    if 'Medication Condition' not in df_full.columns: print("CRITICAL ERROR: 'Medication Condition' column is missing."); sys.exit(1)
except FileNotFoundError: print(f"Error: Input file not found at {merged_csv_file}"); sys.exit(1)
except Exception as e: print(f"Error loading or processing data: {e}"); sys.exit(1)

print("\nFiltering data for Medication Condition == 'off'...")
df_full['Medication Condition'] = df_full['Medication Condition'].astype(str).str.strip().str.lower()
df = df_full[df_full['Medication Condition'] == 'off'].copy() # This is the main DataFrame for OFF analysis

if df.empty: print("Error: No data remaining after filtering for 'OFF' medication state. Exiting."); sys.exit(0)
else: print(f"Data filtered for 'OFF' state. New shape: {df.shape}"); print(f"Patients in OFF state data: {df['Patient ID'].nunique()}")

# --- Setup ---
base_kinematic_cols = [
    "meanamplitude", "stdamplitude", "meanspeed", "stdspeed", "meanrmsvelocity",
    "stdrmsvelocity", "meanopeningspeed", "stdopeningspeed", "meanclosingspeed",
    "stdclosingspeed", "meancycleduration", "stdcycleduration", "rangecycleduration",
    "rate", "amplitudedecay", "velocitydecay", "ratedecay", "cvamplitude",
    "cvcycleduration", "cvspeed", "cvrmsvelocity", "cvopeningspeed", "cvclosingspeed"
]
TARGET_IMAGING_BASE = "Contralateral_Striatum"
TARGET_IMAGING_COL = f"{TARGET_IMAGING_BASE}_Z"
PATIENT_ID_COL_NAME = "Patient ID" # Define the patient ID column name, ensure it matches your CSV
tasks = ['ft', 'hm']
output_base_folder = os.path.join(script_parent_dir, "Output")
data_output_folder = os.path.join(output_base_folder, "Data")
plots_folder = os.path.join(output_base_folder, "Plots")
os.makedirs(data_output_folder, exist_ok=True); os.makedirs(plots_folder, exist_ok=True)
print(f"Output folders created/checked at: {output_base_folder}")

SIGNIFICANCE_ALPHA = 0.05
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0); RIDGE_CV_FOLDS = 5
PLOT_TOP_N_RIDGE = 20; PLOT_BIVAR_TOP_N_LABEL = 7
PLS_MAX_COMPONENTS = 5; PLS_N_PERMUTATIONS = 1000; PLS_N_BOOTSTRAPS = 1000
PLS_ALPHA = 0.05; PLS_BSR_THRESHOLD = 2.0
PLS_RESULTS_FILE_PATH = os.path.join(data_output_folder, "pls_significant_results_all_tasks_combined_sorted.csv")
ENET_L1_RATIOS = np.linspace(0.1, 1.0, 10); ENET_CV_FOLDS = 5
ENET_MAX_ITER = 10000; ENET_RANDOM_STATE = 42; PLOT_TOP_N_ENET = 20

# --- Data Storage ---
all_significant_bivariate_results = []
all_raw_bivariate_results_list = []
all_bivariate_patient_ids = {} # To store patient IDs for all tasks' correlations

# ---------------------------------------------
# 2. Bivariate Correlation Analysis (OFF Data)
# ---------------------------------------------
print("\n=== Starting Bivariate Correlation Analysis (OFF Data Only) ===")
if TARGET_IMAGING_COL not in df.columns:
    print(f"Error: Target imaging column '{TARGET_IMAGING_COL}' not found. Skipping bivariate analysis.")
    all_raw_bivariate_results_df = pd.DataFrame(); combined_significant_bivariate_df = pd.DataFrame(); significant_in_both_tasks = []
else:
    for task in tasks:
        print(f"\n--- Task: {task.upper()} ---")
        if not any(f"{task}_{base}" in df.columns for base in base_kinematic_cols):
            print(f"Skipping task {task}: No valid kinematic columns found in OFF-state data."); continue

        # Run Bivariate Analysis (FDR corrected) & get patient IDs
        significant_results_task_df, patient_ids_this_task_corr = run_correlation_analysis(
            df=df, base_kinematic_cols=base_kinematic_cols, task_prefix=task,
            imaging_base_name=TARGET_IMAGING_BASE, alpha=SIGNIFICANCE_ALPHA,
            patient_id_col=PATIENT_ID_COL_NAME # Pass the defined patient ID column name
        )
        all_bivariate_patient_ids.update(patient_ids_this_task_corr) # Store patient IDs

        # Calculate and Store Raw Correlations (Uncorrected)
        print(f"Calculating all raw correlations for task {task} (OFF Data)...")
        for base_col in base_kinematic_cols:
            kinematic_col = f"{task}_{base_col}"
            if kinematic_col not in df.columns: continue
            data_pair = df[[kinematic_col, TARGET_IMAGING_COL]].copy()
            try:
                 data_pair[kinematic_col] = pd.to_numeric(data_pair[kinematic_col].astype(str).str.replace(',', '.'), errors='coerce')
                 data_pair[TARGET_IMAGING_COL] = pd.to_numeric(data_pair[TARGET_IMAGING_COL].astype(str).str.replace(',', '.'), errors='coerce')
                 data_pair.dropna(inplace=True); n_samples = len(data_pair)
                 if n_samples >= 3:
                     from scipy.stats import pearsonr
                     corr_coef, p_value = pearsonr(data_pair[kinematic_col], data_pair[TARGET_IMAGING_COL])
                     if pd.notna(corr_coef) and pd.notna(p_value):
                         all_raw_bivariate_results_list.append({
                             "Task": task, "Kinematic Variable": kinematic_col,
                             "Pearson Correlation (r)": corr_coef, "P-value (uncorrected)": p_value, "N": n_samples
                         })
            except Exception: continue

        # Save Significant Results Table & Plot
        if not significant_results_task_df.empty:
            output_file = os.path.join(data_output_folder, f"significant_correlations_{TARGET_IMAGING_COL}_{task}_OFF.csv")
            try:
                significant_results_task_df.to_csv(output_file, index=False, sep=';', decimal='.')
                print(f"Significant bivariate results table for task {task} saved to {output_file}")
                all_significant_bivariate_results.append(significant_results_task_df)
            except Exception as e: print(f"Error saving significant bivariate results table for {task}: {e}")

            if PLOT_AVAILABLE:
                print(f"Generating individual plots for significant bivariate correlations for task {task.upper()}...")
                task_plots_folder = os.path.join(plots_folder, task.upper()); os.makedirs(task_plots_folder, exist_ok=True)
                for index, row in significant_results_task_df.iterrows():
                    kinematic_col_to_plot = row['Kinematic Variable']
                    if kinematic_col_to_plot not in df.columns: print(f"    Skipping plot: Column '{kinematic_col_to_plot}' not found."); continue
                    plot_data_subset = df[[kinematic_col_to_plot, TARGET_IMAGING_COL]].copy()
                    try:
                        plot_data_subset[kinematic_col_to_plot] = pd.to_numeric(plot_data_subset[kinematic_col_to_plot].astype(str).str.replace(',', '.'), errors='coerce')
                        plot_data_subset[TARGET_IMAGING_COL] = pd.to_numeric(plot_data_subset[TARGET_IMAGING_COL].astype(str).str.replace(',', '.'), errors='coerce')
                        plot_data_subset.dropna(inplace=True)
                    except Exception as convert_err: print(f"    Skipping plot for {kinematic_col_to_plot}: Error converting data - {convert_err}"); continue
                    stats_dict = {'r': row.get('Pearson Correlation (r)', np.nan), 'p': row.get('P-value (uncorrected)', np.nan),
                                  'q': row.get('Q-value (FDR corrected)', np.nan), 'N': int(row.get('N', 0))}
                    file_name = f"bivar_scatter_{kinematic_col_to_plot}_vs_{TARGET_IMAGING_BASE}_OFF.png"
                    try: plot_single_bivariate_scatter(data=plot_data_subset, kinematic_col=kinematic_col_to_plot, imaging_col=TARGET_IMAGING_COL,
                                                       stats_dict=stats_dict, output_folder=task_plots_folder, file_name=file_name)
                    except Exception as plot_err: print(f"    ERROR generating plot for {kinematic_col_to_plot}: {plot_err}")
                print(f"Finished generating individual bivariate plots for task {task.upper()}.")
            else: print(f"Skipping individual bivariate plots for task {task.upper()}: Plotting functions unavailable.")
        else: print(f"No significant bivariate correlations found for task {task}.")

    # Save all raw bivariate results
    all_raw_bivariate_results_df = pd.DataFrame(all_raw_bivariate_results_list)
    # (Optional save of raw results)

    # Combine and save significant results from all tasks
    if all_significant_bivariate_results:
        combined_significant_bivariate_df = pd.concat(all_significant_bivariate_results, ignore_index=True)
        if not combined_significant_bivariate_df.empty and 'Kinematic Variable' in combined_significant_bivariate_df.columns:
            combined_significant_bivariate_df['Base Kinematic'] = combined_significant_bivariate_df['Kinematic Variable'].apply(lambda x: x.split('_', 1)[1] if '_' in x else x)
            significance_counts = combined_significant_bivariate_df.groupby('Base Kinematic')['Task'].nunique()
            significant_in_both_tasks = significance_counts[significance_counts == 2].index.tolist()
        else: significant_in_both_tasks = []
        output_file_combined = os.path.join(data_output_folder, f"significant_correlations_{TARGET_IMAGING_COL}_combined_OFF.csv")
        try:
            if 'Base Kinematic' in combined_significant_bivariate_df.columns: combined_significant_bivariate_df.sort_values(by=['Base Kinematic', 'Task'], inplace=True)
            combined_significant_bivariate_df.to_csv(output_file_combined, index=False, sep=';', decimal='.')
            print(f"\nCombined significant bivariate results (OFF Data) saved to {output_file_combined}")
            print(f"Base kinematic variables significant (bivariate) in BOTH tasks (OFF Data): {significant_in_both_tasks}")
        except Exception as e: print(f"Error saving combined significant bivariate results: {e}")
    else: combined_significant_bivariate_df = pd.DataFrame(); significant_in_both_tasks = []; print("\nNo significant bivariate correlations found in any task (OFF Data).")

    # Save the collected Patient IDs for bivariate correlations
    if all_bivariate_patient_ids:
        bivar_pids_json_path = os.path.join(data_output_folder, f"bivariate_correlations_patient_ids_OFF.json")
        try:
            all_bivariate_patient_ids_str_keys = {str(k): v for k, v in all_bivariate_patient_ids.items()} # Convert tuple keys to string for JSON
            with open(bivar_pids_json_path, 'w') as f: json.dump(all_bivariate_patient_ids_str_keys, f, indent=4)
            print(f"\nPatient IDs for bivariate correlations saved to JSON: {bivar_pids_json_path}")
        except Exception as e: print(f"\nError saving bivariate Patient IDs as JSON: {e}")
        
        bivar_pids_list_for_csv = []
        for (kin_var, img_var), pids in all_bivariate_patient_ids.items():
            bivar_pids_list_for_csv.append({"Kinematic_Variable": kin_var, "Imaging_Variable": img_var, 
                                           "N_Patients_In_Correlation": len(pids), "Patient_IDs": ",".join(sorted(pids))})
        if bivar_pids_list_for_csv:
            bivar_pids_df_out = pd.DataFrame(bivar_pids_list_for_csv)
            bivar_pids_csv_path = os.path.join(data_output_folder, f"bivariate_correlations_patient_ids_OFF.csv")
            try:
                bivar_pids_df_out.sort_values(by=["Kinematic_Variable", "Imaging_Variable"], inplace=True)
                bivar_pids_df_out.to_csv(bivar_pids_csv_path, index=False, sep=';')
                print(f"Patient IDs for bivariate correlations saved to CSV: {bivar_pids_csv_path}")
            except Exception as e: print(f"\nError saving bivariate Patient IDs as CSV: {e}")
    else: print("\nNo patient ID information collected for bivariate correlations.")

print("=== Bivariate Correlation Analysis Finished (OFF Data Only) ===")

# --- Prepare Config and Data for Ridge/PLS/ENet Scripts ---
ridge_config = {'tasks': tasks, 'base_kinematic_cols': base_kinematic_cols, 'TARGET_IMAGING_COL': TARGET_IMAGING_COL, 
                'RIDGE_ALPHAS': RIDGE_ALPHAS, 'RIDGE_CV_FOLDS': RIDGE_CV_FOLDS, 'data_output_folder': data_output_folder,
                'plots_folder': plots_folder, 'PLOT_TOP_N_RIDGE': PLOT_TOP_N_RIDGE, 'PLOT_BIVAR_TOP_N_LABEL': PLOT_BIVAR_TOP_N_LABEL}
bivar_data_for_ridge = {'raw_df': all_raw_bivariate_results_df, 'significant_df': combined_significant_bivariate_df}
pls_config = {'tasks': tasks, 'base_kinematic_cols': base_kinematic_cols, 'TARGET_IMAGING_COL': TARGET_IMAGING_COL, 
              'PLS_MAX_COMPONENTS': PLS_MAX_COMPONENTS, 'PLS_N_PERMUTATIONS': PLS_N_PERMUTATIONS, 'PLS_N_BOOTSTRAPS': PLS_N_BOOTSTRAPS,
              'PLS_ALPHA': PLS_ALPHA, 'PLS_BSR_THRESHOLD': PLS_BSR_THRESHOLD, 'data_output_folder': data_output_folder, 'plots_folder': plots_folder}
enet_config = {'tasks': tasks, 'base_kinematic_cols': base_kinematic_cols, 'TARGET_IMAGING_COL': TARGET_IMAGING_COL,
               'ENET_L1_RATIOS': ENET_L1_RATIOS, 'ENET_CV_FOLDS': ENET_CV_FOLDS, 'ENET_MAX_ITER': ENET_MAX_ITER, 
               'ENET_RANDOM_STATE': ENET_RANDOM_STATE, 'data_output_folder': data_output_folder, 'plots_folder': plots_folder,
               'PLOT_TOP_N_ENET': PLOT_TOP_N_ENET, 'PLS_RESULTS_FILE': PLS_RESULTS_FILE_PATH, 'PLS_BSR_THRESHOLD': PLS_BSR_THRESHOLD}

# ---------------------------------------------
# 3. Call Ridge Regression Pipeline Script
# ---------------------------------------------
print("\n=== Calling Ridge Regression Pipeline ===")
ridge_results = run_ridge_pipeline(df=df, config=ridge_config, bivariate_results=bivar_data_for_ridge)
print("=== Ridge Regression Pipeline Finished ===")

# ---------------------------------------------
# 4. Call PLS Correlation Pipeline Script
# ---------------------------------------------
print("\n=== Calling PLS Correlation Pipeline ===")
pls_results = run_pls_pipeline(df=df, config=pls_config)
print("=== PLS Correlation Pipeline Finished ===")

# ---------------------------------------------
# 5. Call ElasticNet Regression Pipeline Script
# ---------------------------------------------
print("\n=== Calling ElasticNet Regression Pipeline ===")
enet_results = run_elasticnet_pipeline(df=df, config=enet_config)
print("=== ElasticNet Regression Pipeline Finished ===")

# ---------------------------------------------
# 6. Plotting (Only Bivariate TASK COMPARISON Plots)
# ---------------------------------------------
print("\n=== Generating Bivariate TASK COMPARISON Plots (OFF Data Only) ===")
print("\n--- Generating Bivariate Task Comparison Plots ---")
if PLOT_AVAILABLE and 'significant_in_both_tasks' in locals() and significant_in_both_tasks:
    if TARGET_IMAGING_COL not in df.columns: print("Skipping Bivariate Task Comparison plots: Target imaging column missing.")
    elif not isinstance(all_raw_bivariate_results_df, pd.DataFrame) or all_raw_bivariate_results_df.empty: print("Skipping Bivariate Task Comparison plots: Raw bivariate results data missing or empty.")
    else:
        print(f"Plotting Bivariate Task Comparisons for {len(significant_in_both_tasks)} variables significant in both tasks (OFF Data).")
        for base_col in significant_in_both_tasks:
            print(f"  Plotting Bivariate Comparison: {base_col}")
            ft_col = f"ft_{base_col}"; hm_col = f"hm_{base_col}"
            if ft_col not in df.columns or hm_col not in df.columns: print(f"    Skipping {base_col}: Columns missing."); continue
            plot_data_raw = df[[ft_col, hm_col, TARGET_IMAGING_COL]].copy()
            try:
                plot_data_raw[ft_col] = pd.to_numeric(plot_data_raw[ft_col].astype(str).str.replace(',', '.'), errors='coerce')
                plot_data_raw[hm_col] = pd.to_numeric(plot_data_raw[hm_col].astype(str).str.replace(',', '.'), errors='coerce')
                plot_data_raw[TARGET_IMAGING_COL] = pd.to_numeric(plot_data_raw[TARGET_IMAGING_COL].astype(str).str.replace(',', '.'), errors='coerce')
                plot_data_raw.dropna(subset=[ft_col, hm_col, TARGET_IMAGING_COL], how='any', inplace=True)
            except Exception as e: print(f"    Warning: Error converting plot data for {base_col}. Error: {e}"); continue
            if plot_data_raw.empty: print(f"    Skipping {base_col}: No valid data points after NaN removal."); continue

            ft_stats_row = all_raw_bivariate_results_df[(all_raw_bivariate_results_df['Task'] == 'ft') & (all_raw_bivariate_results_df['Kinematic Variable'] == ft_col)]
            hm_stats_row = all_raw_bivariate_results_df[(all_raw_bivariate_results_df['Task'] == 'hm') & (all_raw_bivariate_results_df['Kinematic Variable'] == hm_col)]
            ft_stats_dict = {}; hm_stats_dict = {}
            if not ft_stats_row.empty: ft_r = ft_stats_row.iloc[0].get('Pearson Correlation (r)', np.nan); ft_p = ft_stats_row.iloc[0].get('P-value (uncorrected)', np.nan); ft_n = ft_stats_row.iloc[0].get('N', np.nan); ft_r2 = ft_r**2 if pd.notna(ft_r) else np.nan; ft_stats_dict = {'r': ft_r, 'p': ft_p, 'r2': ft_r2, 'N': int(ft_n) if pd.notna(ft_n) else 0}
            if not hm_stats_row.empty: hm_r = hm_stats_row.iloc[0].get('Pearson Correlation (r)', np.nan); hm_p = hm_stats_row.iloc[0].get('P-value (uncorrected)', np.nan); hm_n = hm_stats_row.iloc[0].get('N', np.nan); hm_r2 = hm_r**2 if pd.notna(hm_r) else np.nan; hm_stats_dict = {'r': hm_r, 'p': hm_p, 'r2': hm_r2, 'N': int(hm_n) if pd.notna(hm_n) else 0}
            file_name = f"task_comparison_BOTH_SIG_{base_col}_vs_{TARGET_IMAGING_BASE}_OFF.png"
            plot_task_comparison_scatter(data=plot_data_raw, ft_kinematic_col=ft_col, hm_kinematic_col=hm_col, imaging_col=TARGET_IMAGING_COL,
                                         ft_stats=ft_stats_dict, hm_stats=hm_stats_dict, output_folder=plots_folder, file_name=file_name)
elif not PLOT_AVAILABLE: print("Bivariate Task Comparison plotting skipped: Plotting functions unavailable.")
elif 'significant_in_both_tasks' in locals() and not significant_in_both_tasks : print("Bivariate Task Comparison plotting skipped: No variables significant in BOTH tasks (OFF Data).")
elif 'significant_in_both_tasks' not in locals(): print("Bivariate Task Comparison plotting skipped: Prerequisite data not available.")
print("--- Bivariate Task Comparison Plotting Finished ---")

print("\n--- Datnik Main Script Finished ---")

# --- END OF FULL datnik_main.py ---