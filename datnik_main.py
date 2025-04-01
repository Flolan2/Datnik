#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performs bivariate correlation analysis (Strategy 1) and PLS analysis
(potentially multiple significant LVs) between kinematic variables
(FT & HM tasks) and contralateral Striatum Z-scores.
Generates plots for both types of analysis.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr # Keep for potential direct use or inspection

# --- Import the analysis functions ---
from datnik_analysis import run_correlation_analysis, run_pls_analysis
# --- Import the plotting functions ---
try:
    from datnik_plotting import plot_task_comparison_scatter, plot_pls_results
    PLOT_AVAILABLE = True
except ImportError:
    print("Warning: Required plotting functions not found in datnik_plotting.py.")
    print("Please ensure datnik_plotting.py exists and contains plot_task_comparison_scatter and plot_pls_results.")
    PLOT_AVAILABLE = False

# -------------------------
# 1. Load and Process Data
# -------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
merged_csv_file = os.path.join(script_dir, "Input", "merged_summary.csv")
print(f"Loading data from: {merged_csv_file}")
try:
    # Attempt to read with semicolon first, then comma
    try:
        df = pd.read_csv(merged_csv_file, sep=';', decimal='.')
        print("Read merged_summary.csv with ';' separator.")
    except pd.errors.ParserError:
        print("Warning: Failed to parse with ';'. Trying ',' separator...")
        df = pd.read_csv(merged_csv_file, sep=',', decimal='.')
        print("Read merged_summary.csv with ',' separator.")
    except Exception as read_err: # Catch other potential read errors
        print(f"Error reading {merged_csv_file}: {read_err}")
        exit()

    print(f"Data loaded successfully. Shape: {df.shape}")
    # Basic check for expected columns
    if 'Patient ID' not in df.columns: print("Warning: 'Patient ID' column missing.")
    if 'Contralateral_Striatum_Z' not in df.columns: print("Warning: Target imaging column 'Contralateral_Striatum_Z' missing.")

except FileNotFoundError:
    print(f"Error: Input file not found at {merged_csv_file}")
    exit()
except Exception as e:
    print(f"Error loading data from {merged_csv_file}: {e}")
    exit()

# Base kinematic variable names
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

# Output Folders
# Assume script is in a 'Code' or similar folder, place 'Output' parallel to it
script_parent_dir = os.path.dirname(script_dir)
output_base_folder = os.path.join(script_parent_dir, "Output")
data_output_folder = os.path.join(output_base_folder, "Data")
plots_folder = os.path.join(output_base_folder, "Plots")
os.makedirs(data_output_folder, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)
print(f"Output folders created/checked at: {output_base_folder}")

# Parameters
SIGNIFICANCE_ALPHA = 0.05
N_PERMUTATIONS = 100000
N_BOOTSTRAPS = 50000
MAX_PLS_COMPONENTS = 5 # Max components to test in PLS

# --- Data Storage ---
all_significant_bivariate_results = []
all_raw_bivariate_results_list = []
all_pls_results_dict = {}
# List to store DataFrames for the *combined* PLS results file
all_significant_pls_summary_for_combined_file = []

# ---------------------------------------------
# 2. Bivariate Correlation Analysis (Strategy 1)
# ---------------------------------------------
print("\n=== Starting Bivariate Correlation Analysis ===")
# Check if target imaging column exists before proceeding
if TARGET_IMAGING_COL not in df.columns:
    print(f"Error: Target imaging column '{TARGET_IMAGING_COL}' not found in DataFrame. Skipping bivariate analysis.")
else:
    for task in tasks:
        task_kinematic_cols = [f"{task}_{base}" for base in base_kinematic_cols]
        valid_task_cols = [col for col in task_kinematic_cols if col in df.columns]
        print(f"\n--- Task: {task.upper()} ---")
        print(f"Found {len(valid_task_cols)} kinematic columns for task {task} in the DataFrame.")
        if not valid_task_cols:
            print(f"Skipping task {task} for bivariate analysis - no valid kinematic columns found.")
            continue

        # Run analysis to get FDR-corrected significant results
        significant_results_task_df = run_correlation_analysis(
            df=df, base_kinematic_cols=base_kinematic_cols, task_prefix=task,
            imaging_base_name=TARGET_IMAGING_BASE, alpha=SIGNIFICANCE_ALPHA
        )

        # Calculate all raw correlations for this task (needed for plotting later)
        print(f"Calculating all raw correlations for task {task} for plotting lookup...")
        for base_col in base_kinematic_cols:
            kinematic_col = f"{task}_{base_col}"
            if kinematic_col not in df.columns: continue
            data_pair = df[[kinematic_col, TARGET_IMAGING_COL]].copy()
            try:
                 # Convert to numeric, coercing errors
                 data_pair[kinematic_col] = pd.to_numeric(data_pair[kinematic_col].astype(str).str.replace(',', '.'), errors='coerce')
                 data_pair[TARGET_IMAGING_COL] = pd.to_numeric(data_pair[TARGET_IMAGING_COL].astype(str).str.replace(',', '.'), errors='coerce')
            except Exception as e:
                 print(f"Warning: Error converting columns for {kinematic_col} / {TARGET_IMAGING_COL} to numeric: {e}")
                 continue
            data_pair.dropna(inplace=True)
            n_samples = len(data_pair)
            if n_samples >= 3: # Need at least 3 samples for correlation
                try:
                    corr_coef, p_value = pearsonr(data_pair[kinematic_col], data_pair[TARGET_IMAGING_COL])
                    if pd.notna(corr_coef) and pd.notna(p_value):
                         all_raw_bivariate_results_list.append({
                            "Task": task, "Base Kinematic": base_col,
                            "Pearson Correlation (r)": corr_coef,
                            "P-value (uncorrected)": p_value, "N": n_samples
                         })
                except ValueError: # Handle cases where correlation cannot be computed
                    continue

        # Save significant results for the task
        if not significant_results_task_df.empty:
            output_file = os.path.join(data_output_folder, f"significant_correlations_{TARGET_IMAGING_COL}_{task}.csv")
            try:
                significant_results_task_df.to_csv(output_file, index=False, sep=';', decimal='.')
                print(f"Significant bivariate results for task {task} saved to {output_file}")
                all_significant_bivariate_results.append(significant_results_task_df)
            except Exception as e:
                print(f"Error saving significant bivariate results for task {task}: {e}")
        else:
            print(f"No significant bivariate correlations found for task {task}.")

# Process combined bivariate results
all_raw_bivariate_results_df = pd.DataFrame(all_raw_bivariate_results_list)

if all_significant_bivariate_results:
    combined_significant_bivariate_df = pd.concat(all_significant_bivariate_results, ignore_index=True)
    # Identify variables significant in both tasks
    significance_counts = combined_significant_bivariate_df.groupby('Base Kinematic')['Task'].nunique()
    significant_in_both_tasks = significance_counts[significance_counts == 2].index.tolist()
    output_file_combined = os.path.join(data_output_folder, f"significant_correlations_{TARGET_IMAGING_COL}_combined.csv")
    try:
        combined_significant_bivariate_df.sort_values(by=['Base Kinematic', 'Task'], inplace=True)
        combined_significant_bivariate_df.to_csv(output_file_combined, index=False, sep=';', decimal='.')
        print(f"\nCombined significant bivariate results saved to {output_file_combined}")
        print(f"Base kinematic variables significant (bivariate, q<={SIGNIFICANCE_ALPHA}) in BOTH tasks: {significant_in_both_tasks}")
    except Exception as e:
         print(f"Error saving combined significant bivariate results: {e}")
else:
    combined_significant_bivariate_df = pd.DataFrame()
    significant_in_both_tasks = []
    print("\nNo significant bivariate correlations found in any task.")
print("=== Bivariate Correlation Analysis Finished ===")


# ---------------------------------------------
# 3. PLS Correlation Analysis
# ---------------------------------------------
print("\n=== Starting PLS Correlation Analysis ===")
if TARGET_IMAGING_COL not in df.columns:
     print(f"Error: Target imaging column '{TARGET_IMAGING_COL}' not found. Skipping PLS analysis.")
else:
    # List to store summary dataframes for the *combined* saving later
    all_significant_pls_summary_for_combined_file = []
    for task in tasks:
        print(f"\n--- Running PLS for Task: {task.upper()} ---")
        pls_results_task = run_pls_analysis(
            df=df,
            base_kinematic_cols=base_kinematic_cols,
            task_prefix=task,
            imaging_col=TARGET_IMAGING_COL,
            max_components=MAX_PLS_COMPONENTS,
            n_permutations=N_PERMUTATIONS,
            n_bootstraps=N_BOOTSTRAPS,
            alpha=SIGNIFICANCE_ALPHA
        )

        if pls_results_task:
            all_pls_results_dict[task] = pls_results_task # Store full results for plotting

            significant_lvs = pls_results_task.get('significant_lvs', [])
            if significant_lvs:
                print(f"Task {task.upper()}: Found {len(significant_lvs)} significant LVs: {significant_lvs}")

                # Prepare summary data for this task's significant LVs file
                task_pls_summary_list = []
                for lv_index in significant_lvs:
                    lv_data = pls_results_task['lv_results'].get(lv_index)
                    if not lv_data: continue # Should not happen if lv_index is in significant_lvs

                    lv_summary_base = {
                        'Task': task, 'LV_Index': lv_index,
                        'LV_Correlation': lv_data.get('correlation', np.nan),
                        'LV_P_Value': lv_data.get('p_value', np.nan),
                        'Y_Loading': lv_data.get('y_loadings', np.nan),
                        'N_Samples': pls_results_task.get('n_samples_pls', np.nan)
                    }
                    x_loadings = lv_data.get('x_loadings')
                    bsr = lv_data.get('bootstrap_ratios')

                    if x_loadings is not None and isinstance(x_loadings, pd.Series):
                         for k_var, loading in x_loadings.items():
                             row = lv_summary_base.copy()
                             row['Kinematic_Variable'] = k_var
                             row['X_Loading'] = loading
                             row['Bootstrap_Ratio'] = bsr.get(k_var, np.nan) if bsr is not None else np.nan
                             task_pls_summary_list.append(row)
                    else:
                         # If no X loadings (shouldn't happen for valid LV), save base info
                         task_pls_summary_list.append(lv_summary_base)

                # --- Create, Sort, and Save Task-Specific File ---
                if task_pls_summary_list:
                     task_pls_summary_df = pd.DataFrame(task_pls_summary_list)

                     # Sort the task-specific DataFrame
                     print(f"Sorting PLS results for task {task} by P-value and BSR magnitude...")
                     # Create temporary column for sorting by absolute BSR
                     task_pls_summary_df['Absolute_BSR'] = task_pls_summary_df['Bootstrap_Ratio'].abs()
                     # Sort: LV p-value ascending, then absolute BSR descending
                     task_pls_summary_df.sort_values(
                         by=['LV_P_Value', 'Absolute_BSR'],
                         ascending=[True, False], # Sort p-value lowest first, BSR highest first
                         inplace=True,
                         na_position='last' # Keep NaNs at the bottom of BSR sort
                     )
                     # Remove the temporary column
                     task_pls_summary_df.drop(columns=['Absolute_BSR'], inplace=True)

                     # Save the sorted task-specific file
                     task_filename = os.path.join(data_output_folder, f"pls_significant_results_{task}_sorted.csv")
                     try:
                         task_pls_summary_df.to_csv(task_filename, index=False, sep=';', decimal='.')
                         print(f"Sorted PLS results for task {task} saved to {task_filename}")
                     except Exception as e:
                         print(f"Error saving sorted PLS results for task {task}: {e}")

                     # Append the sorted DataFrame to the list for combined saving
                     all_significant_pls_summary_for_combined_file.append(task_pls_summary_df)
                 # -----------------------------------------------

            else:
                print(f"Task {task.upper()}: No significant LVs found (tested up to {pls_results_task.get('max_components_tested', 'N/A')} LVs).")
        else:
             print(f"PLS analysis failed or was skipped for task {task}.")

    # --- Save COMBINED significant PLS results ---
    if all_significant_pls_summary_for_combined_file:
        # Concatenate the (already sorted by task) task DataFrames
        combined_pls_summary_df = pd.concat(all_significant_pls_summary_for_combined_file, ignore_index=True)
        # Optional: could re-sort the combined file if a cross-task sort order is desired
        pls_filename_combined = os.path.join(data_output_folder, f"pls_significant_results_all_tasks_combined_sorted.csv")
        try:
            combined_pls_summary_df.to_csv(pls_filename_combined, index=False, sep=';', decimal='.')
            print(f"\nCombined (and sorted by task/p-value/BSR) significant PLS results saved to {pls_filename_combined}")
        except Exception as e:
            print(f"Error saving combined significant PLS results: {e}")
    else:
        print("\nNo significant PLS LVs found across all tasks to save.")
    # -----------------------------------------------

print("=== PLS Correlation Analysis Finished ===")


# ---------------------------------------------
# 4. Plotting Bivariate Results (Task Comparison - BOTH TASKS Significant)
# ---------------------------------------------
print("\n=== Generating Bivariate Plots (Task Comparison - Significant in BOTH) ===")
if PLOT_AVAILABLE and len(significant_in_both_tasks) > 0:
    if TARGET_IMAGING_COL not in df.columns:
        print(f"Error: Target imaging column '{TARGET_IMAGING_COL}' not found. Cannot generate bivariate plots.")
    else:
        if all_raw_bivariate_results_df.empty:
             print("Warning: Raw bivariate results data frame is empty. Cannot retrieve stats for plotting.")
        else:
            print(f"Plotting comparisons for {len(significant_in_both_tasks)} variables significant in both tasks.")
            for base_col in significant_in_both_tasks:
                print(f"  Plotting: {base_col}")
                ft_col = f"ft_{base_col}"
                hm_col = f"hm_{base_col}"
                # Double check columns exist in the original dataframe
                if ft_col not in df.columns:
                    print(f"    Skipping {base_col}: FT column '{ft_col}' not found in df.")
                    continue
                if hm_col not in df.columns:
                    print(f"    Skipping {base_col}: HM column '{hm_col}' not found in df.")
                    continue

                plot_data_raw = df[[ft_col, hm_col, TARGET_IMAGING_COL]].copy()
                try:
                    # Convert relevant columns to numeric for plotting
                    plot_data_raw[ft_col] = pd.to_numeric(plot_data_raw[ft_col].astype(str).str.replace(',', '.'), errors='coerce')
                    plot_data_raw[hm_col] = pd.to_numeric(plot_data_raw[hm_col].astype(str).str.replace(',', '.'), errors='coerce')
                    plot_data_raw[TARGET_IMAGING_COL] = pd.to_numeric(plot_data_raw[TARGET_IMAGING_COL].astype(str).str.replace(',', '.'), errors='coerce')
                except Exception as e:
                     print(f"    Warning: Error converting data to numeric for {base_col}. Skipping plot. Error: {e}")
                     continue

                # Retrieve stats from the pre-calculated raw results dataframe
                ft_stats_row = all_raw_bivariate_results_df[(all_raw_bivariate_results_df['Task'] == 'ft') & (all_raw_bivariate_results_df['Base Kinematic'] == base_col)]
                hm_stats_row = all_raw_bivariate_results_df[(all_raw_bivariate_results_df['Task'] == 'hm') & (all_raw_bivariate_results_df['Base Kinematic'] == base_col)]

                ft_stats_dict = {}
                if not ft_stats_row.empty:
                    ft_r = ft_stats_row.iloc[0].get('Pearson Correlation (r)', np.nan)
                    ft_p = ft_stats_row.iloc[0].get('P-value (uncorrected)', np.nan)
                    ft_n = ft_stats_row.iloc[0].get('N', 0)
                    ft_stats_dict = {'r': ft_r, 'p': ft_p, 'r2': ft_r**2 if pd.notna(ft_r) else np.nan, 'N': ft_n}

                hm_stats_dict = {}
                if not hm_stats_row.empty:
                    hm_r = hm_stats_row.iloc[0].get('Pearson Correlation (r)', np.nan)
                    hm_p = hm_stats_row.iloc[0].get('P-value (uncorrected)', np.nan)
                    hm_n = hm_stats_row.iloc[0].get('N', 0)
                    hm_stats_dict = {'r': hm_r, 'p': hm_p, 'r2': hm_r**2 if pd.notna(hm_r) else np.nan, 'N': hm_n}

                # Generate plot filename
                file_name = f"task_comparison_BOTH_SIG_{base_col}_vs_{TARGET_IMAGING_BASE}.png"
                plot_task_comparison_scatter(
                    data=plot_data_raw, ft_kinematic_col=ft_col, hm_kinematic_col=hm_col, imaging_col=TARGET_IMAGING_COL,
                    ft_stats=ft_stats_dict, hm_stats=hm_stats_dict, output_folder=plots_folder, file_name=file_name
                )
elif not PLOT_AVAILABLE:
    print("Bivariate plotting skipped because plotting functions were not found/imported.")
elif len(significant_in_both_tasks) == 0 :
    print("Bivariate plotting (Task Comparison) skipped because no kinematic variables were significant in BOTH tasks.")
print("=== Bivariate Plotting Finished ===")


# ---------------------------------------------
# 5. Plotting PLS Results (For Each Significant LV)
# ---------------------------------------------
print("\n=== Generating PLS Plots (For Each Significant LV) ===")
if PLOT_AVAILABLE:
    if not all_pls_results_dict:
        print("No PLS results dictionary available to generate plots.")
    else:
        any_plots_generated = False
        for task, pls_result_full in all_pls_results_dict.items():
            if not pls_result_full: continue # Skip if PLS failed for this task

            significant_lvs = pls_result_full.get('significant_lvs', [])
            if significant_lvs:
                print(f"\n--- Generating PLS plots for Task: {task.upper()} ---")
                for lv_index in significant_lvs:
                    print(f"  Plotting for LV{lv_index}...")
                    # Extract results specifically for this LV from the full dictionary
                    lv_specific_results = pls_result_full['lv_results'].get(lv_index)
                    kinematic_vars = pls_result_full.get('kinematic_variables') # Get list of kinematic var names used
                    n_samples = pls_result_full.get('n_samples_pls') # Get N for this analysis

                    if lv_specific_results and kinematic_vars is not None:
                        # Prepare data structure expected by the plotting function
                        plot_data = lv_specific_results.copy()
                        plot_data['kinematic_variables'] = kinematic_vars # Add list of var names
                        plot_data['n_samples_pls'] = n_samples # Add N
                        plot_data['task'] = task # Add task identifier

                        plot_pls_results(
                            pls_results_lv=plot_data, lv_index=lv_index,
                            output_folder=plots_folder, file_name_base=f"pls_{TARGET_IMAGING_BASE}",
                            bsr_threshold=2.0 # Standard BSR threshold often used
                        )
                        any_plots_generated = True
                    else:
                         print(f"    Skipping plot for LV{lv_index}: Missing results data or kinematic variable list.")

        if not any_plots_generated:
             print("No significant PLS LVs found across tasks, so no PLS plots were generated.")
elif not PLOT_AVAILABLE:
    print("PLS plotting skipped because plotting functions were not found/imported.")

print("\n--- Main script finished ---")