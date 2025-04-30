# --- START OF CLEANED datnik_main.py (OFF Data, Bivariate + Ridge) ---

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performs Bivariate Correlation and Ridge Regression analyses between
OFF-state kinematics (FT & HM tasks) and contralateral Striatum Z-scores.

Generates plots for Bivariate analysis and Ridge Regression coefficients.
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# --- Import the analysis and plotting functions ---
try:
    # Only import needed functions
    from datnik_analysis import run_correlation_analysis, run_ridge_analysis
    from datnik_plotting import plot_task_comparison_scatter, plot_ridge_coefficients, plot_bivariate_vs_ridge_scatter # Added Ridge plot func
    PLOT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Error importing analysis/plotting functions: {e}")
    PLOT_AVAILABLE = False


# -------------------------
# 1. Load and Process Data
# -------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
script_parent_dir = os.path.dirname(script_dir)
input_dir = os.path.join(script_parent_dir, "Input")
merged_csv_file = os.path.join(input_dir, "merged_summary_with_medon.csv") # Ensure correct name

print(f"Loading data from: {merged_csv_file}")
try: # Load data... (Keep the loading logic as it was)
    try:
        df_full = pd.read_csv(merged_csv_file, sep=';', decimal='.')
        print("Read merged_summary CSV with ';' separator.")
    except (FileNotFoundError, pd.errors.ParserError, UnicodeDecodeError):
        print(f"Warning: Failed to parse/find '{os.path.basename(merged_csv_file)}' with ';'. Trying ',' separator...")
        df_full = pd.read_csv(merged_csv_file, sep=',', decimal='.')
        print("Read merged_summary CSV with ',' separator.")
    except Exception as read_err:
        print(f"Error reading {merged_csv_file}: {read_err}")
        sys.exit(1)
    print(f"Original data loaded successfully. Shape: {df_full.shape}")
    if 'Medication Condition' not in df_full.columns:
         print("CRITICAL ERROR: 'Medication Condition' column is missing.")
         sys.exit(1)
except FileNotFoundError:
    print(f"Error: Input file not found at {merged_csv_file}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading data from {merged_csv_file}: {e}")
    sys.exit(1)

# Filter for OFF medication state
print("\nFiltering data for Medication Condition == 'off'...")
df_full['Medication Condition'] = df_full['Medication Condition'].astype(str).str.strip().str.lower()
df = df_full[df_full['Medication Condition'] == 'off'].copy()

if df.empty:
    print("Error: No data remaining after filtering for 'OFF' medication state. Exiting.")
    sys.exit(0)
else:
     print(f"Data filtered for 'OFF' state. New shape: {df.shape}")
     print(f"Patients in OFF state data: {df['Patient ID'].nunique()}")

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
tasks = ['ft', 'hm']
output_base_folder = os.path.join(script_parent_dir, "Output")
data_output_folder = os.path.join(output_base_folder, "Data")
plots_folder = os.path.join(output_base_folder, "Plots")
os.makedirs(data_output_folder, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)
print(f"Output folders created/checked at: {output_base_folder}")

# Parameters
SIGNIFICANCE_ALPHA = 0.05
# RidgeCV parameters
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0) # Example alphas
RIDGE_CV_FOLDS = 5
PLOT_TOP_N_RIDGE = 20 # How many coefficients to show in Ridge plot

# --- Data Storage ---
all_significant_bivariate_results = []
all_raw_bivariate_results_list = []
all_ridge_results_dict = {}

# ---------------------------------------------
# 2. Bivariate Correlation Analysis (OFF Data)
# ---------------------------------------------
print("\n=== Starting Bivariate Correlation Analysis (OFF Data Only) ===")
if TARGET_IMAGING_COL not in df.columns:
    print(f"Error: Target imaging column '{TARGET_IMAGING_COL}' not found in OFF-state DataFrame. Skipping bivariate analysis.")
else:
    for task in tasks:
        task_kinematic_cols = [f"{task}_{base}" for base in base_kinematic_cols]
        valid_task_cols = [col for col in task_kinematic_cols if col in df.columns]
        print(f"\n--- Task: {task.upper()} ---")
        if not valid_task_cols:
            print(f"Skipping task {task}: No valid columns found in OFF-state data.")
            continue

        # Run analysis using the filtered df - this function returns df with 'Kinematic Variable' column
        significant_results_task_df = run_correlation_analysis(
            df=df, base_kinematic_cols=base_kinematic_cols, task_prefix=task,
            imaging_base_name=TARGET_IMAGING_BASE, alpha=SIGNIFICANCE_ALPHA
        )

        # Calculate all raw correlations using the filtered df
        print(f"Calculating all raw correlations for task {task} (OFF Data)...")
        for base_col in base_kinematic_cols:
            kinematic_col = f"{task}_{base_col}" # Full name is here
            if kinematic_col not in df.columns: continue
            data_pair = df[[kinematic_col, TARGET_IMAGING_COL]].copy()
            try:
                 # Convert to numeric, coercing errors
                 data_pair[kinematic_col] = pd.to_numeric(data_pair[kinematic_col].astype(str).str.replace(',', '.'), errors='coerce')
                 data_pair[TARGET_IMAGING_COL] = pd.to_numeric(data_pair[TARGET_IMAGING_COL].astype(str).str.replace(',', '.'), errors='coerce')
                 data_pair.dropna(inplace=True)
                 n_samples = len(data_pair)
                 if n_samples >= 3:
                     corr_coef, p_value = pearsonr(data_pair[kinematic_col], data_pair[TARGET_IMAGING_COL])
                     if pd.notna(corr_coef) and pd.notna(p_value):
                         # <<< --- STORING FULL KINEMATIC NAME --- >>>
                         all_raw_bivariate_results_list.append({
                             "Task": task,
                             # "Base Kinematic": base_col, # Optional: Keep if you need the base name elsewhere
                             "Kinematic Variable": kinematic_col, # STORE THE FULL NAME HERE
                             "Pearson Correlation (r)": corr_coef,
                             "P-value (uncorrected)": p_value,
                             "N": n_samples
                         })
                         # <<< --- END CHANGE --- >>>
            except Exception: # Catch potential errors during calculation
                # print(f"  Warn: Could not calculate correlation for {kinematic_col}") # Optional verbose warning
                continue

        # Save significant results for the task (already contains 'Kinematic Variable')
        if not significant_results_task_df.empty:
            output_file = os.path.join(data_output_folder, f"significant_correlations_{TARGET_IMAGING_COL}_{task}_OFF.csv")
            try:
                significant_results_task_df.to_csv(output_file, index=False, sep=';', decimal='.')
                print(f"Significant bivariate results for task {task} saved to {output_file}")
                all_significant_bivariate_results.append(significant_results_task_df)
            except Exception as e:
                print(f"Error saving significant bivariate results for {task}: {e}")
        else:
            print(f"No significant bivariate correlations found for task {task}.")

# Process combined bivariate results
all_raw_bivariate_results_df = pd.DataFrame(all_raw_bivariate_results_list) # Contains 'Kinematic Variable'
if all_significant_bivariate_results:
    combined_significant_bivariate_df = pd.concat(all_significant_bivariate_results, ignore_index=True) # Also contains 'Kinematic Variable'

    # --- Determine significance in both tasks ---
    # We need to extract the base name again if we want to group by it
    # Define a helper function or use apply
    def get_base_name(feature_name):
        parts = feature_name.split('_', 1)
        return parts[1] if len(parts) > 1 else feature_name

    if not combined_significant_bivariate_df.empty and 'Kinematic Variable' in combined_significant_bivariate_df.columns:
        combined_significant_bivariate_df['Base Kinematic'] = combined_significant_bivariate_df['Kinematic Variable'].apply(get_base_name)
        significance_counts = combined_significant_bivariate_df.groupby('Base Kinematic')['Task'].nunique()
        significant_in_both_tasks = significance_counts[significance_counts == 2].index.tolist()
    else:
        significant_in_both_tasks = [] # Handle case where df is empty or column missing

    # Save combined results
    output_file_combined = os.path.join(data_output_folder, f"significant_correlations_{TARGET_IMAGING_COL}_combined_OFF.csv")
    try:
        # Sort by base name then task for readability
        if 'Base Kinematic' in combined_significant_bivariate_df.columns:
            combined_significant_bivariate_df.sort_values(by=['Base Kinematic', 'Task'], inplace=True)
        combined_significant_bivariate_df.to_csv(output_file_combined, index=False, sep=';', decimal='.')
        print(f"\nCombined significant bivariate results (OFF Data) saved to {output_file_combined}")
        print(f"Base kinematic variables significant (bivariate) in BOTH tasks (OFF Data): {significant_in_both_tasks}")
    except Exception as e:
        print(f"Error saving combined bivariate results: {e}")
else:
    # Ensure these variables exist even if no significant results were found
    combined_significant_bivariate_df = pd.DataFrame()
    significant_in_both_tasks = []
    print("\nNo significant bivariate correlations found in any task (OFF Data).")
print("=== Bivariate Correlation Analysis Finished (OFF Data Only) ===")

# --- End of Updated Section 2 ---
# ---------------------------------------------
# 3. Ridge Regression Analysis (OFF Data)
# ---------------------------------------------
print("\n=== Starting Ridge Regression Analysis (OFF Data Only) ===")
if TARGET_IMAGING_COL not in df.columns:
     print(f"Error: Target imaging column '{TARGET_IMAGING_COL}' not found. Skipping Ridge analysis.")
else:
    for task in tasks:
        print(f"\n--- Running Ridge Regression for Task: {task.upper()} (OFF Data Only) ---")
        ridge_results_task = run_ridge_analysis(
            df=df,
            base_kinematic_cols=base_kinematic_cols,
            task_prefix=task,
            imaging_col=TARGET_IMAGING_COL,
            alphas=RIDGE_ALPHAS,
            cv_folds=RIDGE_CV_FOLDS
        )

        if ridge_results_task:
            all_ridge_results_dict[task] = ridge_results_task # Store results
            # Save Ridge coefficients
            ridge_filename_base = os.path.join(data_output_folder, f"ridge_{task}_OFF")
            try:
                 coeffs_series = ridge_results_task.get('coefficients')
                 if coeffs_series is not None and isinstance(coeffs_series, pd.Series):
                      coeffs_df = coeffs_series.reset_index()
                      coeffs_df.columns = ['Feature', 'Coefficient']
                      coeffs_df['Optimal_Alpha'] = ridge_results_task.get('optimal_alpha')
                      coeffs_df['R2_Full_Data'] = ridge_results_task.get('r2_full_data')
                      coeffs_df.sort_values(by='Coefficient', key=abs, ascending=False, inplace=True)
                      coeffs_df.to_csv(f"{ridge_filename_base}_coefficients.csv", sep=';', decimal='.', index=False)
                      print(f"  Saved Ridge coefficients to {ridge_filename_base}_coefficients.csv")
                 else: print("  No coefficients found in Ridge results to save.")
            except Exception as e: print(f"  Error saving Ridge coefficients for task {task}: {e}")
        else: print(f"Ridge Regression analysis failed or was skipped for task {task}.")

print("=== Ridge Regression Analysis Finished (OFF Data Only) ===")


# --- Corrected Section 4 for datnik_main.py ---

# ---------------------------------------------
# 4. Plotting (Bivariate, Ridge Coeffs, Comparison Scatter)
# ---------------------------------------------
print("\n=== Generating Plots (OFF Data Only) ===")

# --- Plot 1: Bivariate Task Comparison (for vars significant in BOTH tasks) ---
print("\n--- Generating Bivariate Task Comparison Plots ---")
if PLOT_AVAILABLE and 'significant_in_both_tasks' in locals() and len(significant_in_both_tasks) > 0:
    # Check if necessary dataframes are available
    if TARGET_IMAGING_COL not in df.columns:
        print("Skipping Bivariate Task Comparison plots: Target imaging column missing.")
    elif 'all_raw_bivariate_results_df' not in locals() or all_raw_bivariate_results_df.empty:
        print("Skipping Bivariate Task Comparison plots: Raw bivariate results data missing or empty.")
    else:
        print(f"Plotting Bivariate Task Comparisons for {len(significant_in_both_tasks)} variables significant in both tasks (OFF Data).")
        for base_col in significant_in_both_tasks:
            print(f"  Plotting Bivariate Comparison: {base_col}")
            ft_col = f"ft_{base_col}"
            hm_col = f"hm_{base_col}"

            # Ensure columns exist in the filtered DataFrame 'df'
            if ft_col not in df.columns or hm_col not in df.columns:
                print(f"    Skipping {base_col}: Column missing in OFF-state df.")
                continue

            # Prepare data for this specific plot
            plot_data_raw = df[[ft_col, hm_col, TARGET_IMAGING_COL]].copy()
            try: # Ensure data is numeric for plotting
                plot_data_raw[ft_col] = pd.to_numeric(plot_data_raw[ft_col].astype(str).str.replace(',', '.'), errors='coerce')
                plot_data_raw[hm_col] = pd.to_numeric(plot_data_raw[hm_col].astype(str).str.replace(',', '.'), errors='coerce')
                plot_data_raw[TARGET_IMAGING_COL] = pd.to_numeric(plot_data_raw[TARGET_IMAGING_COL].astype(str).str.replace(',', '.'), errors='coerce')
                # Drop NaNs specific to this plot's data
                plot_data_raw.dropna(subset=[ft_col, hm_col, TARGET_IMAGING_COL], how='any', inplace=True)

            except Exception as e:
                 print(f"    Warning: Error converting plot data to numeric for {base_col}. Skipping. Error: {e}")
                 continue

            if plot_data_raw.empty:
                 print(f"    Skipping {base_col}: No valid data points after NaN removal for plotting.")
                 continue

            # Retrieve stats using the correct keys from all_raw_bivariate_results_df
            ft_stats_row = all_raw_bivariate_results_df[
                (all_raw_bivariate_results_df['Task'] == 'ft') &
                (all_raw_bivariate_results_df['Kinematic Variable'] == ft_col) # Match full name
            ]
            hm_stats_row = all_raw_bivariate_results_df[
                (all_raw_bivariate_results_df['Task'] == 'hm') &
                (all_raw_bivariate_results_df['Kinematic Variable'] == hm_col) # Match full name
            ]

            # Create stats dictionaries expected by the plotting function
            ft_stats_dict = {}
            if not ft_stats_row.empty:
                ft_r = ft_stats_row.iloc[0].get('Pearson Correlation (r)', np.nan)
                ft_p = ft_stats_row.iloc[0].get('P-value (uncorrected)', np.nan)
                ft_n = ft_stats_row.iloc[0].get('N', np.nan) # Use N from the specific correlation calc
                ft_stats_dict = {'r': ft_r, 'p': ft_p, 'r2': ft_r**2 if pd.notna(ft_r) else np.nan, 'N': int(ft_n) if pd.notna(ft_n) else 0}

            hm_stats_dict = {}
            if not hm_stats_row.empty:
                hm_r = hm_stats_row.iloc[0].get('Pearson Correlation (r)', np.nan)
                hm_p = hm_stats_row.iloc[0].get('P-value (uncorrected)', np.nan)
                hm_n = hm_stats_row.iloc[0].get('N', np.nan) # Use N from the specific correlation calc
                hm_stats_dict = {'r': hm_r, 'p': hm_p, 'r2': hm_r**2 if pd.notna(hm_r) else np.nan, 'N': int(hm_n) if pd.notna(hm_n) else 0}

            # Generate plot filename
            file_name = f"task_comparison_BOTH_SIG_{base_col}_vs_{TARGET_IMAGING_BASE}_OFF.png"

            # Call the plotting function
            plot_task_comparison_scatter(
                data=plot_data_raw, # Use the potentially smaller df specific to this plot
                ft_kinematic_col=ft_col,
                hm_kinematic_col=hm_col,
                imaging_col=TARGET_IMAGING_COL,
                ft_stats=ft_stats_dict,
                hm_stats=hm_stats_dict,
                output_folder=plots_folder,
                file_name=file_name
            )

elif not PLOT_AVAILABLE:
    print("Bivariate Task Comparison plotting skipped: Plotting functions unavailable.")
elif 'significant_in_both_tasks' not in locals() or len(significant_in_both_tasks) == 0 :
    print("Bivariate Task Comparison plotting skipped: No variables significant in BOTH tasks (OFF Data).")
print("--- Bivariate Task Comparison Plotting Finished ---")


# --- Plot 2: Ridge Coefficient Plotting ---
print("\n--- Generating Ridge Coefficient Plots ---")
if PLOT_AVAILABLE:
    if 'all_ridge_results_dict' not in locals() or not all_ridge_results_dict:
        print("Skipping Ridge plots: Ridge results dictionary missing or empty.")
    else:
        for task, ridge_result_full in all_ridge_results_dict.items():
             if ridge_result_full:
                 print(f"Generating Ridge plot for Task: {task.upper()} (OFF Data)...")
                 plot_ridge_coefficients(
                     ridge_results_task=ridge_result_full,
                     top_n=PLOT_TOP_N_RIDGE,
                     output_folder=plots_folder,
                     file_name_base="ridge_coefficients" # Will add _{task}_OFF.png
                 )
             else:
                 print(f"Skipping Ridge plot for task {task}: No results found.")
elif not PLOT_AVAILABLE:
    print("Ridge plotting skipped: Plotting functions unavailable.")
print("--- Ridge Coefficient Plotting Finished ---")


# --- Plot 3: Bivariate vs Ridge Comparison Scatter Plot ---
print("\n--- Generating Bivariate vs Ridge Comparison Scatter Plots ---")
if PLOT_AVAILABLE:
    # Check necessary variables exist before proceeding
    if 'all_raw_bivariate_results_df' not in locals() or all_raw_bivariate_results_df.empty:
        print("Skipping Bivar vs Ridge plot: Raw bivariate results missing or empty.")
    elif 'all_ridge_results_dict' not in locals() or not all_ridge_results_dict:
         print("Skipping Bivar vs Ridge plot: Ridge results missing or empty.")
    else:
        # Pass the combined DF containing FDR results for highlighting points
        fdr_sig_df = combined_significant_bivariate_df if 'combined_significant_bivariate_df' in locals() else pd.DataFrame()

        for task in tasks:
            if task in all_ridge_results_dict:
                plot_bivariate_vs_ridge_scatter(
                    bivariate_results_df=all_raw_bivariate_results_df,
                    ridge_results_task=all_ridge_results_dict[task],
                    significant_bivariate_df=fdr_sig_df, # Pass the FDR results
                    task_prefix=task,
                    top_n_label=7, # Adjust how many points to label
                    output_folder=plots_folder
                    # file_name_base default is okay
                )
            else:
                print(f"Skipping Bivar vs Ridge plot for task {task}: Ridge results not found.")

elif not PLOT_AVAILABLE:
    print("Bivariate vs Ridge comparison plotting skipped: Plotting functions unavailable.")
print("--- Bivariate vs Ridge Plotting Finished ---")


# --- End of Corrected Section 4 ---