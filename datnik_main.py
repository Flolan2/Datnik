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

# --- START OF CORRECTED SECTION in datnik_main.py ---

# --- Import analysis and plotting functions ---
try:
    # Bivariate functions needed in main
    from datnik_analysis import run_correlation_analysis
    # <<< Import both plotting functions >>>
    from datnik_plotting import plot_task_comparison_scatter, plot_single_bivariate_scatter
    # Import pipeline runners for other analyses
    from datnik_run_ridge import run_ridge_pipeline
    from datnik_run_pls import run_pls_pipeline
    from datnik_run_elasticnet import run_elasticnet_pipeline
    PLOT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Error importing analysis/plotting/pipeline functions: {e}")
    print("Some functionality might be unavailable.")
    PLOT_AVAILABLE = False
    # Define dummies if imports failed
    # --- CORRECTED DUMMY DEFINITIONS ---
    if 'run_correlation_analysis' not in locals():
        def run_correlation_analysis(*args, **kwargs):
            print("ERROR: run_correlation_analysis not imported.")
            return pd.DataFrame()
    if 'plot_task_comparison_scatter' not in locals():
        def plot_task_comparison_scatter(*args, **kwargs):
            print("WARNING: plot_task_comparison_scatter not imported. Task comparison plots skipped.")
    # <<< ADD DUMMY FOR NEW PLOT FUNCTION >>>
    if 'plot_single_bivariate_scatter' not in locals():
        def plot_single_bivariate_scatter(*args, **kwargs):
            print("WARNING: plot_single_bivariate_scatter not imported. Individual bivariate plots skipped.")
    # --- End new dummy ---
    if 'run_ridge_pipeline' not in locals():
        def run_ridge_pipeline(*args, **kwargs):
            print("ERROR: run_ridge_pipeline not imported. Ridge analysis skipped.")
            return {}
    if 'run_pls_pipeline' not in locals():
        def run_pls_pipeline(*args, **kwargs):
            print("ERROR: run_pls_pipeline not imported. PLS analysis skipped.")
            return {}
    if 'run_elasticnet_pipeline' not in locals():
        def run_elasticnet_pipeline(*args, **kwargs):
            print("ERROR: run_elasticnet_pipeline not imported. ElasticNet analysis skipped.")
            return {}
    # --- END CORRECTION ---

# --- END OF CORRECTED SECTION in datnik_main.py ---
# -------------------------
# 1. Load and Process Data
# -------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
script_parent_dir = os.path.dirname(script_dir)
input_dir = os.path.join(script_parent_dir, "Input")
merged_csv_file = os.path.join(input_dir, "merged_summary_with_medon.csv")

print(f"Loading data from: {merged_csv_file}")
try:
    # Try reading with semicolon first
    try:
        df_full = pd.read_csv(merged_csv_file, sep=';', decimal='.')
        print("Read merged_summary CSV with ';' separator.")
    except (FileNotFoundError, pd.errors.ParserError, UnicodeDecodeError):
        # Fallback to comma separator if semicolon fails or file not found with semicolon
        print(f"Warning: Failed to parse/find '{os.path.basename(merged_csv_file)}' with ';'. Trying ',' separator...")
        df_full = pd.read_csv(merged_csv_file, sep=',', decimal='.')
        print("Read merged_summary CSV with ',' separator.")
    except Exception as read_err: # Catch other potential reading errors
        print(f"Error reading {merged_csv_file}: {read_err}")
        sys.exit(1)

    print(f"Original data loaded successfully. Shape: {df_full.shape}")
    # Check for essential column *after* successful loading
    if 'Medication Condition' not in df_full.columns:
         print("CRITICAL ERROR: 'Medication Condition' column is missing.")
         sys.exit(1)

except FileNotFoundError: # Catch specifically if file doesn't exist at all
    print(f"Error: Input file not found at {merged_csv_file}")
    sys.exit(1)
except Exception as e: # Catch any other unexpected errors during loading block
    print(f"Error loading or processing data from {merged_csv_file}: {e}")
    sys.exit(1)

# Filter for OFF medication state
print("\nFiltering data for Medication Condition == 'off'...")
df_full['Medication Condition'] = df_full['Medication Condition'].astype(str).str.strip().str.lower()
df = df_full[df_full['Medication Condition'] == 'off'].copy()

if df.empty:
    print("Error: No data remaining after filtering for 'OFF' medication state. Exiting.")
    sys.exit(0) # Exit gracefully if no OFF data
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
plots_folder = os.path.join(output_base_folder, "Plots") # Main plots folder
os.makedirs(data_output_folder, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)
print(f"Output folders created/checked at: {output_base_folder}")

# Parameters
SIGNIFICANCE_ALPHA = 0.05
# --- RidgeCV parameters ---
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0)
RIDGE_CV_FOLDS = 5
PLOT_TOP_N_RIDGE = 20
PLOT_BIVAR_TOP_N_LABEL = 7
# --- PLS parameters ---
PLS_MAX_COMPONENTS = 5
PLS_N_PERMUTATIONS = 1000
PLS_N_BOOTSTRAPS = 1000
PLS_ALPHA = 0.05
PLS_BSR_THRESHOLD = 2.0
PLS_RESULTS_FILE_PATH = os.path.join(data_output_folder, "pls_significant_results_all_tasks_combined_sorted.csv")

# --- ElasticNet parameters ---
ENET_L1_RATIOS = np.linspace(0.1, 1.0, 10) # Example: 10 ratios from 0.1 to 1.0
ENET_CV_FOLDS = 5 # Can reuse CV folds or set separately
ENET_MAX_ITER = 10000
ENET_RANDOM_STATE = 42
PLOT_TOP_N_ENET = 20 # How many non-zero coeffs to plot

# --- Data Storage ---
all_significant_bivariate_results = [] # Stores dataframes of significant results per task
all_raw_bivariate_results_list = [] # Stores list of dicts for *all* raw correlations

# ---------------------------------------------
# 2. Bivariate Correlation Analysis (OFF Data)
# ---------------------------------------------
print("\n=== Starting Bivariate Correlation Analysis (OFF Data Only) ===")
if TARGET_IMAGING_COL not in df.columns:
    print(f"Error: Target imaging column '{TARGET_IMAGING_COL}' not found. Skipping bivariate analysis.")
    all_raw_bivariate_results_df = pd.DataFrame() # Initialize empty df
    combined_significant_bivariate_df = pd.DataFrame() # Initialize empty df
    significant_in_both_tasks = [] # Initialize empty list
else:
    # --- Bivariate Loop ---
    for task in tasks:
        task_kinematic_cols = [f"{task}_{base}" for base in base_kinematic_cols]
        valid_task_cols = [col for col in task_kinematic_cols if col in df.columns]
        print(f"\n--- Task: {task.upper()} ---")
        if not valid_task_cols:
            print(f"Skipping task {task}: No valid kinematic columns found in OFF-state data.")
            continue

        # --- Run Bivariate Analysis (FDR corrected) ---
        significant_results_task_df = run_correlation_analysis(
            df=df, base_kinematic_cols=base_kinematic_cols, task_prefix=task,
            imaging_base_name=TARGET_IMAGING_BASE, alpha=SIGNIFICANCE_ALPHA
        )

        # --- Calculate and Store Raw Correlations (Uncorrected) ---
        print(f"Calculating all raw correlations for task {task} (OFF Data)...")
        for base_col in base_kinematic_cols:
            kinematic_col = f"{task}_{base_col}"
            if kinematic_col not in df.columns: continue
            data_pair = df[[kinematic_col, TARGET_IMAGING_COL]].copy()
            try:
                 # Robust conversion to numeric
                 data_pair[kinematic_col] = pd.to_numeric(data_pair[kinematic_col].astype(str).str.replace(',', '.'), errors='coerce')
                 data_pair[TARGET_IMAGING_COL] = pd.to_numeric(data_pair[TARGET_IMAGING_COL].astype(str).str.replace(',', '.'), errors='coerce')
                 data_pair.dropna(inplace=True) # Drop rows with NaN in either column
                 n_samples = len(data_pair)
                 if n_samples >= 3: # Need at least 3 pairs for correlation
                     # Local import if needed and not globally imported
                     from scipy.stats import pearsonr
                     corr_coef, p_value = pearsonr(data_pair[kinematic_col], data_pair[TARGET_IMAGING_COL])
                     # Ensure correlation and p-value are valid numbers
                     if pd.notna(corr_coef) and pd.notna(p_value):
                         all_raw_bivariate_results_list.append({
                             "Task": task,
                             "Kinematic Variable": kinematic_col,
                             "Pearson Correlation (r)": corr_coef,
                             "P-value (uncorrected)": p_value,
                             "N": n_samples
                         })
            except Exception as corr_err:
                # Optionally log the error: print(f" Could not process {kinematic_col}: {corr_err}")
                continue # Ignore errors for single pairs and continue

        # --- Save Significant Results Table ---
        if not significant_results_task_df.empty:
            # Save the table of significant results
            output_file = os.path.join(data_output_folder, f"significant_correlations_{TARGET_IMAGING_COL}_{task}_OFF.csv")
            try:
                significant_results_task_df.to_csv(output_file, index=False, sep=';', decimal='.')
                print(f"Significant bivariate results table for task {task} saved to {output_file}")
                all_significant_bivariate_results.append(significant_results_task_df) # Append df to list
            except Exception as e:
                print(f"Error saving significant bivariate results table for {task}: {e}")

            # <<< --- START: NEW PLOTTING LOOP FOR SIGNIFICANT BIVARIATE RESULTS --- >>>
            if PLOT_AVAILABLE:
                print(f"Generating individual plots for significant bivariate correlations for task {task.upper()}...")
                # Create task-specific subfolder for plots
                task_plots_folder = os.path.join(plots_folder, task.upper()) # e.g., Output/Plots/FT or Output/Plots/HM
                os.makedirs(task_plots_folder, exist_ok=True)

                # Iterate through the significant results dataframe for this task
                for index, row in significant_results_task_df.iterrows():
                    kinematic_col_to_plot = row['Kinematic Variable']
                    print(f"  Plotting: {kinematic_col_to_plot} vs {TARGET_IMAGING_COL}")

                    # Double-check if column exists in the original dataframe 'df'
                    if kinematic_col_to_plot not in df.columns:
                        print(f"    Skipping plot: Column '{kinematic_col_to_plot}' not found in main dataframe.")
                        continue

                    # Prepare data subset for this specific plot from the main dataframe 'df'
                    plot_data_subset = df[[kinematic_col_to_plot, TARGET_IMAGING_COL]].copy()
                    try:
                        # Ensure data is numeric, coercing errors
                        plot_data_subset[kinematic_col_to_plot] = pd.to_numeric(plot_data_subset[kinematic_col_to_plot].astype(str).str.replace(',', '.'), errors='coerce')
                        plot_data_subset[TARGET_IMAGING_COL] = pd.to_numeric(plot_data_subset[TARGET_IMAGING_COL].astype(str).str.replace(',', '.'), errors='coerce')
                        # Drop rows where *either* column became NaN after conversion
                        plot_data_subset.dropna(inplace=True)
                    except Exception as convert_err:
                        print(f"    Skipping plot for {kinematic_col_to_plot}: Error converting data to numeric - {convert_err}")
                        continue

                    # Extract stats needed for annotation from the results row
                    stats_dict = {
                        'r': row.get('Pearson Correlation (r)', np.nan),
                        'p': row.get('P-value (uncorrected)', np.nan),
                        'q': row.get('Q-value (FDR corrected)', np.nan), # Include FDR q-value
                        'N': int(row.get('N', 0)) # Ensure N is integer
                    }

                    # Define filename for the plot
                    file_name = f"bivar_scatter_{kinematic_col_to_plot}_vs_{TARGET_IMAGING_BASE}_OFF.png"

                    # Call the plotting function to generate and save the scatter plot
                    try:
                        plot_single_bivariate_scatter(
                            data=plot_data_subset,
                            kinematic_col=kinematic_col_to_plot,
                            imaging_col=TARGET_IMAGING_COL,
                            stats_dict=stats_dict,
                            output_folder=task_plots_folder, # Save to task-specific folder
                            file_name=file_name
                        )
                    except Exception as plot_err:
                         print(f"    ERROR generating plot for {kinematic_col_to_plot}: {plot_err}")

                print(f"Finished generating individual bivariate plots for task {task.upper()}.")
            else:
                 print(f"Skipping individual bivariate plots for task {task.upper()}: Plotting functions unavailable.")
            # <<< --- END: NEW PLOTTING LOOP --- >>>

        else:
            print(f"No significant bivariate correlations found for task {task}.") # Message if analysis yielded no significant results

    # --- Combine and Save Overall Bivariate Results (Significant & Raw) ---
    # Create DataFrame from the list of raw correlation dicts
    all_raw_bivariate_results_df = pd.DataFrame(all_raw_bivariate_results_list)
    # Save raw results if desired (optional)
    # raw_output_file = os.path.join(data_output_folder, f"raw_correlations_{TARGET_IMAGING_COL}_combined_OFF.csv")
    # try:
    #     all_raw_bivariate_results_df.sort_values(by=['Task', 'Kinematic Variable'], inplace=True)
    #     all_raw_bivariate_results_df.to_csv(raw_output_file, index=False, sep=';', decimal='.')
    #     print(f"\nAll raw bivariate results saved to {raw_output_file}")
    # except Exception as e:
    #     print(f"Error saving raw bivariate results: {e}")


    # Combine significant results from all tasks if any were found
    if all_significant_bivariate_results:
        combined_significant_bivariate_df = pd.concat(all_significant_bivariate_results, ignore_index=True)

        # Identify features significant in both tasks
        def get_base_name(feature_name):
            """Helper to extract base kinematic name from task-prefixed name."""
            parts = feature_name.split('_', 1)
            return parts[1] if len(parts) > 1 else feature_name

        # Ensure required column exists before processing
        if not combined_significant_bivariate_df.empty and 'Kinematic Variable' in combined_significant_bivariate_df.columns:
            combined_significant_bivariate_df['Base Kinematic'] = combined_significant_bivariate_df['Kinematic Variable'].apply(get_base_name)
            significance_counts = combined_significant_bivariate_df.groupby('Base Kinematic')['Task'].nunique()
            significant_in_both_tasks = significance_counts[significance_counts == 2].index.tolist()
        else:
            significant_in_both_tasks = [] # Ensure it's defined as empty list

        # Save combined significant results table
        output_file_combined = os.path.join(data_output_folder, f"significant_correlations_{TARGET_IMAGING_COL}_combined_OFF.csv")
        try:
            # Sort for better readability
            if 'Base Kinematic' in combined_significant_bivariate_df.columns:
                combined_significant_bivariate_df.sort_values(by=['Base Kinematic', 'Task'], inplace=True)
            combined_significant_bivariate_df.to_csv(output_file_combined, index=False, sep=';', decimal='.')
            print(f"\nCombined significant bivariate results (OFF Data) saved to {output_file_combined}")
            print(f"Base kinematic variables significant (bivariate) in BOTH tasks (OFF Data): {significant_in_both_tasks}")
        except Exception as e:
            print(f"Error saving combined significant bivariate results: {e}")
    else:
        # Ensure these variables exist even if no significant results were found
        combined_significant_bivariate_df = pd.DataFrame()
        significant_in_both_tasks = []
        print("\nNo significant bivariate correlations found in any task (OFF Data).")

print("=== Bivariate Correlation Analysis Finished (OFF Data Only) ===")


# --- Prepare Config and Data for Ridge/PLS/ENet Scripts ---
ridge_config = {
    'tasks': tasks, 'base_kinematic_cols': base_kinematic_cols,
    'TARGET_IMAGING_COL': TARGET_IMAGING_COL, 'RIDGE_ALPHAS': RIDGE_ALPHAS,
    'RIDGE_CV_FOLDS': RIDGE_CV_FOLDS, 'data_output_folder': data_output_folder,
    'plots_folder': plots_folder, 'PLOT_TOP_N_RIDGE': PLOT_TOP_N_RIDGE,
    'PLOT_BIVAR_TOP_N_LABEL': PLOT_BIVAR_TOP_N_LABEL,
}
# Use the DataFrames created/populated above
bivar_data_for_ridge = {
    'raw_df': all_raw_bivariate_results_df, # Always pass the raw df (might be empty)
    'significant_df': combined_significant_bivariate_df # Always pass the combined sig df (might be empty)
}
pls_config = {
    'tasks': tasks, 'base_kinematic_cols': base_kinematic_cols,
    'TARGET_IMAGING_COL': TARGET_IMAGING_COL, 'PLS_MAX_COMPONENTS': PLS_MAX_COMPONENTS,
    'PLS_N_PERMUTATIONS': PLS_N_PERMUTATIONS, 'PLS_N_BOOTSTRAPS': PLS_N_BOOTSTRAPS,
    'PLS_ALPHA': PLS_ALPHA, 'PLS_BSR_THRESHOLD': PLS_BSR_THRESHOLD,
    'data_output_folder': data_output_folder, 'plots_folder': plots_folder,
}
enet_config = {
    'tasks': tasks,
    'base_kinematic_cols': base_kinematic_cols,
    'TARGET_IMAGING_COL': TARGET_IMAGING_COL,
    'ENET_L1_RATIOS': ENET_L1_RATIOS,
    'ENET_CV_FOLDS': ENET_CV_FOLDS,
    'ENET_MAX_ITER': ENET_MAX_ITER,
    'ENET_RANDOM_STATE': ENET_RANDOM_STATE,
    'data_output_folder': data_output_folder,
    'plots_folder': plots_folder,
    'PLOT_TOP_N_ENET': PLOT_TOP_N_ENET,
    'PLS_RESULTS_FILE': PLS_RESULTS_FILE_PATH,
    'PLS_BSR_THRESHOLD': PLS_BSR_THRESHOLD
}

# ---------------------------------------------
# 3. Call Ridge Regression Pipeline Script
# ---------------------------------------------
print("\n=== Calling Ridge Regression Pipeline ===")
ridge_results = run_ridge_pipeline(
    df=df, config=ridge_config, bivariate_results=bivar_data_for_ridge
)
print("=== Ridge Regression Pipeline Finished ===")

# ---------------------------------------------
# 4. Call PLS Correlation Pipeline Script
# ---------------------------------------------
print("\n=== Calling PLS Correlation Pipeline ===")
pls_results = run_pls_pipeline(
    df=df, config=pls_config
)
print("=== PLS Correlation Pipeline Finished ===")

# ---------------------------------------------
# 5. Call ElasticNet Regression Pipeline Script
# ---------------------------------------------
print("\n=== Calling ElasticNet Regression Pipeline ===")
enet_results = run_elasticnet_pipeline(
    df=df, config=enet_config
)
print("=== ElasticNet Regression Pipeline Finished ===")

# ---------------------------------------------
# 6. Plotting (Only Bivariate TASK COMPARISON Plots)
#    (Individual significant plots are now done within section 2)
# ---------------------------------------------
print("\n=== Generating Bivariate TASK COMPARISON Plots (OFF Data Only) ===")
print("\n--- Generating Bivariate Task Comparison Plots ---")
# Check necessary conditions for plotting task comparisons
if PLOT_AVAILABLE and 'significant_in_both_tasks' in locals() and significant_in_both_tasks: # Checks list exists and is not empty
    if TARGET_IMAGING_COL not in df.columns:
        print("Skipping Bivariate Task Comparison plots: Target imaging column missing.")
    elif not isinstance(all_raw_bivariate_results_df, pd.DataFrame) or all_raw_bivariate_results_df.empty:
        print("Skipping Bivariate Task Comparison plots: Raw bivariate results data missing or empty.")
    else:
        print(f"Plotting Bivariate Task Comparisons for {len(significant_in_both_tasks)} variables significant in both tasks (OFF Data).")
        # Loop through base kinematic names significant in both tasks
        for base_col in significant_in_both_tasks:
            print(f"  Plotting Bivariate Comparison: {base_col}")
            ft_col = f"ft_{base_col}"
            hm_col = f"hm_{base_col}"

            # Ensure both task columns exist in the primary dataframe 'df'
            if ft_col not in df.columns or hm_col not in df.columns:
                print(f"    Skipping {base_col}: One or both columns ({ft_col}, {hm_col}) missing in OFF-state df.")
                continue

            # Prepare data subset, convert to numeric, drop NaNs
            plot_data_raw = df[[ft_col, hm_col, TARGET_IMAGING_COL]].copy()
            try:
                plot_data_raw[ft_col] = pd.to_numeric(plot_data_raw[ft_col].astype(str).str.replace(',', '.'), errors='coerce')
                plot_data_raw[hm_col] = pd.to_numeric(plot_data_raw[hm_col].astype(str).str.replace(',', '.'), errors='coerce')
                plot_data_raw[TARGET_IMAGING_COL] = pd.to_numeric(plot_data_raw[TARGET_IMAGING_COL].astype(str).str.replace(',', '.'), errors='coerce')
                # Drop rows if *any* of the three essential columns are NaN
                plot_data_raw.dropna(subset=[ft_col, hm_col, TARGET_IMAGING_COL], how='any', inplace=True)
            except Exception as e:
                 print(f"    Warning: Error converting plot data to numeric for {base_col}. Skipping. Error: {e}")
                 continue

            # Check if data remains after cleaning
            if plot_data_raw.empty:
                 print(f"    Skipping {base_col}: No valid data points after NaN removal for plotting.")
                 continue

            # Fetch stats from the raw results dataframe (which should always exist)
            ft_stats_row = all_raw_bivariate_results_df[
                (all_raw_bivariate_results_df['Task'] == 'ft') &
                (all_raw_bivariate_results_df['Kinematic Variable'] == ft_col)
            ]
            hm_stats_row = all_raw_bivariate_results_df[
                (all_raw_bivariate_results_df['Task'] == 'hm') &
                (all_raw_bivariate_results_df['Kinematic Variable'] == hm_col)
            ]

            # Prepare stats dictionaries safely for annotation
            ft_stats_dict = {}
            if not ft_stats_row.empty:
                ft_r = ft_stats_row.iloc[0].get('Pearson Correlation (r)', np.nan)
                ft_p = ft_stats_row.iloc[0].get('P-value (uncorrected)', np.nan)
                ft_n = ft_stats_row.iloc[0].get('N', np.nan)
                # Calculate R^2 only if r is valid
                ft_r2 = ft_r**2 if pd.notna(ft_r) else np.nan
                ft_stats_dict = {'r': ft_r, 'p': ft_p, 'r2': ft_r2, 'N': int(ft_n) if pd.notna(ft_n) else 0}

            hm_stats_dict = {}
            if not hm_stats_row.empty:
                hm_r = hm_stats_row.iloc[0].get('Pearson Correlation (r)', np.nan)
                hm_p = hm_stats_row.iloc[0].get('P-value (uncorrected)', np.nan)
                hm_n = hm_stats_row.iloc[0].get('N', np.nan)
                # Calculate R^2 only if r is valid
                hm_r2 = hm_r**2 if pd.notna(hm_r) else np.nan
                hm_stats_dict = {'r': hm_r, 'p': hm_p, 'r2': hm_r2, 'N': int(hm_n) if pd.notna(hm_n) else 0}

            # Define filename and call plotting function
            file_name = f"task_comparison_BOTH_SIG_{base_col}_vs_{TARGET_IMAGING_BASE}_OFF.png"
            plot_task_comparison_scatter(
                data=plot_data_raw, ft_kinematic_col=ft_col, hm_kinematic_col=hm_col,
                imaging_col=TARGET_IMAGING_COL, ft_stats=ft_stats_dict, hm_stats=hm_stats_dict,
                output_folder=plots_folder, # Save comparison plots to main plots folder
                file_name=file_name
            )

elif not PLOT_AVAILABLE:
    print("Bivariate Task Comparison plotting skipped: Plotting functions unavailable.")
# Check if list exists but is empty
elif 'significant_in_both_tasks' in locals() and not significant_in_both_tasks :
    print("Bivariate Task Comparison plotting skipped: No variables significant in BOTH tasks (OFF Data).")
# Check if list was never created (e.g., if imaging column was missing)
elif 'significant_in_both_tasks' not in locals():
     print("Bivariate Task Comparison plotting skipped: Prerequisite data ('significant_in_both_tasks') not available.")

print("--- Bivariate Task Comparison Plotting Finished ---")


print("\n--- Datnik Main Script Finished ---")

# --- END OF FULL datnik_main.py ---