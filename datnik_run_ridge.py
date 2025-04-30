#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 14:56:22 2025

@author: Lange_L
"""

# --- START OF FILE datnik_run_ridge.py ---

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performs Ridge Regression analysis for specified tasks (e.g., FT, HM)
predicting an imaging variable from kinematic features.
Includes saving coefficients and generating relevant plots.
"""

import os
import pandas as pd
import numpy as np

# --- Import analysis and plotting functions ---
# Assume these are in the same directory or correctly in the Python path
try:
    from datnik_analysis import run_ridge_analysis
    from datnik_plotting import plot_ridge_coefficients, plot_bivariate_vs_ridge_scatter
    PLOT_AVAILABLE = True # Assume plotting is available if imports succeed
except ImportError as e:
    print(f"Warning: Could not import analysis/plotting functions in datnik_run_ridge.py: {e}")
    print("Plot generation will be skipped.")
    # Define dummy functions if needed, or just rely on PLOT_AVAILABLE flag
    def run_ridge_analysis(*args, **kwargs): return None
    def plot_ridge_coefficients(*args, **kwargs): pass
    def plot_bivariate_vs_ridge_scatter(*args, **kwargs): pass
    PLOT_AVAILABLE = False

def run_ridge_pipeline(
    df: pd.DataFrame,
    config: dict,
    bivariate_results: dict = None
):
    """
    Executes the Ridge Regression analysis pipeline for specified tasks.

    Args:
        df (pd.DataFrame): The input DataFrame (typically filtered for OFF state).
        config (dict): A dictionary containing configuration parameters like:
            - tasks (list): e.g., ['ft', 'hm']
            - base_kinematic_cols (list): Base names of kinematic features.
            - TARGET_IMAGING_COL (str): Name of the target imaging column.
            - RIDGE_ALPHAS (tuple): Alphas for RidgeCV.
            - RIDGE_CV_FOLDS (int): CV folds for RidgeCV.
            - data_output_folder (str): Path to save data files.
            - plots_folder (str): Path to save plot files.
            - PLOT_TOP_N_RIDGE (int): How many coefficients to plot.
            - PLOT_BIVAR_TOP_N_LABEL (int): How many points to label on comparison plot.
        bivariate_results (dict, optional): A dictionary containing results
            from bivariate analysis needed for comparison plots. Expected keys:
            - 'raw_df': DataFrame of all raw bivariate correlations.
            - 'significant_df': DataFrame of FDR-significant bivariate correlations.
            Defaults to None.

    Returns:
        dict: A dictionary containing the ridge results for each task,
              keyed by task prefix (e.g., {'ft': {...}, 'hm': {...}}).
              Returns an empty dict if analysis fails broadly.
    """
    print("\n=== Starting Ridge Regression Analysis & Plotting (External Script) ===")
    all_ridge_results_dict = {}

    # --- Extract config variables ---
    tasks = config.get('tasks', [])
    base_kinematic_cols = config.get('base_kinematic_cols', [])
    imaging_col = config.get('TARGET_IMAGING_COL', None)
    alphas = config.get('RIDGE_ALPHAS', (0.1, 1.0, 10.0))
    cv_folds = config.get('RIDGE_CV_FOLDS', 5)
    data_output_folder = config.get('data_output_folder', './Output/Data')
    plots_folder = config.get('plots_folder', './Output/Plots')
    top_n_ridge_plot = config.get('PLOT_TOP_N_RIDGE', 20)
    top_n_bivar_plot_label = config.get('PLOT_BIVAR_TOP_N_LABEL', 7) # Added config

    if not tasks or not base_kinematic_cols or not imaging_col:
        print("Error: Missing essential configuration (tasks, base_kinematic_cols, TARGET_IMAGING_COL) for Ridge pipeline.")
        return {}

    # --- Run Ridge Analysis Loop ---
    for task in tasks:
        print(f"\n--- Running Ridge Regression for Task: {task.upper()} ---")
        ridge_results_task = run_ridge_analysis(
            df=df,
            base_kinematic_cols=base_kinematic_cols,
            task_prefix=task,
            imaging_col=imaging_col,
            alphas=alphas,
            cv_folds=cv_folds
        )

        if ridge_results_task:
            all_ridge_results_dict[task] = ridge_results_task # Store results

            # --- Save Ridge coefficients ---
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
                 else: print("  Could not save Ridge coefficients: No coefficients found in results.")
            except Exception as e: print(f"  Error saving Ridge coefficients for task {task}: {e}")
        else: print(f"Ridge Regression analysis failed or was skipped for task {task}.")

    print("=== Ridge Regression Analysis Finished ===")

    # --- Generate Ridge-Related Plots ---
    if PLOT_AVAILABLE:
        print("\n=== Generating Ridge-Related Plots ===")

        # --- Plot 1: Ridge Coefficient Plotting ---
        print("\n--- Generating Ridge Coefficient Plots ---")
        if not all_ridge_results_dict:
            print("Skipping Ridge plots: No Ridge results available.")
        else:
            for task, ridge_result_full in all_ridge_results_dict.items():
                 if ridge_result_full:
                     print(f"Generating Ridge plot for Task: {task.upper()}...")
                     plot_ridge_coefficients(
                         ridge_results_task=ridge_result_full,
                         top_n=top_n_ridge_plot,
                         output_folder=plots_folder,
                         file_name_base="ridge_coefficients" # Will add _{task}_OFF.png
                     )
                 else:
                     print(f"Skipping Ridge plot for task {task}: No results found.")

        # --- Plot 2: Bivariate vs Ridge Comparison Scatter Plot ---
        print("\n--- Generating Bivariate vs Ridge Comparison Scatter Plots ---")
        if bivariate_results is None:
             print("Skipping Bivar vs Ridge plot: No bivariate results provided.")
        elif not all_ridge_results_dict:
             print("Skipping Bivar vs Ridge plot: No Ridge results available for comparison.")
        else:
            # Extract bivariate data safely
            all_raw_bivariate_results_df = bivariate_results.get('raw_df')
            significant_bivariate_df = bivariate_results.get('significant_df')

            if all_raw_bivariate_results_df is None or all_raw_bivariate_results_df.empty:
                 print("Skipping Bivar vs Ridge plot: Raw bivariate results missing or empty.")
            elif significant_bivariate_df is None:
                 print("Warning: Significant bivariate results DataFrame is missing. Plots will not highlight significant points.")
                 significant_bivariate_df = pd.DataFrame() # Use empty DF to avoid errors
            else:
                # Proceed with plotting
                for task in tasks:
                    if task in all_ridge_results_dict and all_ridge_results_dict[task]:
                        plot_bivariate_vs_ridge_scatter(
                            bivariate_results_df=all_raw_bivariate_results_df,
                            ridge_results_task=all_ridge_results_dict[task],
                            significant_bivariate_df=significant_bivariate_df,
                            task_prefix=task,
                            top_n_label=top_n_bivar_plot_label, # Use config value
                            output_folder=plots_folder
                            # file_name_base default is okay
                        )
                    else:
                        print(f"Skipping Bivar vs Ridge plot for task {task}: Ridge results not found for this task.")

        print("--- Ridge Plotting Finished ---")
    else:
        print("\nPlot generation skipped as plotting functions were not available.")

    return all_ridge_results_dict

# Example of how it might be called (for testing standalone, if needed)
# if __name__ == '__main__':
#     print("This script is intended to be called by datnik_main.py")
    # Add minimal setup here to load data and call run_ridge_pipeline for testing
    # e.g., load sample df, define dummy config, call run_ridge_pipeline(df, config)

# --- END OF FILE datnik_run_ridge.py ---