#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 15:27:07 2025

@author: Lange_L
"""

# --- START OF FILE datnik_run_elasticnet.py ---

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performs ElasticNet Regression analysis for specified tasks (e.g., FT, HM)
predicting an imaging variable from kinematic features using ElasticNetCV.
Includes saving coefficients and generating coefficient plots.
"""

import os
import pandas as pd
import numpy as np

# --- Import analysis and plotting functions ---
try:
    from datnik_analysis import run_elasticnet_analysis
    # Reuse ridge plotting function for coefficients, adapt if needed later
    from datnik_plotting import plot_ridge_coefficients as plot_elasticnet_coefficients
    PLOT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import analysis/plotting functions in datnik_run_elasticnet.py: {e}")
    print("Plot generation will be skipped.")
    def run_elasticnet_analysis(*args, **kwargs): return None
    def plot_elasticnet_coefficients(*args, **kwargs): pass
    PLOT_AVAILABLE = False

def run_elasticnet_pipeline(
    df: pd.DataFrame,
    config: dict
):
    """
    Executes the ElasticNet Regression analysis pipeline for specified tasks.

    Args:
        df (pd.DataFrame): The input DataFrame (typically filtered for OFF state).
        config (dict): A dictionary containing configuration parameters like:
            - tasks (list): e.g., ['ft', 'hm']
            - base_kinematic_cols (list): Base names of kinematic features.
            - TARGET_IMAGING_COL (str): Name of the target imaging column.
            - ENET_L1_RATIOS (list): l1_ratios for ElasticNetCV.
            - ENET_CV_FOLDS (int): CV folds for ElasticNetCV.
            - ENET_MAX_ITER (int): Max iterations.
            - ENET_RANDOM_STATE (int): Random state.
            - data_output_folder (str): Path to save data files.
            - plots_folder (str): Path to save plot files.
            - PLOT_TOP_N_ENET (int): How many non-zero coefficients to plot.

    Returns:
        dict: A dictionary containing the ElasticNet results for each task,
              keyed by task prefix (e.g., {'ft': {...}, 'hm': {...}}).
              Returns an empty dict if analysis fails broadly.
    """
    print("\n=== Starting ElasticNet Regression Analysis & Plotting (External Script) ===")
    all_enet_results_dict = {}

    # --- Extract config variables ---
    tasks = config.get('tasks', [])
    base_kinematic_cols = config.get('base_kinematic_cols', [])
    imaging_col = config.get('TARGET_IMAGING_COL', None)
    l1_ratios = config.get('ENET_L1_RATIOS', np.linspace(0.1, 1.0, 10))
    cv_folds = config.get('ENET_CV_FOLDS', 5)
    max_iter = config.get('ENET_MAX_ITER', 10000)
    random_state = config.get('ENET_RANDOM_STATE', 42)
    data_output_folder = config.get('data_output_folder', './Output/Data')
    plots_folder = config.get('plots_folder', './Output/Plots')
    top_n_enet_plot = config.get('PLOT_TOP_N_ENET', 20) # Plot top N non-zero features

    if not tasks or not base_kinematic_cols or not imaging_col:
        print("Error: Missing essential configuration (tasks, base_kinematic_cols, TARGET_IMAGING_COL) for ElasticNet pipeline.")
        return {}

    # --- Run ElasticNet Analysis Loop ---
    for task in tasks:
        print(f"\n--- Running ElasticNetCV for Task: {task.upper()} ---")
        # Use a consistent random state per task if desired, or vary it
        task_random_state = random_state # Or random_state + tasks.index(task)
        enet_results_task = run_elasticnet_analysis(
            df=df,
            base_kinematic_cols=base_kinematic_cols,
            task_prefix=task,
            imaging_col=imaging_col,
            l1_ratios=l1_ratios,
            cv_folds=cv_folds,
            max_iter=max_iter,
            random_state=task_random_state
        )

        if enet_results_task:
            all_enet_results_dict[task] = enet_results_task # Store results

            # --- Save ElasticNet coefficients (including zeros) ---
            enet_filename_base = os.path.join(data_output_folder, f"elasticnet_{task}_OFF")
            try:
                 coeffs_series = enet_results_task.get('coefficients')
                 if coeffs_series is not None and isinstance(coeffs_series, pd.Series):
                      coeffs_df = coeffs_series.reset_index()
                      coeffs_df.columns = ['Feature', 'Coefficient']
                      # Add model parameters to the output file
                      coeffs_df['Optimal_Alpha'] = enet_results_task.get('optimal_alpha')
                      coeffs_df['Optimal_L1_Ratio'] = enet_results_task.get('optimal_l1_ratio')
                      coeffs_df['R2_Full_Data'] = enet_results_task.get('r2_full_data')
                      # Sort by absolute coefficient value for inspection
                      coeffs_df.sort_values(by='Coefficient', key=abs, ascending=False, inplace=True)
                      coeffs_df.to_csv(f"{enet_filename_base}_coefficients.csv", sep=';', decimal='.', index=False)
                      print(f"  Saved ElasticNet coefficients to {enet_filename_base}_coefficients.csv")
                 else: print("  Could not save ElasticNet coefficients: No coefficients found in results.")
            except Exception as e: print(f"  Error saving ElasticNet coefficients for task {task}: {e}")
        else: print(f"ElasticNetCV analysis failed or was skipped for task {task}.")

    print("=== ElasticNet Regression Analysis Finished ===")

    # --- Generate ElasticNet Coefficient Plots ---
    if PLOT_AVAILABLE:
        print("\n=== Generating ElasticNet Coefficient Plots ===")
        if not all_enet_results_dict:
            print("Skipping ElasticNet plots: No ElasticNet results available.")
        else:
            for task, enet_result_full in all_enet_results_dict.items():
                 if enet_result_full:
                     print(f"Generating ElasticNet plot for Task: {task.upper()}...")
                     # Prepare data for plotting (filter non-zero, take top N)
                     coeffs_series = enet_result_full.get('coefficients')
                     if coeffs_series is not None and not coeffs_series.empty:
                         non_zero_coeffs = coeffs_series[coeffs_series != 0]
                         if not non_zero_coeffs.empty:
                             # Create a temporary dict mimicking ridge_results for the plot function
                             plot_data_dict = {
                                 'task': enet_result_full.get('task'),
                                 'coefficients': non_zero_coeffs, # Pass only non-zero
                                 'optimal_alpha': enet_result_full.get('optimal_alpha'), # For title
                                 'optimal_l1_ratio': enet_result_full.get('optimal_l1_ratio'), # Add l1_ratio to title maybe?
                                 'r2_full_data': enet_result_full.get('r2_full_data'),
                                 'n_samples_ridge': enet_result_full.get('n_samples_enet'), # Use enet N
                                 'imaging_variable': enet_result_full.get('imaging_variable')
                             }
                             # Adapt the plot function call or modify the plot function itself later if needed
                             plot_elasticnet_coefficients(
                                 ridge_results_task=plot_data_dict, # Pass the prepared dict
                                 top_n=top_n_enet_plot, # Plot top N of the non-zero coeffs
                                 output_folder=plots_folder,
                                 file_name_base="elasticnet_coefficients" # New base name
                             )
                         else:
                             print(f"  Skipping ElasticNet plot for task {task}: No non-zero coefficients found.")
                     else:
                          print(f"  Skipping ElasticNet plot for task {task}: Coefficients Series missing or empty.")
                 else:
                     print(f"Skipping ElasticNet plot for task {task}: No results found.")
        print("--- ElasticNet Plotting Finished ---")
    else:
        print("\nPlot generation skipped as plotting functions were not available.")

    return all_enet_results_dict

# --- END OF FILE datnik_run_elasticnet.py ---