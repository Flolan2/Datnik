#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 15:04:20 2025

@author: Lange_L
"""

# --- START OF FILE datnik_run_pls.py ---

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performs PLS Correlation analysis for specified tasks (e.g., FT, HM)
predicting an imaging variable from kinematic features.
Includes saving significant results (Loadings, BSRs) and generating plots.
"""

import os
import pandas as pd
import numpy as np

# --- Import analysis and plotting functions ---
try:
    from datnik_analysis import run_pls_analysis
    from datnik_plotting import plot_pls_results
    PLOT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import analysis/plotting functions in datnik_run_pls.py: {e}")
    print("Plot generation will be skipped.")
    def run_pls_analysis(*args, **kwargs): return None
    def plot_pls_results(*args, **kwargs): pass
    PLOT_AVAILABLE = False

def run_pls_pipeline(
    df: pd.DataFrame,
    config: dict
):
    """
    Executes the PLS Correlation analysis pipeline for specified tasks.

    Args:
        df (pd.DataFrame): The input DataFrame (typically filtered for OFF state).
        config (dict): A dictionary containing configuration parameters like:
            - tasks (list): e.g., ['ft', 'hm']
            - base_kinematic_cols (list): Base names of kinematic features.
            - TARGET_IMAGING_COL (str): Name of the target imaging column.
            - PLS_MAX_COMPONENTS (int): Max LVs for PLS.
            - PLS_N_PERMUTATIONS (int): Permutations for significance.
            - PLS_N_BOOTSTRAPS (int): Bootstraps for BSRs.
            - PLS_ALPHA (float): Significance level for LV permutations.
            - PLS_BSR_THRESHOLD (float): Threshold for considering BSR significant.
            - data_output_folder (str): Path to save data files.
            - plots_folder (str): Path to save plot files.

    Returns:
        dict: A dictionary containing the raw PLS results for each task,
              keyed by task prefix (e.g., {'ft': {...}, 'hm': {...}}).
              Returns an empty dict if analysis fails broadly.
    """
    print("\n=== Starting PLS Correlation Analysis & Plotting (External Script) ===")
    all_pls_results_dict = {}
    significant_results_to_save = [] # Store results for combined CSV

    # --- Extract config variables ---
    tasks = config.get('tasks', [])
    base_kinematic_cols = config.get('base_kinematic_cols', [])
    imaging_col = config.get('TARGET_IMAGING_COL', None)
    max_components = config.get('PLS_MAX_COMPONENTS', 5)
    n_permutations = config.get('PLS_N_PERMUTATIONS', 1000)
    n_bootstraps = config.get('PLS_N_BOOTSTRAPS', 1000)
    alpha = config.get('PLS_ALPHA', 0.05)
    bsr_threshold = config.get('PLS_BSR_THRESHOLD', 1.5) # Default threshold
    data_output_folder = config.get('data_output_folder', './Output/Data')
    plots_folder = config.get('plots_folder', './Output/Plots')

    if not tasks or not base_kinematic_cols or not imaging_col:
        print("Error: Missing essential configuration (tasks, base_kinematic_cols, TARGET_IMAGING_COL) for PLS pipeline.")
        return {}

    # --- Run PLS Analysis Loop ---
    for task in tasks:
        print(f"\n--- Running PLS Analysis for Task: {task.upper()} ---")
        pls_results_task = run_pls_analysis(
            df=df,
            base_kinematic_cols=base_kinematic_cols,
            task_prefix=task,
            imaging_col=imaging_col,
            max_components=max_components,
            n_permutations=n_permutations,
            n_bootstraps=n_bootstraps,
            alpha=alpha
        )

        if pls_results_task:
            all_pls_results_dict[task] = pls_results_task # Store raw results

            # --- Process and Save Significant PLS Results ---
            significant_lvs = pls_results_task.get('significant_lvs', [])
            print(f"  Task {task.upper()} - Significant LVs found: {significant_lvs if significant_lvs else 'None'}")

            if significant_lvs:
                lv_results_dict = pls_results_task.get('lv_results', {})
                kinematic_vars = pls_results_task.get('kinematic_variables', [])

                for lv_index in significant_lvs: # lv_index is 1-based here
                    lv_data = lv_results_dict.get(lv_index)
                    if not lv_data: continue

                    loadings = lv_data.get('x_loadings') # Should be a pandas Series
                    bsrs = lv_data.get('bootstrap_ratios') # Should be a pandas Series

                    if loadings is not None and bsrs is not None:
                        # Combine loadings and BSRs, ensuring index alignment
                        combined_lv_df = pd.DataFrame({
                            'Kinematic_Variable': loadings.index,
                            'Loading': loadings.values,
                            'Bootstrap_Ratio': bsrs.reindex(loadings.index).values # Reindex BSRs to match loadings index
                        })
                        combined_lv_df['LV'] = lv_index
                        combined_lv_df['Task'] = task
                        combined_lv_df['Correlation_LV'] = lv_data.get('correlation', np.nan)
                        combined_lv_df['P_value_LV'] = lv_data.get('p_value', np.nan)

                        # Filter for variables meeting the BSR threshold
                        significant_vars_lv = combined_lv_df[
                            abs(combined_lv_df['Bootstrap_Ratio']) >= bsr_threshold
                        ].copy()

                        if not significant_vars_lv.empty:
                            significant_results_to_save.append(significant_vars_lv)
                            print(f"    LV{lv_index}: Found {len(significant_vars_lv)} variables with |BSR| >= {bsr_threshold}")
                        else:
                             print(f"    LV{lv_index}: No variables met |BSR| >= {bsr_threshold}")

                    # --- Generate PLS Plots for this significant LV ---
                    if PLOT_AVAILABLE:
                        print(f"    Generating PLS plots for LV{lv_index}...")
                        # Create a dictionary for plotting that includes necessary top-level info
                        plot_data_for_lv = lv_data.copy() # lv_data is pls_results_task['lv_results'][lv_index]
                        plot_data_for_lv['task'] = pls_results_task.get('task') # Pass the task prefix
                        plot_data_for_lv['kinematic_variables'] = pls_results_task.get('kinematic_variables') # Pass the list of features
                        plot_data_for_lv['n_samples_pls'] = pls_results_task.get('n_samples_pls') # Pass N samples
                    
                        plot_pls_results(
                             pls_results_lv=plot_data_for_lv, # Pass the augmented dictionary
                             lv_index=lv_index,
                             output_folder=plots_folder,
                             file_name_base="pls_results",
                             bsr_threshold=bsr_threshold
                        )

        else: print(f"PLS Analysis failed or was skipped for task {task}.")

    print("=== PLS Correlation Analysis Finished ===")

    # --- Save Combined Significant PLS Results (Loadings & BSRs) ---
    if significant_results_to_save:
         final_significant_df = pd.concat(significant_results_to_save, ignore_index=True)

         # Reorder columns for clarity first
         cols_order = ['Task', 'LV', 'Kinematic_Variable', 'Loading', 'Bootstrap_Ratio', 'Correlation_LV', 'P_value_LV']
         # Check if all expected columns are present before reordering
         present_cols = [col for col in cols_order if col in final_significant_df.columns]
         final_significant_df = final_significant_df[present_cols] # Reorder using only present cols

         # --- Corrected Sorting Logic ---
         # Check if essential sorting columns exist
         sort_by_cols = ['Task', 'LV', 'Bootstrap_Ratio']
         if all(col in final_significant_df.columns for col in sort_by_cols):
             print("Sorting combined significant PLS results...")
             # 1. Sort by Task and LV first (ascending is default)
             final_significant_df.sort_values(by=['Task', 'LV'], inplace=True, ascending=[True, True], na_position='last')

             # 2. Then sort by the absolute value of BSR descendingly within those groups
             #    We use reindex with a calculated sorting key for this level.
             #    Handle potential NaNs in BSR before taking abs()
             abs_bsr_sorted_index = final_significant_df['Bootstrap_Ratio'].fillna(0).abs().sort_values(ascending=False).index
             final_significant_df = final_significant_df.reindex(abs_bsr_sorted_index)
             print("Sorting complete.")
         else:
             print("Warning: Cannot perform full sort because one or more key columns ('Task', 'LV', 'Bootstrap_Ratio') are missing.")
         # --- End Corrected Sorting Logic ---

         output_filename = os.path.join(data_output_folder, "pls_significant_results_all_tasks_combined_sorted.csv")
         try:
              final_significant_df.to_csv(output_filename, index=False, sep=';', decimal='.')
              print(f"\nCombined significant PLS results (|BSR| >= {bsr_threshold}) saved to: {output_filename}")
         except Exception as e:
              print(f"\nError saving combined significant PLS results: {e}")
    else:
         print("\nNo variables met the BSR threshold in any significant LV across tasks. No combined results file saved.")


    if not PLOT_AVAILABLE:
        print("\nPlot generation skipped as plotting functions were not available.")

    return all_pls_results_dict

# Example of how it might be called (for testing standalone, if needed)
# if __name__ == '__main__':
#     print("This script is intended to be called by a main orchestrator.")
    # Add minimal setup here to load data and call run_pls_pipeline for testing

# --- END OF FILE datnik_run_pls.py ---