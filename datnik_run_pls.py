# --- START OF FILE datnik_run_pls.py (MODIFIED) ---

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performs PLS Correlation analysis for a single task using pre-processed,
age-controlled (residualized) data.
"""

import os
import pandas as pd
import numpy as np

try:
    from datnik_analysis import run_pls_analysis
    from datnik_plotting import plot_pls_results
    PLOT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import in datnik_run_pls.py: {e}")
    def run_pls_analysis(*args, **kwargs): return None
    def plot_pls_results(*args, **kwargs): pass
    PLOT_AVAILABLE = False

def run_pls_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict
):
    """
    Executes the PLS Correlation analysis pipeline for a single task.

    Args:
        X (pd.DataFrame): DataFrame of predictor variables (kinematic residuals).
        y (pd.Series): Series of the target variable (imaging residual).
        config (dict): A dictionary containing configuration parameters.
    
    Returns:
        dict: A dictionary containing the raw PLS results for the task.
    """
    print("\n=== Running Single-Task PLS Pipeline ===")
    
    # --- Extract config variables ---
    task = config.get('task_prefix', 'unknown_task')
    max_components = config.get('PLS_MAX_COMPONENTS', 5)
    n_permutations = config.get('PLS_N_PERMUTATIONS', 1000)
    n_bootstraps = config.get('PLS_N_BOOTSTRAPS', 1000)
    alpha = config.get('PLS_ALPHA', 0.05)
    bsr_threshold = config.get('PLS_BSR_THRESHOLD', 1.8)
    data_output_folder = config.get('data_output_folder', './Output/Data')
    plots_folder = config.get('plots_folder', './Output/Plots')

    # --- Run PLS Analysis ---
    # The core analysis function is now called with X and y directly
    pls_results_task = run_pls_analysis(
        X=X,
        y=y,
        task_prefix=task, # Pass task for metadata
        max_components=max_components,
        n_permutations=n_permutations,
        n_bootstraps=n_bootstraps,
        alpha=alpha
    )

    if not pls_results_task:
        print(f"PLS Analysis failed or returned no results for task {task.upper()}.")
        return None

    # --- Process and Save Significant PLS Results for this task ---
    significant_lvs = pls_results_task.get('significant_lvs', [])
    print(f"  Task {task.upper()} - Significant LVs found: {significant_lvs if significant_lvs else 'None'}")

    if significant_lvs:
        lv_results_dict = pls_results_task.get('lv_results', {})
        for lv_index in significant_lvs:
            lv_data = lv_results_dict.get(lv_index)
            if not lv_data: continue

            # --- Generate PLS Plots for this significant LV ---
            if PLOT_AVAILABLE:
                print(f"    Generating PLS plots for LV{lv_index}...")
                plot_data_for_lv = lv_data.copy()
                plot_data_for_lv.update(pls_results_task) # Add top-level info
                
                plot_pls_results(
                     pls_results_lv=plot_data_for_lv,
                     lv_index=lv_index,
                     output_folder=plots_folder,
                     file_name_base=f"pls_results_{task}", # Task-specific filename
                     bsr_threshold=bsr_threshold
                )
    
    return pls_results_task