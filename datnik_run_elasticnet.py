# --- START OF FILE datnik_run_elasticnet.py (MODIFIED) ---

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performs ElasticNet Regression for a single task using pre-processed,
age-controlled (residualized) data.
"""

import os
import pandas as pd
import numpy as np

try:
    from datnik_analysis import run_elasticnet_analysis
    from datnik_plotting import plot_ridge_coefficients as plot_elasticnet_coefficients
    PLOT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import in datnik_run_elasticnet.py: {e}")
    def run_elasticnet_analysis(*args, **kwargs): return None
    def plot_elasticnet_coefficients(*args, **kwargs): pass
    PLOT_AVAILABLE = False

def run_elasticnet_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict
):
    """
    Executes the ElasticNet Regression analysis pipeline for a single task.

    Args:
        X (pd.DataFrame): DataFrame of predictor variables (kinematic residuals).
        y (pd.Series): Series of the target variable (imaging residual).
        config (dict): A dictionary containing configuration parameters.

    Returns:
        dict: A dictionary containing the ElasticNet results for the task.
    """
    print("\n=== Running Single-Task ElasticNet Pipeline ===")

    # --- Extract config variables ---
    task = config.get('task_prefix', 'unknown_task')
    l1_ratios = config.get('ENET_L1_RATIOS', np.linspace(0.1, 1.0, 10))
    cv_folds = config.get('ENET_CV_FOLDS', 5)
    max_iter = config.get('ENET_MAX_ITER', 10000)
    random_state = config.get('ENET_RANDOM_STATE', 42)
    data_output_folder = config.get('data_output_folder', './Output/Data')
    plots_folder = config.get('plots_folder', './Output/Plots')
    top_n_enet_plot = config.get('PLOT_TOP_N_ENET', 20)

    # --- Run ElasticNet Analysis ---
    enet_results_task = run_elasticnet_analysis(
        X=X,
        y=y,
        task_prefix=task,
        l1_ratios=l1_ratios,
        cv_folds=cv_folds,
        max_iter=max_iter,
        random_state=random_state
    )

    if not enet_results_task:
        print(f"ElasticNetCV analysis failed or was skipped for task {task}.")
        return None

    # --- Save ElasticNet coefficients ---
    enet_filename_base = os.path.join(data_output_folder, f"elasticnet_{task}_OFF_agecontrolled")
    try:
         coeffs_series = enet_results_task.get('coefficients')
         if coeffs_series is not None:
              coeffs_df = coeffs_series.reset_index(); coeffs_df.columns = ['Feature', 'Coefficient']
              perf = enet_results_task.get('performance', {})
              coeffs_df['Optimal_Alpha'] = perf.get('alpha'); coeffs_df['Optimal_L1_Ratio'] = perf.get('l1_ratio')
              coeffs_df['CV_R2'] = perf.get('R2')
              coeffs_df.sort_values(by='Coefficient', key=abs, ascending=False, inplace=True)
              coeffs_df.to_csv(f"{enet_filename_base}_coefficients.csv", sep=';', decimal='.', index=False)
              print(f"  Saved ElasticNet coefficients to {enet_filename_base}_coefficients.csv")
    except Exception as e: print(f"  Error saving ElasticNet coefficients for task {task}: {e}")

    # --- Generate ElasticNet Coefficient Plots ---
    if PLOT_AVAILABLE:
        print(f"Generating ElasticNet plot for Task: {task.upper()}...")
        coeffs_series = enet_results_task.get('coefficients')
        if coeffs_series is not None and not coeffs_series.empty:
            non_zero_coeffs = coeffs_series[coeffs_series != 0]
            if not non_zero_coeffs.empty:
                plot_data_dict = enet_results_task.copy()
                plot_data_dict['coefficients'] = non_zero_coeffs
                plot_elasticnet_coefficients(
                    ridge_results_task=plot_data_dict,
                    top_n=top_n_enet_plot,
                    output_folder=plots_folder,
                    file_name_base=f"elasticnet_coeffs_{task}_agecontrolled"
                )
            else:
                print(f"  Skipping plot for task {task}: No non-zero coefficients found.")

    return enet_results_task