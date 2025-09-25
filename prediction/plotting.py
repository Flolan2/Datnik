# -*- coding: utf-8 -*-
"""
Plotting functions for visualizing experiment results,
handling different splitting modes.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# --- Plotting Functions (Adapted) ---

def plot_metric_distributions(aggregated_metrics_df, metric_key, base_title, base_filename, config):
    """
    Generates box plots comparing a metric across configurations and tasks,
    creating separate plots for each splitting mode found in the DataFrame.

    Args:
        aggregated_metrics_df (pd.DataFrame): DataFrame from aggregate_metrics (must include 'Mode' column).
        metric_key (str): The metric column name suffix (e.g., 'roc_auc').
        base_title (str): Base plot title (mode name will be added).
        base_filename (str): Base path to save plots (mode name and '.png' will be added).
        config (module): The main configuration module (e.g., config or config_multiclass).
    """
    mean_col = f'{metric_key}_mean'
    mode_col = 'Mode' # Expects this column name

    if mean_col not in aggregated_metrics_df.columns or mode_col not in aggregated_metrics_df.columns:
        print(f"Warning: Required columns ('{mean_col}', '{mode_col}') not found. Skipping plot '{base_filename}'.")
        return

    # Get unique modes present in the data
    modes = aggregated_metrics_df[mode_col].unique()

    for mode in modes:
        plot_df_mode = aggregated_metrics_df[aggregated_metrics_df[mode_col] == mode].copy()
        plot_df_mode = plot_df_mode.dropna(subset=[mean_col])

        if plot_df_mode.empty:
            print(f"Warning: No valid data found for metric '{metric_key}' in mode '{mode}'. Skipping plot.")
            continue

        # Create a combined label for plotting within this mode
        plot_df_mode['Config_Task'] = plot_df_mode['Config_Name'] + " (" + plot_df_mode['Task_Name'].str.upper() + ")"

        # Sort by median performance for better visualization within this mode
        order = plot_df_mode.groupby('Config_Task')[mean_col].median().sort_values(ascending=False).index

        plt.figure(figsize=(max(8, len(order) * 0.7), 6)) # Adjust size
        sns.boxplot(x=mean_col, y='Config_Task', data=plot_df_mode, order=order, palette='viridis', width=0.6)

        # Add title indicating the mode
        title = f"{base_title}\n(Split Mode: {mode.upper()})"
        plt.title(title, fontsize=14, weight='bold')
        plt.xlabel(metric_key.replace('_', ' ').title() + " (Mean over Reps)", fontsize=12)
        plt.ylabel("Configuration (Task)", fontsize=12)
        plt.yticks(fontsize=9)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        sns.despine()
        plt.tight_layout()

        # Add mode to filename
        filename = base_filename.replace(".png", f"_{mode}.png")
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True) # Ensure directory exists
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Metric distribution plot for mode '{mode}' saved to: {filename}")
        except Exception as e: print(f"Error saving metric distribution plot {filename}: {e}")
        finally: plt.close()

def plot_aggregated_roc_curves(all_runs_roc_data, all_runs_metrics, configs_tasks_to_plot, title, filename, config):
    """
    Plots average ROC curves with variability for selected configurations and tasks FOR A SINGLE MODE.
    (Assumes input dictionaries 'all_runs_roc_data' and 'all_runs_metrics' are pre-filtered for one mode).

    Args:
        all_runs_roc_data (dict): config -> task -> list of (fpr, tpr) tuples for ONE mode.
        all_runs_metrics (dict): config -> task -> list of metric dicts for ONE mode.
        configs_tasks_to_plot (list): List of tuples (config_name, task_name) to include.
        title (str): Plot title (should include mode info).
        filename (str): Full path to save the plot.
        config (module): The main configuration module (e.g., config or config_multiclass).
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 8))
    base_fpr = np.linspace(0, 1, 101)
    plot_successful = False
    if not configs_tasks_to_plot: # Handle empty list
         print(f"Warning: No configurations/tasks specified for ROC plot '{filename}'. Skipping.")
         plt.close(); return

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(configs_tasks_to_plot)))

    for i, (config_name, task_name) in enumerate(configs_tasks_to_plot):
        roc_runs = all_runs_roc_data.get(config_name, {}).get(task_name, [])
        metric_runs = all_runs_metrics.get(config_name, {}).get(task_name, [])
        auc_list = [m.get('roc_auc') for m in metric_runs if m and pd.notna(m.get('roc_auc'))]
        label_base = f"{config_name} ({task_name.upper()})"

        if not roc_runs or not auc_list: continue # Skip if no data

        tprs_interp = []; valid_run_indices = []
        for idx, run_data in enumerate(roc_runs):
             if isinstance(run_data, (list, tuple)) and len(run_data) == 2:
                 fpr, tpr = run_data
                 if hasattr(fpr, '__len__') and hasattr(tpr, '__len__') and len(fpr) >= 2 and len(tpr) >= 2:
                     fpr, tpr = np.array(fpr), np.array(tpr)
                     if np.all(np.diff(fpr) >= 0):
                         tpr_interp = np.interp(base_fpr, fpr, tpr); tpr_interp[0] = 0.0
                         tprs_interp.append(tpr_interp); valid_run_indices.append(idx)

        if not tprs_interp: continue # Skip if no valid curves

        auc_list_filtered = [auc for idx, auc in enumerate(auc_list) if idx in valid_run_indices]
        if not auc_list_filtered: continue # Skip if no matching AUCs

        mean_tprs = np.mean(tprs_interp, axis=0); std_tprs = np.std(tprs_interp, axis=0)
        tprs_upper = np.minimum(mean_tprs + std_tprs, 1); tprs_lower = np.maximum(mean_tprs - std_tprs, 0)
        mean_auc = np.mean(auc_list_filtered); std_auc = np.std(auc_list_filtered)

        label = f'{label_base} (AUC = {mean_auc:.3f} ± {std_auc:.3f})'
        plt.plot(base_fpr, mean_tprs, label=label, color=colors[i % len(colors)], lw=2)
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, color=colors[i % len(colors)], alpha=0.15)
        plot_successful = True

    if not plot_successful:
         print(f"Warning: No ROC curves successfully plotted for '{filename}'.")
         plt.close(); return

    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
    plt.xlim([-0.01, 1.01]); plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title(title, fontsize=14, weight='bold') # Title should mention the mode
    legend_fontsize = 10 if len(configs_tasks_to_plot) < 8 else 8
    plt.legend(loc='lower right', fontsize=legend_fontsize, frameon=True, facecolor='white', framealpha=0.8)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Aggregated ROC curve plot saved to: {filename}")
    except Exception as e: print(f"Error saving ROC curve plot {filename}: {e}")
    finally: plt.close()

def plot_aggregated_importances(importance_df, config_name, task_name, top_n, title, filename, config):
    """
    Plots aggregated feature importances/coefficients with error bars FOR A SINGLE MODE/CONFIG/TASK.
    (Title and filename should include mode info, passed from the calling script).

    Args:
        importance_df (pd.DataFrame): Aggregated importances for one mode/config/task.
        config_name (str): Name of the configuration.
        task_name (str): Name of the task ('ft', 'hm', 'meta_coeffs').
        top_n (int): Number of top features to plot.
        title (str): Plot title (should include mode info).
        filename (str): Full path to save the plot.
        config (module): The main configuration module (e.g., config or config_multiclass).
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    if importance_df is None or importance_df.empty or top_n <= 0: return # Skip silently if no data
    if not {'Mean_Importance', 'Std_Importance'}.issubset(importance_df.columns): return # Skip silently

    imp_df_sorted = importance_df.reindex(importance_df['Mean_Importance'].abs().sort_values(ascending=False, na_position='last').index)
    plot_df = imp_df_sorted.head(top_n).copy()
    plot_df.dropna(subset=['Mean_Importance'], inplace=True)
    if plot_df.empty: return # Skip silently

    plot_df = plot_df.iloc[::-1]
    is_coefficient_like = (plot_df['Mean_Importance'] < 0).any()
    colors = ['#d62728' if c < 0 else '#1f77b4' for c in plot_df['Mean_Importance']] if is_coefficient_like else '#1f77b4'
    center_line = 0 if is_coefficient_like else None
    xlabel = 'Mean Coefficient (Log-Odds Change)' if is_coefficient_like else 'Mean Feature Importance'
    ylabel = 'Feature'
    if 'meta' in task_name.lower():
         ylabel = 'Base Model Prediction'
         if is_coefficient_like: xlabel = 'Mean Meta-Model Coefficient'
         else: xlabel = 'Mean Meta-Model Importance'

    plt.figure(figsize=(10, max(4, len(plot_df) * 0.35)))
    plt.barh( plot_df.index, plot_df['Mean_Importance'], xerr=plot_df['Std_Importance'].fillna(0),
              color=colors, alpha=0.85, edgecolor='black', linewidth=0.7, capsize=4 )
    if center_line is not None: plt.axvline(center_line, color='dimgrey', linestyle='--', linewidth=1)
    plt.xlabel(xlabel, fontsize=12); plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, weight='bold') # Title should include mode
    plt.grid(axis='x', linestyle=':', alpha=0.7); plt.yticks(fontsize=9)
    sns.despine(left=True, bottom=False); plt.tight_layout()
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Aggregated importances plot saved to: {filename}")
    except Exception as e: print(f"Error saving importances plot {filename}: {e}")
    finally: plt.close()

# <<< NEW FUNCTION ADDED FOR THRESHOLD SWEEP PLOTTING >>>
def plot_performance_vs_threshold(summary_df, x_col, y_col, filename, config):
    """
    Generates a line plot showing a performance metric vs. a changing threshold.
    Includes a shaded region for standard deviation.

    Args:
        summary_df (pd.DataFrame): DataFrame with experiment results.
        x_col (str): Column name for the x-axis (e.g., 'Threshold_Value').
        y_col (str): Column name for the y-axis (e.g., 'Mean_ROC_AUC').
        filename (str): Full path to save the plot.
        config (module): The main configuration module.
    """
    std_col = y_col.replace('Mean_', 'Std_')
    if not {x_col, y_col, std_col, 'Config_Name'}.issubset(summary_df.columns):
        print(f"Warning: Required columns for threshold plot not found. Skipping plot '{filename}'.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))

    configs = summary_df['Config_Name'].unique()
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(configs)))

    for i, cfg_name in enumerate(configs):
        df_cfg = summary_df[summary_df['Config_Name'] == cfg_name].copy()
        df_cfg = df_cfg.sort_values(by=x_col).dropna(subset=[x_col, y_col, std_col])

        if df_cfg.empty:
            continue

        x = df_cfg[x_col]
        y_mean = df_cfg[y_col]
        y_std = df_cfg[std_col]

        # Plot the mean performance line
        plt.plot(x, y_mean, marker='o', linestyle='-', label=cfg_name, color=colors[i])

        # Add the shaded standard deviation area
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, color=colors[i], alpha=0.2)

    plt.axhline(y=0.5, color='black', linestyle='--', label='Chance (AUC = 0.50)')

    # Formatting the plot
    plt.title('Model Performance vs. DaTscan Abnormality Threshold', fontsize=16, weight='bold')
    plt.xlabel('Z-Score Threshold', fontsize=12)
    plt.ylabel('Mean ROC AUC', fontsize=12)
    plt.legend(title='Model Configuration')
    plt.grid(True, which='both', linestyle=':', linewidth='0.5')
    
    # Invert x-axis because more negative Z-scores are "more abnormal"
    plt.gca().invert_xaxis()
    
    sns.despine()
    plt.tight_layout()

    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Performance vs. Threshold plot saved to: {filename}")
    except Exception as e:
        print(f"Error saving performance vs. threshold plot {filename}: {e}")
    finally:
        plt.close()