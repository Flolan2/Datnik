# -*- coding: utf-8 -*-
"""
Functions for aggregating and summarizing results across multiple runs and configurations,
now handling different splitting modes.
"""

import pandas as pd
import numpy as np
import os
import collections

# <<< MODIFIED: Removed internal config import >>>
# try: from . import config
# except ImportError: import config; print("Warning: Used direct import for 'config' in results_processor.py")

# <<< MODIFIED: Function signature accepts config object >>>
def aggregate_metrics(all_runs_metrics, config):
    """
    Aggregates metrics across repetitions for each mode, configuration, and task.

    Args:
        all_runs_metrics (dict): Nested dict: mode -> config_name -> task_name -> list of metric dicts.
        config (module): The main configuration module (e.g., config or config_multiclass).

    Returns:
        pd.DataFrame: A DataFrame summarizing the aggregated metrics, including a 'Mode' column.
                      Returns None if input is empty or no valid results found.
    """
    print("\n===== Aggregating Performance Metrics (Across Modes) =====")
    aggregated_summary = []
    metric_keys_display = ['roc_auc', 'accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

    if not all_runs_metrics:
        print("Input 'all_runs_metrics' dictionary is empty.")
        return None

    # <<< ADDED: Outer loop for mode >>>
    for mode, mode_results in all_runs_metrics.items():
        print(f"\n--- Processing Mode: {mode.upper()} ---")
        if not mode_results:
            print("No configurations found for this mode.")
            continue

        for config_name, config_results in mode_results.items():
            if not config_results:
                print(f"  Config: {config_name} - No tasks found.")
                continue

            for task_name, task_runs_metrics in config_results.items():
                valid_runs = [run for run in task_runs_metrics if isinstance(run, dict) and run]
                valid_metrics_list = [
                    {k: run.get(k) for k in metric_keys_display}
                    for run in valid_runs if any(pd.notna(run.get(k)) for k in metric_keys_display)
                ]
                n_total_runs = len(task_runs_metrics)
                n_valid_runs = len(valid_metrics_list)

                print(f"\n  Config: {config_name} / Task: {task_name.upper()} (Mode: {mode})")
                print(f"  Total runs attempted: {n_total_runs}, Valid runs with metrics: {n_valid_runs}")

                # <<< MODIFIED: Create base row with Mode >>>
                row = {
                    'Mode': mode, # Add mode identifier
                    'Config_Name': config_name,
                    'Task_Name': task_name,
                    'N_Valid_Runs': int(n_valid_runs)
                }

                if n_valid_runs == 0:
                    print("  No valid runs found for aggregation.")
                    for key in metric_keys_display:
                        row[f'{key}_mean'] = np.nan; row[f'{key}_std'] = np.nan
                else:
                    metrics_df = pd.DataFrame(valid_metrics_list)
                    for mkey in metric_keys_display:
                        if mkey not in metrics_df.columns: metrics_df[mkey] = np.nan
                    means = metrics_df.mean(skipna=True); stds = metrics_df.std(skipna=True)
                    print("  Mean +/- Std Dev:")
                    for key in metric_keys_display:
                        mean_val = means.get(key, np.nan); std_val = stds.get(key, np.nan)
                        row[f'{key}_mean'] = mean_val; row[f'{key}_std'] = std_val
                        if pd.notna(mean_val): print(f"    {key}: {mean_val:.4f} ± {std_val:.4f}" if pd.notna(std_val) else f"    {key}: {mean_val:.4f}")
                        else: print(f"    {key}: N/A")

                aggregated_summary.append(row)

    if not aggregated_summary:
        print("No valid results found across all modes to create summary DataFrame.")
        return None

    summary_df = pd.DataFrame(aggregated_summary)
    # <<< MODIFIED: Add 'Mode' to column order >>>
    cols_order = ['Mode', 'Config_Name', 'Task_Name', 'N_Valid_Runs'] + \
                 sorted([col for col in summary_df.columns if col not in ['Mode','Config_Name','Task_Name','N_Valid_Runs']])
    summary_df = summary_df[cols_order]

    # <<< MODIFIED: Use the passed config object >>>
    if config.SAVE_AGGREGATED_SUMMARY:
        os.makedirs(config.DATA_OUTPUT_FOLDER, exist_ok=True)
        # <<< MODIFIED: Update filename slightly >>>
        filename = os.path.join(config.DATA_OUTPUT_FOLDER, "experiment_metrics_summary_modes.csv")
        try:
            summary_df.to_csv(filename, index=False, sep=';', decimal='.', float_format='%.6f')
            print(f"\nAggregated metrics summary saved to: {filename}")
        except Exception as e: print(f"\nError saving aggregated metrics summary: {e}")

    return summary_df

# <<< MODIFIED: Function signature accepts config object >>>
def aggregate_importances(all_runs_importances, config, file_prefix="importance"):
    """
    Aggregates feature importances/coefficients across repetitions for each mode, config, task.
    Saves separate files per mode/config/task.

    Args:
        all_runs_importances (dict): Nested dict: mode -> config_name -> task_name -> list of pd.Series.
        config (module): The main configuration module (e.g., config or config_multiclass).
        file_prefix (str): Prefix for the output CSV filenames.

    Returns:
        dict: Nested dict: mode -> config_name -> task_name -> aggregated importance DataFrame.
              Returns defaultdict structure.
    """
    print(f"\n===== Aggregating Feature Importances ({file_prefix}, Across Modes) =====")
    aggregated_dfs = collections.defaultdict(lambda: collections.defaultdict(dict))

    if not all_runs_importances:
        print("Input 'all_runs_importances' dictionary is empty.")
        return aggregated_dfs

    # <<< ADDED: Outer loop for mode >>>
    for mode, mode_results in all_runs_importances.items():
        print(f"\n--- Processing Mode: {mode.upper()} ---")
        if not mode_results:
            print("No configurations found for this mode.")
            continue

        for config_name, config_imps in mode_results.items():
            if not config_imps:
                 print(f"  Config: {config_name} - No tasks found.")
                 continue

            for task_name, task_runs_imps in config_imps.items():
                valid_imps = [s for s in task_runs_imps if isinstance(s, pd.Series) and not s.empty]
                n_valid = len(valid_imps)

                print(f"\n  Config: {config_name} / Task: {task_name.upper()} (Mode: {mode})")
                print(f"  Found {n_valid} valid importance/coefficient sets.")

                if n_valid > 0:
                    try:
                        imp_df = pd.concat(valid_imps, axis=1, join='outer', keys=range(n_valid))
                        agg_imp = pd.DataFrame({
                            'Mean_Importance': imp_df.mean(axis=1, skipna=True),
                            'Std_Importance': imp_df.std(axis=1, skipna=True),
                            'N_Valid_Runs': imp_df.notna().sum(axis=1).astype(int)
                        }, index=imp_df.index)
                        agg_imp = agg_imp.reindex(agg_imp['Mean_Importance'].abs().sort_values(ascending=False, na_position='last').index)

                        # <<< MODIFIED: Store under mode >>>
                        aggregated_dfs[mode][config_name][task_name] = agg_imp

                        # <<< MODIFIED: Use the passed config object >>>
                        if config.SAVE_AGGREGATED_IMPORTANCES:
                            os.makedirs(config.DATA_OUTPUT_FOLDER, exist_ok=True)
                            # <<< MODIFIED: Include mode in filename >>>
                            filename = os.path.join(config.DATA_OUTPUT_FOLDER, f"{file_prefix}_{mode}_{config_name}_{task_name}_agg.csv")
                            agg_imp.to_csv(filename, sep=';', decimal='.', index_label='Feature', float_format='%.6f')
                            print(f"    -> Saved aggregated importances to: {filename}")

                    except Exception as e:
                        print(f"    Error aggregating/saving importances for {mode}/{config_name}/{task_name}: {e}")
                        aggregated_dfs[mode][config_name][task_name] = pd.DataFrame()
                else:
                    print("    No valid importance data to aggregate for this task.")
                    aggregated_dfs[mode][config_name][task_name] = pd.DataFrame()

    return aggregated_dfs

def aggregate_rfe_features(all_runs_rfe_selected_features, config):
    """
    Aggregates the lists of RFE selected features across repetitions for each
    mode, configuration, and task. Calculates the frequency of each feature's selection.

    Args:
        all_runs_rfe_selected_features (dict): Nested dict:
            mode -> config_name -> task_name -> list of lists of selected feature names.
        config (module): The main configuration module.

    Returns:
        dict: Nested dict: mode -> config_name -> task_name -> DataFrame
              Each DataFrame has 'Feature' and 'Selection_Frequency' (0-1) and 'Selection_Count'.
              Returns defaultdict structure.
    """
    print(f"\n===== Aggregating RFE Selected Features (Across Modes) =====")
    aggregated_rfe_dfs = collections.defaultdict(lambda: collections.defaultdict(dict))

    if not all_runs_rfe_selected_features:
        print("Input 'all_runs_rfe_selected_features' dictionary is empty.")
        return aggregated_rfe_dfs

    for mode, mode_results in all_runs_rfe_selected_features.items():
        print(f"\n--- Processing Mode: {mode.upper()} for RFE features ---")
        if not mode_results:
            print("  No configurations found for this mode.")
            continue

        for config_name, config_data in mode_results.items():
            # Only process configs that likely used RFE (can be made more robust by checking exp_config)
            if 'RFE' not in config_name.upper(): # Simple heuristic
                # print(f"  Config: {config_name} - Skipping RFE aggregation (name doesn't suggest RFE).")
                continue
            
            if not config_data:
                 print(f"  Config: {config_name} - No tasks found with RFE selections.")
                 continue

            for task_name, lists_of_selected_features in config_data.items():
                print(f"\n  Config: {config_name} / Task: {task_name.upper()} (Mode: {mode}) for RFE aggregation")
                
                if not lists_of_selected_features:
                    print("    No RFE selection lists found for this task/config.")
                    aggregated_rfe_dfs[mode][config_name][task_name] = pd.DataFrame(columns=['Feature', 'Selection_Count', 'Selection_Frequency'])
                    continue

                n_runs_with_rfe_lists = len(lists_of_selected_features)
                print(f"    Found {n_runs_with_rfe_lists} runs with RFE feature lists.")

                if n_runs_with_rfe_lists == 0:
                    aggregated_rfe_dfs[mode][config_name][task_name] = pd.DataFrame(columns=['Feature', 'Selection_Count', 'Selection_Frequency'])
                    continue
                
                # Flatten the list of lists and count occurrences
                all_selected_features_flat = [feature for sublist in lists_of_selected_features for feature in sublist]
                feature_counts = collections.Counter(all_selected_features_flat)

                if not feature_counts:
                    print("    No features were selected across any run.")
                    aggregated_rfe_dfs[mode][config_name][task_name] = pd.DataFrame(columns=['Feature', 'Selection_Count', 'Selection_Frequency'])
                    continue

                rfe_summary_df = pd.DataFrame(feature_counts.items(), columns=['Feature', 'Selection_Count'])
                rfe_summary_df['Selection_Frequency'] = rfe_summary_df['Selection_Count'] / n_runs_with_rfe_lists
                rfe_summary_df.sort_values(by='Selection_Frequency', ascending=False, inplace=True)
                rfe_summary_df.reset_index(drop=True, inplace=True)

                aggregated_rfe_dfs[mode][config_name][task_name] = rfe_summary_df
                
                if config.SAVE_AGGREGATED_IMPORTANCES: # Reuse this flag, or add a new one
                    os.makedirs(config.DATA_OUTPUT_FOLDER, exist_ok=True)
                    filename = os.path.join(config.DATA_OUTPUT_FOLDER, f"rfe_selection_frequency_{mode}_{config_name}_{task_name}.csv")
                    try:
                        rfe_summary_df.to_csv(filename, index=False, sep=';', decimal='.', float_format='%.6f')
                        print(f"    -> Saved RFE selection frequency to: {filename}")
                    except Exception as e:
                        print(f"    Error saving RFE selection frequency for {mode}/{config_name}/{task_name}: {e}")
            
    return aggregated_rfe_dfs