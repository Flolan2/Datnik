# -*- coding: utf-8 -*-
"""
Functions for aggregating and summarizing results across multiple runs and configurations,
now handling different splitting modes.
"""

import pandas as pd
import numpy as np
import os
import collections
import logging # Added for consistency if not already present

logger = logging.getLogger('DatnikExperiment')


def aggregate_metrics(all_runs_metrics, config, output_dir_override=None): # Added output_dir_override
    """
    Aggregates metrics across repetitions for each mode, configuration, and task.

    Args:
        all_runs_metrics (dict): Nested dict: mode -> config_name -> task_name -> list of metric dicts.
        config (module): The main configuration module.
        output_dir_override (str, optional): If provided, save summary here instead of config.DATA_OUTPUT_FOLDER.
    """
    logger.info("\n===== Aggregating Performance Metrics (Across Modes) =====") # Changed from print to logger.info
    aggregated_summary = []
    metric_keys_display = ['roc_auc', 'accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'best_cv_score'] # Added best_cv_score

    if not all_runs_metrics:
        logger.warning("Input 'all_runs_metrics' dictionary is empty for metric aggregation.") # Changed from print
        return None

    for mode, mode_results in all_runs_metrics.items():
        logger.info(f"\n--- Processing Mode for Metrics: {mode.upper()} ---")
        if not mode_results:
            logger.info("No configurations found for this mode in metric aggregation.")
            continue

        for config_name, config_results in mode_results.items():
            if not config_results:
                 logger.info(f"  Config: {config_name} - No tasks found for metric aggregation.")
                 continue

            for task_name, task_runs_metrics in config_results.items():
                valid_runs = [run for run in task_runs_metrics if isinstance(run, dict) and run]
                
                # Ensure all metric_keys_display are present, defaulting to NaN if missing
                valid_metrics_list = []
                for run in valid_runs:
                    metric_dict = {}
                    has_any_metric = False
                    for k_display in metric_keys_display:
                        metric_dict[k_display] = run.get(k_display, np.nan)
                        if pd.notna(metric_dict[k_display]):
                            has_any_metric = True
                    if has_any_metric: # Only add if at least one display metric is not NaN
                        valid_metrics_list.append(metric_dict)

                n_total_runs = len(task_runs_metrics)
                n_valid_runs_with_any_metric = len(valid_metrics_list)

                logger.info(f"\n  Config: {config_name} / Task: {task_name.upper()} (Mode: {mode}) for metrics")
                logger.info(f"  Total runs attempted: {n_total_runs}, Valid runs with any displayable metrics: {n_valid_runs_with_any_metric}")

                row = {
                    'Mode': mode,
                    'Config_Name': config_name,
                    'Task_Name': task_name,
                    'N_Valid_Runs_With_Metrics': int(n_valid_runs_with_any_metric) # Renamed for clarity
                }

                if n_valid_runs_with_any_metric == 0:
                    logger.info("  No valid runs with metrics found for aggregation.")
                    for key in metric_keys_display:
                        row[f'{key}_mean'] = np.nan; row[f'{key}_std'] = np.nan
                else:
                    metrics_df = pd.DataFrame(valid_metrics_list)
                    # N_Valid_Runs_For_Metric can be calculated per metric if needed, but summary usually uses overall valid runs
                    
                    logger.info("  Mean +/- Std Dev:")
                    for key in metric_keys_display:
                        # Calculate mean/std only if the column exists and has non-NaN values
                        mean_val, std_val = np.nan, np.nan
                        n_for_this_metric = 0
                        if key in metrics_df.columns and metrics_df[key].notna().any():
                            mean_val = metrics_df[key].mean(skipna=True)
                            std_val = metrics_df[key].std(skipna=True)
                            n_for_this_metric = metrics_df[key].notna().sum()
                        
                        row[f'{key}_mean'] = mean_val; row[f'{key}_std'] = std_val
                        row[f'{key}_N'] = int(n_for_this_metric) # Number of valid runs for this specific metric
                        
                        if pd.notna(mean_val): 
                            logger.info(f"    {key}: {mean_val:.4f} ± {std_val:.4f} (N={n_for_this_metric})" if pd.notna(std_val) else f"    {key}: {mean_val:.4f} (N={n_for_this_metric})")
                        else: 
                            logger.info(f"    {key}: N/A (N={n_for_this_metric})")
                aggregated_summary.append(row)

    if not aggregated_summary:
        logger.warning("No valid results found across all modes to create metrics summary DataFrame.")
        return None

    summary_df = pd.DataFrame(aggregated_summary)
    cols_order = ['Mode', 'Config_Name', 'Task_Name', 'N_Valid_Runs_With_Metrics'] + \
                 sorted([col for col in summary_df.columns if col not in ['Mode','Config_Name','Task_Name','N_Valid_Runs_With_Metrics']])
    summary_df = summary_df[cols_order]

    if config.SAVE_AGGREGATED_SUMMARY:
        # MODIFIED: Use output_dir_override if provided, else use config.DATA_OUTPUT_FOLDER
        output_directory = output_dir_override if output_dir_override else config.DATA_OUTPUT_FOLDER
        os.makedirs(output_directory, exist_ok=True)
        
        filename = os.path.join(output_directory, "experiment_metrics_summary_modes.csv")
        try:
            summary_df.to_csv(filename, index=False, sep=';', decimal='.', float_format='%.6f')
            logger.info(f"\nAggregated metrics summary saved to: {filename}")
        except Exception as e: 
            logger.error(f"\nError saving aggregated metrics summary: {e}", exc_info=True)

    return summary_df


def aggregate_importances(all_runs_importances, config, output_dir_override=None, file_prefix="importance"): # Added output_dir_override
    """
    Aggregates feature importances/coefficients.
    Args:
        output_dir_override (str, optional): If provided, save summary here.
    """
    logger.info(f"\n===== Aggregating Feature Importances ({file_prefix}, Across Modes) =====")
    aggregated_dfs = collections.defaultdict(lambda: collections.defaultdict(dict))

    if not all_runs_importances:
        logger.warning("Input 'all_runs_importances' dictionary is empty.")
        return aggregated_dfs
    
    output_directory = output_dir_override if output_dir_override else config.DATA_OUTPUT_FOLDER


    for mode, mode_results in all_runs_importances.items():
        logger.info(f"\n--- Processing Importances for Mode: {mode.upper()} ---")
        if not mode_results:
            logger.info("No configurations found for this mode.")
            continue

        for config_name, config_imps in mode_results.items():
            if not config_imps:
                 logger.info(f"  Config: {config_name} - No tasks found for importances.")
                 continue

            for task_name, task_runs_imps in config_imps.items():
                valid_imps = [s for s in task_runs_imps if isinstance(s, pd.Series) and not s.empty]
                n_valid = len(valid_imps)

                logger.info(f"\n  Importances - Config: {config_name} / Task: {task_name.upper()} (Mode: {mode})")
                logger.info(f"  Found {n_valid} valid importance/coefficient sets.")

                if n_valid > 0:
                    try:
                        imp_df = pd.concat(valid_imps, axis=1, join='outer', keys=[f'run_{i}' for i in range(n_valid)]) # Use unique keys
                        agg_imp = pd.DataFrame({
                            'Mean_Importance': imp_df.mean(axis=1, skipna=True),
                            'Std_Importance': imp_df.std(axis=1, skipna=True),
                            'N_Valid_Runs': imp_df.notna().sum(axis=1).astype(int) # How many runs this feature appeared in
                        }, index=imp_df.index)
                        # Sort by absolute mean importance
                        agg_imp = agg_imp.reindex(agg_imp['Mean_Importance'].abs().sort_values(ascending=False, na_position='last').index)

                        aggregated_dfs[mode][config_name][task_name] = agg_imp

                        if config.SAVE_AGGREGATED_IMPORTANCES:
                            os.makedirs(output_directory, exist_ok=True)
                            # Ensure task_name is filename-safe
                            safe_task_name = str(task_name).replace('/','_').replace('\\','_').replace(':','_')
                            filename = os.path.join(output_directory, f"{file_prefix}_{mode}_{config_name}_{safe_task_name}_agg.csv")
                            agg_imp.to_csv(filename, sep=';', decimal='.', index_label='Feature', float_format='%.6f')
                            logger.info(f"    -> Saved aggregated importances to: {filename}")

                    except Exception as e:
                        logger.error(f"    Error aggregating/saving importances for {mode}/{config_name}/{task_name}: {e}", exc_info=True)
                        aggregated_dfs[mode][config_name][task_name] = pd.DataFrame()
                else:
                    logger.info("    No valid importance data to aggregate for this task/config/mode.")
                    aggregated_dfs[mode][config_name][task_name] = pd.DataFrame()
    return aggregated_dfs


def aggregate_rfe_features(all_runs_rfe_selected_features, config, output_dir_override=None): # Added output_dir_override
    """
    Aggregates RFE selected features.
    Args:
        output_dir_override (str, optional): If provided, save summary here.
    """
    logger.info(f"\n===== Aggregating RFE Selected Features (Across Modes) =====")
    aggregated_rfe_dfs = collections.defaultdict(lambda: collections.defaultdict(dict))

    if not all_runs_rfe_selected_features:
        logger.warning("Input 'all_runs_rfe_selected_features' dictionary is empty.")
        return aggregated_rfe_dfs

    output_directory = output_dir_override if output_dir_override else config.DATA_OUTPUT_FOLDER

    for mode, mode_results in all_runs_rfe_selected_features.items():
        logger.info(f"\n--- Processing RFE Features for Mode: {mode.upper()} ---")
        if not mode_results:
            logger.info("  No configurations found for this mode.")
            continue

        for config_name, config_data in mode_results.items():
            if 'RFE' not in config_name.upper(): # Heuristic to only process RFE configs
                continue
            
            if not config_data:
                 logger.info(f"  Config: {config_name} - No tasks found with RFE selections.")
                 continue

            for task_name, lists_of_selected_features in config_data.items():
                logger.info(f"\n  RFE - Config: {config_name} / Task: {task_name.upper()} (Mode: {mode})")
                
                if not lists_of_selected_features: # Ensure it's not None and not empty
                    logger.info("    No RFE selection lists found for this task/config/mode.")
                    aggregated_rfe_dfs[mode][config_name][task_name] = pd.DataFrame(columns=['Feature', 'Selection_Count', 'Selection_Frequency'])
                    continue

                # Filter out any None entries just in case, though append logic should prevent this
                valid_lists_of_features = [lst for lst in lists_of_selected_features if lst is not None]
                if not valid_lists_of_features:
                    logger.info("    No valid (non-None) RFE selection lists found after filtering.")
                    aggregated_rfe_dfs[mode][config_name][task_name] = pd.DataFrame(columns=['Feature', 'Selection_Count', 'Selection_Frequency'])
                    continue

                n_runs_with_rfe_lists = len(valid_lists_of_features)
                logger.info(f"    Found {n_runs_with_rfe_lists} valid runs with RFE feature lists.")

                if n_runs_with_rfe_lists == 0: # Should be caught by above, but defensive
                    aggregated_rfe_dfs[mode][config_name][task_name] = pd.DataFrame(columns=['Feature', 'Selection_Count', 'Selection_Frequency'])
                    continue
                
                all_selected_features_flat = [feature for sublist in valid_lists_of_features for feature in sublist]
                feature_counts = collections.Counter(all_selected_features_flat)

                if not feature_counts:
                    logger.info("    No features were selected across any valid run.")
                    aggregated_rfe_dfs[mode][config_name][task_name] = pd.DataFrame(columns=['Feature', 'Selection_Count', 'Selection_Frequency'])
                    continue

                rfe_summary_df = pd.DataFrame(feature_counts.items(), columns=['Feature', 'Selection_Count'])
                rfe_summary_df['Selection_Frequency'] = rfe_summary_df['Selection_Count'] / n_runs_with_rfe_lists
                rfe_summary_df.sort_values(by=['Selection_Frequency', 'Selection_Count', 'Feature'], ascending=[False, False, True], inplace=True)
                rfe_summary_df.reset_index(drop=True, inplace=True)

                aggregated_rfe_dfs[mode][config_name][task_name] = rfe_summary_df
                
                # Reuse SAVE_AGGREGATED_IMPORTANCES flag, or add a specific one like SAVE_RFE_SUMMARY
                if config.SAVE_AGGREGATED_IMPORTANCES: 
                    os.makedirs(output_directory, exist_ok=True)
                    safe_task_name = str(task_name).replace('/','_').replace('\\','_').replace(':','_')
                    filename = os.path.join(output_directory, f"rfe_selection_frequency_{mode}_{config_name}_{safe_task_name}.csv")
                    try:
                        rfe_summary_df.to_csv(filename, index=False, sep=';', decimal='.', float_format='%.6f')
                        logger.info(f"    -> Saved RFE selection frequency to: {filename}")
                    except Exception as e:
                        logger.error(f"    Error saving RFE selection frequency for {mode}/{config_name}/{task_name}: {e}", exc_info=True)
            
    return aggregated_rfe_dfs