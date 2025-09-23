# -*- coding: utf-8 -*-
"""
Functions for loading and preparing the data FOR BINARY Classification.
Includes feature engineering capabilities based on global config.
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
# Ensure feature_engineering module can be imported if FE is enabled
try:
    from . import feature_engineering as fe
except ImportError:
    fe = None # Will be checked later if FE is enabled

logger = logging.getLogger('DatnikExperiment')

def load_data(input_folder, csv_name):
    """Loads the specified CSV file, trying different separators."""
    input_file_path = os.path.join(input_folder, csv_name)
    logger.info(f"Attempting to load data from: {input_file_path}")
    if not os.path.exists(input_file_path):
        logger.error(f"Error: Input file not found at {input_file_path}")
        sys.exit(1)
    df = None
    try:
        # Try reading with semicolon first
        try:
            df = pd.read_csv(input_file_path, sep=';', decimal='.')
            logger.info("Successfully loaded data using ';' separator.")
        except (pd.errors.ParserError, UnicodeDecodeError, FileNotFoundError) : # More specific exceptions
             logger.warning("Failed read with ';' or file not found, trying ',' separator...")
             df = pd.read_csv(input_file_path, sep=',', decimal='.')
             logger.info("Successfully loaded data using ',' separator.")
        logger.info(f"Data loaded successfully. Initial shape: {df.shape}")
        return df
    except Exception as e: # Catch any other exception during loading
        logger.exception(f"Error loading or parsing data from {input_file_path}:")
        sys.exit(1)

def prepare_data(df, config, 
                 target_z_score_column_override=None, 
                 abnormality_threshold_override=None):
    """
    Prepares the dataframe for BINARY classification.

    Args:
        df (pd.DataFrame): The raw input DataFrame.
        config (module): The main configuration module.
        target_z_score_column_override (str, optional): The name of the DaTscan Z-score
            column to use for defining the binary target. If None, uses config.TARGET_Z_SCORE_COL.
        abnormality_threshold_override (float, optional): If provided, this Z-score
            threshold will be used. If None, uses config.ABNORMALITY_THRESHOLD.
    Returns:
        tuple: X_full (all features), y_full, groups_full,
               task_features_map (mapping task prefix to list of all its features - original + engineered),
               all_feature_cols (list of all unique feature column names created).
    """
    current_target_z_col_name = target_z_score_column_override if target_z_score_column_override else config.TARGET_Z_SCORE_COL
    current_threshold_to_use = abnormality_threshold_override if abnormality_threshold_override is not None else config.ABNORMALITY_THRESHOLD

    # Reduced verbosity for this function as it's called many times
    # logger.info(f"Preparing data for BINARY classification...")
    # logger.info(f"Using DaTscan Z-score column for target: '{current_target_z_col_name}'")
    # logger.info(f"Using abnormality Z-score threshold: {current_threshold_to_use:.3f}")

    data_full = df.copy()

    if config.GROUP_ID_COL not in data_full.columns:
        logger.error(f"Group ID column '{config.GROUP_ID_COL}' not found.")
        return pd.DataFrame(), pd.Series(dtype='int'), pd.Series(dtype='str'), {}, [] # Fatal for this iteration
    data_full[config.GROUP_ID_COL] = data_full[config.GROUP_ID_COL].astype(str).str.strip()
    initial_rows_df = len(data_full)
    data_full.dropna(subset=[config.GROUP_ID_COL], inplace=True)
    # if len(data_full) < initial_rows_df: logger.warning(f"Dropped {initial_rows_df - len(data_full)} rows with missing group ID.")

    if current_target_z_col_name not in data_full.columns:
        logger.error(f"Specified Target Z-score column '{current_target_z_col_name}' not found in input CSV for target definition.")
        return pd.DataFrame(), pd.Series(dtype='int'), pd.Series(dtype='str'), {}, []
        
    data_full[current_target_z_col_name] = pd.to_numeric(data_full[current_target_z_col_name].astype(str).str.replace(',', '.'), errors='coerce')
    initial_rows_after_group_drop = len(data_full)
    data_full.dropna(subset=[current_target_z_col_name], inplace=True)
    # if len(data_full) < initial_rows_after_group_drop: logger.warning(f"Dropped {initial_rows_after_group_drop - len(data_full)} rows with missing/invalid values in '{current_target_z_col_name}'.")

    if data_full.empty:
        # logger.error(f"No data remaining after processing Group ID and '{current_target_z_col_name}'.") # Covered by main script
        return pd.DataFrame(), pd.Series(dtype='int'), pd.Series(dtype='str'), {}, []

    target_col_name_for_y = config.TARGET_COLUMN_NAME
    data_full[target_col_name_for_y] = (data_full[current_target_z_col_name] <= current_threshold_to_use).astype(int)

    y_full = data_full[target_col_name_for_y].copy()
    groups_full = data_full[config.GROUP_ID_COL].copy()

    # Minimal logging of distribution, full log in main script
    # n_classes = y_full.nunique()
    # logger.debug(f"Target '{target_col_name_for_y}' (from '{current_target_z_col_name}' @ {current_threshold_to_use:.3f}): N={len(y_full)}, Classes={n_classes}, Groups={groups_full.nunique()}")

    if len(y_full) != len(groups_full):
         logger.error(f"Mismatch length y_full/groups_full for target '{current_target_z_col_name}'.")
         return pd.DataFrame(), pd.Series(dtype='int'), pd.Series(dtype='str'), {}, []

    defined_task_prefixes = sorted(list(set(
       exp_conf.get('task_prefix_for_features') for exp_conf in config.CONFIGURATIONS_TO_RUN
       if exp_conf.get('task_prefix_for_features')
    ))) # Assuming config.CONFIGURATIONS_TO_RUN is always valid as checked in main
    
    task_features_map = {task_prefix: [] for task_prefix in defined_task_prefixes}
    all_initial_feature_cols = []
    for task_prefix in defined_task_prefixes:
        task_base_cols = [f"{task_prefix}_{base}" for base in config.BASE_KINEMATIC_COLS if f"{task_prefix}_{base}" in data_full.columns]
        if task_base_cols:
            task_features_map[task_prefix].extend(task_base_cols)
            all_initial_feature_cols.extend(task_base_cols)

    all_initial_feature_cols = sorted(list(set(all_initial_feature_cols)))
    if not all_initial_feature_cols:
        # logger.warning(f"No base kinematic features found for target '{current_target_z_col_name}'. X will be empty.")
        return pd.DataFrame(index=y_full.index), y_full, groups_full, {}, []

    existing_initial_feature_cols = [col for col in all_initial_feature_cols if col in data_full.columns]
    if not existing_initial_feature_cols:
        return pd.DataFrame(index=y_full.index), y_full, groups_full, {}, []
        
    X_full = data_full[existing_initial_feature_cols].copy()
    for col in X_full.columns:
        X_full[col] = pd.to_numeric(X_full[col].astype(str).str.replace(',', '.'), errors='coerce')
    all_feature_cols = list(X_full.columns) # Base features that exist

    if hasattr(config, 'ENABLE_FEATURE_ENGINEERING') and config.ENABLE_FEATURE_ENGINEERING:
        if fe is not None and hasattr(config, 'FEATURE_ENGINEERING_SETS_OPTIMIZED') and config.FEATURE_ENGINEERING_SETS_OPTIMIZED:
            fe_sets_to_apply = config.FEATURE_ENGINEERING_SETS_OPTIMIZED
            original_X_cols_before_fe_loop = X_full.columns.tolist()
            for fe_set in fe_sets_to_apply:
                fe_func_name = fe_set.get('function')
                fe_params = fe_set.get('params', {}).copy()
                if not fe_func_name or not hasattr(fe, fe_func_name): continue
                fe_function = getattr(fe, fe_func_name)
                try:
                    new_features_df_subset = fe_function(X_full, **fe_params)
                    if new_features_df_subset is not None and not new_features_df_subset.empty:
                        for new_col in new_features_df_subset.columns:
                            X_full[new_col] = new_features_df_subset[new_col]
                except Exception: pass # Errors in FE funcs logged there or too verbose here
            
            current_all_X_columns = X_full.columns.tolist()
            all_engineered_cols_added = [col for col in current_all_X_columns if col not in original_X_cols_before_fe_loop]
            if all_engineered_cols_added:
                all_feature_cols.extend(all_engineered_cols_added)
                all_feature_cols = sorted(list(set(all_feature_cols)))
                for task_prefix_map_update in defined_task_prefixes:
                    task_specific_engineered_cols = [col for col in all_engineered_cols_added if col.startswith(task_prefix_map_update + "_")]
                    if task_prefix_map_update in task_features_map: # Should always be true
                        task_features_map[task_prefix_map_update].extend(task_specific_engineered_cols)
                        task_features_map[task_prefix_map_update] = sorted(list(set(task_features_map[task_prefix_map_update])))
    
    cols_all_nan_in_X = X_full.columns[X_full.isnull().all()].tolist()
    if cols_all_nan_in_X:
         X_full.drop(columns=cols_all_nan_in_X, inplace=True)
         all_feature_cols = X_full.columns.tolist()
         for task_key_cleanup in list(task_features_map.keys()):
             task_features_map[task_key_cleanup] = [f_name for f_name in task_features_map[task_key_cleanup] if f_name in all_feature_cols]
             if not task_features_map[task_key_cleanup]: del task_features_map[task_key_cleanup]

    if X_full.empty :
        return pd.DataFrame(index=y_full.index), y_full, groups_full, task_features_map, []

    common_index = X_full.index.intersection(y_full.index).intersection(groups_full.index)
    if len(common_index) != len(X_full) or len(common_index) != len(y_full) or len(common_index) != len(groups_full) :
        X_full = X_full.loc[common_index]
        y_full = y_full.loc[common_index]
        groups_full = groups_full.loc[common_index]

    if not (len(X_full) == len(y_full) == len(groups_full)):
        logger.error(f"CRITICAL FINAL LENGTH MISMATCH for target '{current_target_z_col_name}'.")
        return pd.DataFrame(), pd.Series(dtype='int'), pd.Series(dtype='str'), {}, []

    return X_full, y_full, groups_full, task_features_map, all_feature_cols