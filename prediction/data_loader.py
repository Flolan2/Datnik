# --- START OF FILE prediction/data_loader.py (CORRECTED) ---
# -*- coding: utf-8 -*-
"""
Functions for loading and preparing the data FOR BINARY Classification.
Includes feature engineering capabilities based on global config.
All data is controlled for Age before binarization and modeling.
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from statsmodels.formula.api import ols

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
        try:
            df = pd.read_csv(input_file_path, sep=';', decimal='.')
            logger.info("Successfully loaded data using ';' separator.")
        except (pd.errors.ParserError, UnicodeDecodeError, FileNotFoundError) :
             logger.warning("Failed read with ';' or file not found, trying ',' separator...")
             df = pd.read_csv(input_file_path, sep=',', decimal='.')
             logger.info("Successfully loaded data using ',' separator.")
        logger.info(f"Data loaded successfully. Initial shape: {df.shape}")
        return df
    except Exception as e:
        logger.exception(f"Error loading or parsing data from {input_file_path}:")
        sys.exit(1)

def prepare_data(df, config,
                 target_z_score_column_override=None,
                 abnormality_threshold_override=None):
    """
    Prepares the dataframe for BINARY classification, controlling for Age.
    It residualizes both kinematic features (X) and the continuous imaging
    target (y) against Age before binarizing the target.

    Args:
        df (pd.DataFrame): The raw input DataFrame.
        config (module): The main configuration module.
        target_z_score_column_override (str, optional): The DaTscan column to use.
        abnormality_threshold_override (float, optional): The threshold for binarization.
    Returns:
        tuple: X_full (residualized features), y_full (binary target from residuals),
               groups_full, task_features_map, all_feature_cols.
    """
    current_target_z_col_name = target_z_score_column_override or config.TARGET_Z_SCORE_COL
    current_threshold_to_use = abnormality_threshold_override if abnormality_threshold_override is not None else config.ABNORMALITY_THRESHOLD

    data_full = df.copy()

    # --- Initial Data Cleaning and Feature Identification (as before) ---
    if config.GROUP_ID_COL not in data_full.columns:
        logger.error(f"Group ID column '{config.GROUP_ID_COL}' not found.")
        return pd.DataFrame(), pd.Series(dtype='int'), pd.Series(dtype='str'), {}, []
    data_full[config.GROUP_ID_COL] = data_full[config.GROUP_ID_COL].astype(str).str.strip()

    defined_task_prefixes = sorted(list(set(
       exp_conf.get('task_prefix_for_features') for exp_conf in config.CONFIGURATIONS_TO_RUN
       if exp_conf.get('task_prefix_for_features')
    )))

    task_features_map = {prefix: [] for prefix in defined_task_prefixes}
    all_initial_feature_cols = []
    for prefix in defined_task_prefixes:
        base_cols = [f"{prefix}_{b}" for b in config.BASE_KINEMATIC_COLS if f"{prefix}_{b}" in data_full.columns]
        if base_cols:
            task_features_map[prefix].extend(base_cols)
            all_initial_feature_cols.extend(base_cols)

    all_initial_feature_cols = sorted(list(set(all_initial_feature_cols)))
    if not all_initial_feature_cols:
        return pd.DataFrame(index=data_full.index), pd.Series(dtype='int'), data_full.get(config.GROUP_ID_COL), {}, []

    X_full = data_full[all_initial_feature_cols].copy()
    for col in X_full.columns:
        X_full[col] = pd.to_numeric(X_full[col].astype(str).str.replace(',', '.'), errors='coerce')

    # --- Apply Feature Engineering (BEFORE residualization) ---
    any_config_uses_fe = any(c.get('apply_feature_engineering') for c in config.CONFIGURATIONS_TO_RUN)
    if any_config_uses_fe:
        if fe is not None and hasattr(config, 'FEATURE_ENGINEERING_SETS_OPTIMIZED'):
            fe_sets = config.FEATURE_ENGINEERING_SETS_OPTIMIZED
            original_cols = X_full.columns.tolist()
            for fe_set in fe_sets:
                fe_func_name = fe_set.get('function')
                fe_params = fe_set.get('params', {}).copy()
                if not fe_func_name or not hasattr(fe, fe_func_name): continue
                try:
                    new_features_df = getattr(fe, fe_func_name)(X_full, **fe_params)
                    if new_features_df is not None:
                        for new_col in new_features_df.columns: X_full[new_col] = new_features_df[new_col]
                except Exception: pass
            
            all_engineered_cols = [c for c in X_full.columns if c not in original_cols]
            for prefix in defined_task_prefixes:
                engineered_for_task = [c for c in all_engineered_cols if c.startswith(prefix + "_")]
                if prefix in task_features_map: # Check key exists before extending
                    task_features_map[prefix].extend(engineered_for_task)
                    task_features_map[prefix] = sorted(list(set(task_features_map[prefix])))

    # <<< --- AGE CONTROL LOGIC (WITH CORRECTION) START --- >>>
    logger.debug(f"Applying mandatory age control using '{config.AGE_COL}' column...")

    # 1. Check for essential columns in the original dataframe
    required_cols = [current_target_z_col_name, config.AGE_COL, config.GROUP_ID_COL]
    missing_req_cols = [c for c in required_cols if c not in data_full.columns]
    if missing_req_cols:
        logger.error(f"Missing essential columns for age control: {missing_req_cols}. Cannot proceed.")
        return pd.DataFrame(), pd.Series(dtype='int'), pd.Series(dtype='str'), {}, []

    # <<< --- OLD FAULTY LOGIC (FOR REFERENCE) --- >>>
    # cols_for_residualization = required_cols + list(X_full.columns)
    # analysis_df = data_full[list(set(cols_for_residualization))].dropna().copy()
    # <<< --- END OLD FAULTY LOGIC --- >>>

    # <<< --- NEW CORRECTED LOGIC --- >>>
    # 2. Correctly construct the dataframe for analysis by joining metadata with the fully-featured X_full
    # Select metadata columns from the original dataframe
    metadata_df = data_full[required_cols]
    
    # Join metadata with the feature dataframe (which now includes engineered features)
    # An inner join ensures we only have rows with complete data for both parts.
    temp_analysis_df = metadata_df.join(X_full, how='inner')

    # 3. Now, drop rows with any missing values across all columns needed for residualization
    analysis_df = temp_analysis_df.dropna().copy()
    # <<< --- END NEW CORRECTED LOGIC --- >>>

    if analysis_df.empty or analysis_df[config.AGE_COL].nunique() < 2:
        logger.warning("Not enough valid data (after dropping NaNs for Age, Target, and Features) to perform age control.")
        return pd.DataFrame(), pd.Series(dtype='int'), pd.Series(dtype='str'), {}, []

    # 4. Control the CONTINUOUS imaging target first
    y_model = ols(f"Q('{current_target_z_col_name}') ~ Q('{config.AGE_COL}')", data=analysis_df).fit()
    y_continuous_resid = y_model.resid

    # 5. Binarize the RESIDUALIZED target
    y_full = (y_continuous_resid <= current_threshold_to_use).astype(int)
    y_full.name = config.TARGET_COLUMN_NAME

    # 6. Control all features (X)
    X_resid_df = pd.DataFrame(index=analysis_df.index)
    for col in X_full.columns:
        if col in analysis_df.columns:
            x_model = ols(f"Q('{col}') ~ Q('{config.AGE_COL}')", data=analysis_df).fit()
            X_resid_df[col] = x_model.resid

    X_full = X_resid_df
    groups_full = analysis_df[config.GROUP_ID_COL].copy()
    # <<< --- AGE CONTROL LOGIC END --- >>>

    # --- Final Data Cleaning and Alignment ---
    all_feature_cols = list(X_full.columns)
    cols_all_nan_in_X = X_full.columns[X_full.isnull().all()].tolist()
    if cols_all_nan_in_X:
         X_full.drop(columns=cols_all_nan_in_X, inplace=True)
         all_feature_cols = X_full.columns.tolist()
         for task_key in list(task_features_map.keys()):
             task_features_map[task_key] = [f for f in task_features_map[task_key] if f in all_feature_cols]
             if not task_features_map[task_key]: del task_features_map[task_key]

    if X_full.empty:
        return pd.DataFrame(index=y_full.index), y_full, groups_full, task_features_map, []

    if not (len(X_full) == len(y_full) == len(groups_full)):
        logger.error(f"CRITICAL FINAL LENGTH MISMATCH after age control. X:{len(X_full)}, y:{len(y_full)}, groups:{len(groups_full)}")
        return pd.DataFrame(), pd.Series(dtype='int'), pd.Series(dtype='str'), {}, []

    return X_full, y_full, groups_full, task_features_map, all_feature_cols
# --- END OF FILE prediction/data_loader.py (CORRECTED) ---