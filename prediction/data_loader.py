import pandas as pd
import numpy as np
import os
import sys
import logging

try:
    from . import feature_engineering as fe
except ImportError:
    fe = None 

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

def prepare_data_pre_split(df, config, target_z_score_column_override=None):
    """
    Prepares the dataframe BEFORE splitting for classification.
    - Loads data
    - Performs feature engineering
    - Filters to necessary columns
    - Returns RAW features (X), RAW continuous target (y), groups, and Age.
    - DOES NOT perform age residualization to prevent data leakage.

    Args:
        df (pd.DataFrame): The raw input DataFrame.
        config (module): The main configuration module.
        target_z_score_column_override (str, optional): The DaTscan column to use.

    Returns:
        tuple: X_full (features), y_continuous (continuous target),
               groups_full, age_full, task_features_map, all_feature_cols.
    """
    current_target_z_col_name = target_z_score_column_override or config.TARGET_Z_SCORE_COL
    data_full = df.copy()

    # --- Initial Data Cleaning and Feature Identification ---
    required_cols = [config.GROUP_ID_COL, config.AGE_COL, current_target_z_col_name]
    missing_req_cols = [c for c in required_cols if c not in data_full.columns]
    if missing_req_cols:
        logger.error(f"Missing essential columns for analysis: {missing_req_cols}. Cannot proceed.")
        return pd.DataFrame(), pd.Series(dtype='float'), pd.Series(dtype='str'), pd.Series(dtype='float'), {}, []

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
        return pd.DataFrame(index=data_full.index), pd.Series(dtype='int'), data_full.get(config.GROUP_ID_COL), data_full.get(config.AGE_COL), {}, []

    X_full = data_full[all_initial_feature_cols].copy()
    for col in X_full.columns:
        X_full[col] = pd.to_numeric(X_full[col].astype(str).str.replace(',', '.'), errors='coerce')

    # --- Apply Feature Engineering ---
    any_config_uses_fe = any(c.get('apply_feature_engineering') for c in config.CONFIGURATIONS_TO_RUN)
    if any_config_uses_fe:
        if fe is not None and hasattr(config, 'FEATURE_ENGINEERING_SETS_OPTIMIZED'):
            all_fe_sets = config.FEATURE_ENGINEERING_SETS_OPTIMIZED
            relevant_fe_sets = [
                fe_set for fe_set in all_fe_sets
                if any(fe_set.get('name', '').startswith(prefix + '_') for prefix in defined_task_prefixes)
            ]
            logger.info(f"Identified {len(relevant_fe_sets)} relevant feature engineering definitions for active tasks: {defined_task_prefixes}")

            original_cols = X_full.columns.tolist()
            for fe_set in relevant_fe_sets:
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
                if prefix in task_features_map:
                    task_features_map[prefix].extend(engineered_for_task)
                    task_features_map[prefix] = sorted(list(set(task_features_map[prefix])))

   
    temp_analysis_df = X_full.join(data_full[required_cols], how='inner')

    analysis_df = temp_analysis_df.dropna().copy()

    X_full = analysis_df.drop(columns=required_cols, errors='ignore')
    y_continuous = analysis_df[current_target_z_col_name]
    groups_full = analysis_df[config.GROUP_ID_COL]
    age_full = analysis_df[config.AGE_COL]
    
    all_feature_cols = list(X_full.columns)

    if not (len(X_full) == len(y_continuous) == len(groups_full) == len(age_full)):
        logger.error(f"CRITICAL FINAL LENGTH MISMATCH after pre-split prep. X:{len(X_full)}, y:{len(y_continuous)}, groups:{len(groups_full)}, age:{len(age_full)}")
        return pd.DataFrame(), pd.Series(dtype='float'), pd.Series(dtype='str'), pd.Series(dtype='float'), {}, []
    
    return X_full, y_continuous, groups_full, age_full, task_features_map, all_feature_cols
