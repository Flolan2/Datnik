# -*- coding: utf-8 -*-
"""
Functions for loading and preparing the data FOR BINARY Classification.
Includes feature engineering capabilities.
"""

import pandas as pd
import numpy as np
import os
import sys
import logging

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
        except:
             logger.warning("Failed read with ';', trying ',' separator...")
             df = pd.read_csv(input_file_path, sep=',', decimal='.')
             logger.info("Successfully loaded data using ',' separator.")
        logger.info(f"Data loaded successfully. Initial shape: {df.shape}")
        return df
    except Exception as e:
        logger.exception(f"Error loading or parsing data from {input_file_path}:")
        sys.exit(1)

def prepare_data(df, config):
    """
    Prepares the dataframe for BINARY classification: handles target, groups, features, cleaning,
    and applies feature engineering if enabled.
    Returns X_full, y_full, groups_full, task_features_map, all_feature_cols
    """
    logger.info(f"Preparing data for BINARY classification using settings from {config.__name__}.py")

    data_full = df.copy()

    # 1. Prepare Group ID
    if config.GROUP_ID_COL not in data_full.columns:
        logger.error(f"Group ID column '{config.GROUP_ID_COL}' not found.")
        sys.exit(1)
    data_full[config.GROUP_ID_COL] = data_full[config.GROUP_ID_COL].astype(str).str.strip()
    initial_rows = len(data_full)
    data_full.dropna(subset=[config.GROUP_ID_COL], inplace=True)
    if len(data_full) < initial_rows:
        logger.warning(f"Dropped {initial_rows - len(data_full)} rows with missing group ID ('{config.GROUP_ID_COL}').")

    # 2. Prepare Z-Score Column (needed for binary target)
    if config.TARGET_Z_SCORE_COL not in data_full.columns:
        logger.error(f"Target Z-score column '{config.TARGET_Z_SCORE_COL}' not found.")
        sys.exit(1)
    data_full[config.TARGET_Z_SCORE_COL] = pd.to_numeric(data_full[config.TARGET_Z_SCORE_COL].astype(str).str.replace(',', '.'), errors='coerce')
    initial_rows = len(data_full)
    data_full.dropna(subset=[config.TARGET_Z_SCORE_COL], inplace=True)
    if len(data_full) < initial_rows:
        logger.warning(f"Dropped {initial_rows - len(data_full)} rows with missing/invalid target Z-score ('{config.TARGET_Z_SCORE_COL}').")
    if data_full.empty:
        logger.error("No data remaining after handling missing Group IDs or Target Z-scores.")
        sys.exit(1)

    # 3. Define BINARY Target Variable
    target_col_name = config.TARGET_COLUMN_NAME
    threshold = config.ABNORMALITY_THRESHOLD
    logger.info(f"Defining binary target variable '{target_col_name}' using threshold Z={threshold}")
    data_full[target_col_name] = (data_full[config.TARGET_Z_SCORE_COL] <= threshold).astype(int)

    initial_rows = len(data_full)
    data_full.dropna(subset=[target_col_name], inplace=True)
    if len(data_full) < initial_rows:
        logger.warning(f"Dropped {initial_rows - len(data_full)} rows with missing target label ('{target_col_name}') after assignment.")
    if data_full.empty:
        logger.error(f"No data remaining after assigning target variable '{target_col_name}'.")
        sys.exit(1)
    data_full[target_col_name] = data_full[target_col_name].astype(int)

    y_full = data_full[target_col_name]
    groups_full = data_full[config.GROUP_ID_COL]

    logger.info(f"Target variable '{target_col_name}' distribution (N={len(y_full)}):")
    logger.info(f"\n-- Normalized --\n{y_full.value_counts(normalize=True).sort_index().round(3)}")
    logger.info(f"\n-- Counts --\n{y_full.value_counts(normalize=False).sort_index()}")
    n_classes = y_full.nunique()
    logger.info(f"Number of unique classes in target: {n_classes}")
    logger.info(f"Number of unique patients: {groups_full.nunique()}")
    if n_classes < 2:
        logger.error("Target variable has only one class after final preparation. Cannot perform binary classification.")
        sys.exit(1)
    if len(y_full) != len(groups_full):
         logger.error("Mismatch between length of y_full and groups_full after processing.")
         sys.exit(1)

    # 4. Identify and Prepare Base Features
    all_initial_feature_cols = []
    task_features_map = {}
    logger.info("Identifying base features for tasks:")
    for task_prefix in config.TASKS_TO_RUN_SEPARATELY:
        task_cols = [f"{task_prefix}_{base}" for base in config.BASE_KINEMATIC_COLS if f"{task_prefix}_{base}" in data_full.columns]
        if task_cols:
            logger.info(f"  - Task '{task_prefix}': Found {len(task_cols)} base features.")
            task_features_map[task_prefix] = task_cols
            all_initial_feature_cols.extend(task_cols)
        else:
            logger.warning(f"  - Task '{task_prefix}': No base features found matching BASE_KINEMATIC_COLS.")

    all_initial_feature_cols = sorted(list(set(all_initial_feature_cols)))
    if not all_initial_feature_cols:
        logger.error("No base kinematic features found. Check BASE_KINEMATIC_COLS and task prefixes.")
        sys.exit(1)
    logger.info(f"Total unique base kinematic features identified: {len(all_initial_feature_cols)}")

    X_full = data_full[all_initial_feature_cols].copy()
    for col in all_initial_feature_cols:
        X_full[col] = pd.to_numeric(X_full[col].astype(str).str.replace(',', '.'), errors='coerce')

    # Make a working copy of all_initial_feature_cols to become all_feature_cols
    all_feature_cols = list(all_initial_feature_cols)

    # --- 5. Apply Feature Engineering ---
    if hasattr(config, 'ENABLE_FEATURE_ENGINEERING') and config.ENABLE_FEATURE_ENGINEERING:
        logger.info("--- Applying Feature Engineering ---")
        fe = None
        try:
            from prediction import feature_engineering as fe
        except ImportError:
            logger.error("Could not import prediction.feature_engineering. Skipping feature engineering.")
        
        if fe and hasattr(config, 'FEATURE_ENGINEERING_SETS') and config.FEATURE_ENGINEERING_SETS:
            original_X_cols = X_full.columns.tolist() # Columns before adding new ones
            
            for fe_set_idx, fe_set in enumerate(config.FEATURE_ENGINEERING_SETS):
                fe_func_name = fe_set.get('function')
                fe_params = fe_set.get('params', {}).copy()
                fe_set_name = fe_set.get('name', f'unnamed_fe_set_{fe_set_idx}')

                if not fe_func_name or not hasattr(fe, fe_func_name):
                    logger.warning(f"  FE Set '{fe_set_name}': Function '{fe_func_name}' not found in feature_engineering.py. Skipping.")
                    continue

                fe_function = getattr(fe, fe_func_name)
                logger.info(f"  Applying FE Set '{fe_set_name}' using function '{fe_func_name}' with params: {fe_params}")

                try:
                    # The feature engineering function is expected to take X_full and params,
                    # and return a DataFrame containing *only* the new feature column(s).
                    new_features_df_subset = fe_function(X_full, **fe_params)

                    if new_features_df_subset is not None and not new_features_df_subset.empty:
                        if new_features_df_subset.isnull().all().all():
                            logger.warning(f"    FE Set '{fe_set_name}' resulted in all NaN columns. Check inputs or function logic.")
                        else:
                            # Add new features to X_full, handling potential overwrites
                            for new_col in new_features_df_subset.columns:
                                if new_col in X_full.columns:
                                    logger.warning(f"    Engineered feature '{new_col}' from set '{fe_set_name}' already exists. Overwriting.")
                                X_full[new_col] = new_features_df_subset[new_col]
                            logger.info(f"    Successfully added/updated {len(new_features_df_subset.columns)} features from '{fe_set_name}'.")
                    else:
                        logger.warning(f"    Function {fe_func_name} for FE Set '{fe_set_name}' returned empty or None DataFrame.")
                except Exception as e_fe:
                    logger.error(f"    Error applying function '{fe_func_name}' for FE Set '{fe_set_name}': {e_fe}", exc_info=True)
            
            # Update all_feature_cols and task_features_map after all FE sets are processed
            newly_added_or_updated_cols = [col for col in X_full.columns if col not in original_X_cols]
            all_feature_cols.extend(newly_added_or_updated_cols)
            all_feature_cols = sorted(list(set(all_feature_cols))) # Unique and sorted

            for task_prefix_map_update in config.TASKS_TO_RUN_SEPARATELY:
                # Add new features that start with this task_prefix to its map
                # This assumes new engineered features are also prefixed (e.g., "ft_ratio_xyz")
                task_specific_new_cols = [col for col in newly_added_or_updated_cols if col.startswith(task_prefix_map_update + "_")]
                if task_prefix_map_update in task_features_map:
                    task_features_map[task_prefix_map_update].extend(task_specific_new_cols)
                    task_features_map[task_prefix_map_update] = sorted(list(set(task_features_map[task_prefix_map_update])))
                else: # If task had no base features but gets engineered ones
                    task_features_map[task_prefix_map_update] = sorted(list(set(task_specific_new_cols)))
            
            if newly_added_or_updated_cols:
                 logger.info(f"  Finished feature engineering. Added/updated {len(newly_added_or_updated_cols)} columns. Total features now: {len(all_feature_cols)}")
                 logger.debug(f"   Columns after FE: {newly_added_or_updated_cols}")
            else:
                 logger.info("  Feature engineering was enabled, but no new features were added/updated.")
        else:
            logger.info("Feature engineering is enabled but no 'FEATURE_ENGINEERING_SETS' defined in config, or module not found.")
    else:
        logger.info("Feature engineering is disabled in config.")


    # --- 6. Clean up features (Drop all-NaN columns) ---
    # This step is crucial after base feature creation and after feature engineering
    logger.info("Cleaning feature set (dropping all-NaN columns)...")
    cols_all_nan = X_full.columns[X_full.isnull().all()].tolist()
    if cols_all_nan:
         logger.warning(f"Dropping fully NaN feature columns: {cols_all_nan}")
         X_full.drop(columns=cols_all_nan, inplace=True)
         all_feature_cols = X_full.columns.tolist() # Update the master list
         # Update task_features_map as well
         for task_key_cleanup in list(task_features_map.keys()): # Iterate over a copy of keys
             task_features_map[task_key_cleanup] = [f_name for f_name in task_features_map[task_key_cleanup] if f_name in all_feature_cols]
             if not task_features_map[task_key_cleanup]: # If a task no longer has any features
                  logger.warning(f"Task '{task_key_cleanup}' removed from task_features_map as it has no remaining valid features after NaN drop.")
                  del task_features_map[task_key_cleanup]
         if not all_feature_cols: # If all features for all tasks were NaN
              logger.error("CRITICAL: No valid feature columns remaining after dropping all-NaN columns. Cannot proceed.")
              sys.exit(1)
         logger.info(f"Remaining unique features after dropping all-NaN columns: {len(all_feature_cols)}")

    # Ensure X_full and y_full are aligned by index (important if any rows were dropped from data_full after y_full was defined)
    common_index = data_full.index.intersection(X_full.index).intersection(y_full.index).intersection(groups_full.index)
    X_full = X_full.loc[common_index]
    y_full = y_full.loc[common_index]
    groups_full = groups_full.loc[common_index]
    
    # Re-check target distribution after any index alignment
    if len(y_full) < len(data_full):
        logger.info(f"Data aligned. y_full length changed. New target variable '{target_col_name}' distribution (N={len(y_full)}):")
        logger.info(f"\n-- Normalized --\n{y_full.value_counts(normalize=True).sort_index().round(3)}")
        logger.info(f"\n-- Counts --\n{y_full.value_counts(normalize=False).sort_index()}")
        n_classes = y_full.nunique()
        if n_classes < 2:
            logger.error("Target variable has only one class after final alignment. Cannot perform binary classification.")
            sys.exit(1)

    # --- 7. Final Checks ---
    min_patients_for_cv = config.N_SPLITS_CV
    if len(X_full) < 20 or groups_full.nunique() < min_patients_for_cv * 2 : # Recommended patients = folds * 2 at minimum
        logger.error(f"Insufficient data after final preparation. Rows={len(X_full)}, Unique Patients={groups_full.nunique()}. "
                     f"Need more data or patients for reliable splitting/CV (minimum {min_patients_for_cv*2} patients recommended for {config.N_SPLITS_CV}-fold CV).")
        sys.exit(1)

    if len(X_full) != len(y_full) or len(X_full) != len(groups_full):
        logger.error(f"Final length mismatch: X_full ({len(X_full)}), y_full ({len(y_full)}), groups_full ({len(groups_full)}).")
        sys.exit(1)

    logger.info(f"Final data shape prepared for modeling: X={X_full.shape}, y={len(y_full)}, Groups={groups_full.nunique()}")
    logger.debug(f"Final 'all_feature_cols' ({len(all_feature_cols)}): {all_feature_cols[:5]}...")
    for task_name, feats in task_features_map.items():
        logger.debug(f"Final features for task '{task_name}' ({len(feats)}): {feats[:3]}...")


    return X_full, y_full, groups_full, task_features_map, all_feature_cols