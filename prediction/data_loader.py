# -*- coding: utf-8 -*-
"""
Functions for loading and preparing the data FOR BINARY Classification.
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

# Accepts config object passed from main script
def prepare_data(df, config):
    """
    Prepares the dataframe for BINARY classification: handles target, groups, features, cleaning.
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
    y_full = None
    target_col_name = config.TARGET_COLUMN_NAME
    threshold = config.ABNORMALITY_THRESHOLD
    logger.info(f"Defining binary target variable '{target_col_name}' using threshold Z={threshold}")
    data_full[target_col_name] = (data_full[config.TARGET_Z_SCORE_COL] <= threshold).astype(int)

    # Clean up (shouldn't have NaNs here unless Z-score was NaN, already handled)
    initial_rows = len(data_full)
    data_full.dropna(subset=[target_col_name], inplace=True) # Just in case
    if len(data_full) < initial_rows:
        logger.warning(f"Dropped {initial_rows - len(data_full)} rows with missing target label ('{target_col_name}') after assignment.")
    if data_full.empty:
        logger.error(f"No data remaining after assigning target variable '{target_col_name}'.")
        sys.exit(1)
    data_full[target_col_name] = data_full[target_col_name].astype(int)

    y_full = data_full[target_col_name]
    groups_full = data_full[config.GROUP_ID_COL]

    # --- Log target distribution ---
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

    # 4. Identify and Prepare Features
    all_feature_cols = []
    task_features_map = {}
    logger.info("Identifying features for tasks:")
    for task_prefix in config.TASKS_TO_RUN_SEPARATELY:
        task_cols = [f"{task_prefix}_{base}" for base in config.BASE_KINEMATIC_COLS if f"{task_prefix}_{base}" in data_full.columns]
        if task_cols:
            logger.info(f"  - Task '{task_prefix}': Found {len(task_cols)} features.")
            task_features_map[task_prefix] = task_cols
            all_feature_cols.extend(task_cols)
        else:
            logger.warning(f"  - Task '{task_prefix}': No features found matching BASE_KINEMATIC_COLS.")

    all_feature_cols = sorted(list(set(all_feature_cols)))
    if not all_feature_cols:
        logger.error("No kinematic features found across all specified tasks. Check BASE_KINEMATIC_COLS and task prefixes.")
        sys.exit(1)
    logger.info(f"Total unique kinematic features identified: {len(all_feature_cols)}")

    X_full = data_full[all_feature_cols].copy()
    for col in all_feature_cols:
        X_full[col] = pd.to_numeric(X_full[col].astype(str).str.replace(',', '.'), errors='coerce')

    cols_all_nan = X_full.columns[X_full.isnull().all()].tolist()
    if cols_all_nan:
         logger.warning(f"Dropping fully NaN feature columns: {cols_all_nan}")
         X_full.drop(columns=cols_all_nan, inplace=True)
         all_feature_cols = X_full.columns.tolist()
         for task in list(task_features_map.keys()):
             task_features_map[task] = [f for f in task_features_map[task] if f in all_feature_cols]
             if not task_features_map[task]:
                  logger.warning(f"Task '{task}' removed as it has no remaining valid features.")
                  del task_features_map[task]
         if not all_feature_cols:
              logger.error("No valid feature columns remaining after dropping all-NaN columns.")
              sys.exit(1)
         logger.info(f"Remaining unique features after dropping all-NaN columns: {len(all_feature_cols)}")

    X_full = X_full.loc[y_full.index] # Align after potential row drops

    # 5. Final Checks
    min_patients_for_cv = config.N_SPLITS_CV
    if len(X_full) < 20 or groups_full.nunique() < min_patients_for_cv * 2 :
        logger.error(f"Insufficient data after final preparation. Rows={len(X_full)}, Unique Patients={groups_full.nunique()}. "
                     f"Need more data or patients for reliable splitting/CV (minimum {min_patients_for_cv*2} patients recommended for {config.N_SPLITS_CV}-fold CV).")
        sys.exit(1)

    if len(X_full) != len(y_full):
        logger.error(f"Final length mismatch between X_full ({len(X_full)}) and y_full ({len(y_full)}).")
        sys.exit(1)

    logger.info(f"Final data shape prepared for modeling: X={X_full.shape}, y={len(y_full)}, Groups={groups_full.nunique()}")

    return X_full, y_full, groups_full, task_features_map, all_feature_cols