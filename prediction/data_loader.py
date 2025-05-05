#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 16:07:12 2025

@author: Lange_L
"""

# -*- coding: utf-8 -*-
"""
Functions for loading and preparing the data.
"""

import pandas as pd
import numpy as np
import os
import sys
from . import config # Use relative import

def load_data(input_folder, csv_name):
    """Loads the specified CSV file, trying different separators."""
    input_file_path = os.path.join(input_folder, csv_name)
    print(f"Attempting to load data from: {input_file_path}")
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at {input_file_path}")
        sys.exit(1)

    df = None
    try:
        # Try semicolon first, then comma
        try:
            df = pd.read_csv(input_file_path, sep=';', decimal='.')
            print("Successfully loaded data using ';' separator.")
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError, KeyError): # Added KeyError
             print("Failed read with ';', trying ',' separator...")
             df = pd.read_csv(input_file_path, sep=',', decimal='.')
             print("Successfully loaded data using ',' separator.")
        except Exception as e: # Catch any other potential read errors
             print(f"An unexpected error occurred during CSV reading: {e}")
             sys.exit(1)

        print(f"Data loaded successfully. Initial shape: {df.shape}")
        return df

    except Exception as e:
        print(f"Error loading or parsing data from {input_file_path}: {e}")
        sys.exit(1)


def prepare_data(df):
    """
    Prepares the dataframe: handles target, groups, features, and basic cleaning.
    Returns X_full, y_full, groups_full, task_features_map, all_feature_cols
    """
    print("Preparing data...")
    data_full = df.copy()

    # 1. Prepare Group ID
    if config.GROUP_ID_COL not in data_full.columns:
        print(f"Error: Group ID column '{config.GROUP_ID_COL}' not found in data.")
        sys.exit(1)
    data_full[config.GROUP_ID_COL] = data_full[config.GROUP_ID_COL].astype(str).str.strip()
    initial_rows = len(data_full)
    data_full.dropna(subset=[config.GROUP_ID_COL], inplace=True)
    if len(data_full) < initial_rows:
        print(f"Dropped {initial_rows - len(data_full)} rows with missing group ID ('{config.GROUP_ID_COL}').")

    # 2. Prepare Target Variable
    if config.TARGET_Z_SCORE_COL not in data_full.columns:
        print(f"Error: Target Z-score column '{config.TARGET_Z_SCORE_COL}' not found.")
        sys.exit(1)
    # Convert target Z-score, handling potential non-numeric values robustly
    data_full[config.TARGET_Z_SCORE_COL] = pd.to_numeric(data_full[config.TARGET_Z_SCORE_COL].astype(str).str.replace(',', '.'), errors='coerce')
    initial_rows = len(data_full)
    data_full.dropna(subset=[config.TARGET_Z_SCORE_COL], inplace=True)
    if len(data_full) < initial_rows:
        print(f"Dropped {initial_rows - len(data_full)} rows with missing target Z-score ('{config.TARGET_Z_SCORE_COL}').")

    if data_full.empty:
        print("Error: No data remaining after handling missing Group IDs or Target Z-scores.")
        sys.exit(1)

    # Create binary target column
    data_full[config.TARGET_COLUMN_NAME] = (data_full[config.TARGET_Z_SCORE_COL] <= config.ABNORMALITY_THRESHOLD).astype(int)
    y_full = data_full[config.TARGET_COLUMN_NAME]
    groups_full = data_full[config.GROUP_ID_COL] # Patient IDs for splitting

    print(f"Target variable '{config.TARGET_COLUMN_NAME}' distribution:")
    print(y_full.value_counts(normalize=True).round(3))
    print(f"Number of unique patients: {groups_full.nunique()}")
    if len(y_full.unique()) < 2:
        print("Error: Target variable has only one class after preparation. Cannot perform classification.")
        sys.exit(1)

    # 3. Identify and Prepare Features
    all_feature_cols = []
    task_features_map = {}
    print("\nIdentifying features for tasks:")
    for task_prefix in config.TASKS_TO_RUN_SEPARATELY:
        task_cols = []
        for base_col in config.BASE_KINEMATIC_COLS:
            col_name = f"{task_prefix}_{base_col}"
            if col_name in data_full.columns:
                task_cols.append(col_name)
        # Check if any features were found for the task
        if task_cols:
            print(f"  - Task '{task_prefix}': Found {len(task_cols)} features.")
            task_features_map[task_prefix] = task_cols
            all_feature_cols.extend(task_cols)
        else:
            print(f"  - Task '{task_prefix}': Warning - No features found matching BASE_KINEMATIC_COLS.")

    # Ensure we only keep unique feature columns
    all_feature_cols = sorted(list(set(all_feature_cols)))
    if not all_feature_cols:
        print("Error: No kinematic features found across all specified tasks. Check BASE_KINEMATIC_COLS and task prefixes.")
        sys.exit(1)
    print(f"Total unique kinematic features identified: {len(all_feature_cols)}")

    X_full = data_full[all_feature_cols].copy()
    # Convert all feature columns to numeric, coercing errors
    for col in all_feature_cols:
        X_full[col] = pd.to_numeric(X_full[col].astype(str).str.replace(',', '.'), errors='coerce')

    # Note: We do NOT drop NaNs here. Imputation is handled within the pipeline.
    # Check for columns that are *entirely* NaN after conversion, as these will cause problems.
    cols_all_nan = X_full.columns[X_full.isnull().all()].tolist()
    if cols_all_nan:
         print(f"Warning: The following feature columns contain only NaN values and will be dropped: {cols_all_nan}")
         X_full.drop(columns=cols_all_nan, inplace=True)
         all_feature_cols = X_full.columns.tolist() # Update the list of features
         # Update task_features_map as well
         for task in task_features_map:
             task_features_map[task] = [f for f in task_features_map[task] if f in all_feature_cols]
         print(f"Remaining unique features after dropping all-NaN columns: {len(all_feature_cols)}")
         if not all_feature_cols:
              print("Error: No valid feature columns remaining after dropping all-NaN columns.")
              sys.exit(1)


    # Final check on data size viability
    min_patients_for_cv = config.N_SPLITS_CV # Need at least N_SPLITS_CV unique patients for StratifiedGroupKFold
    if len(X_full) < 20 or groups_full.nunique() < min_patients_for_cv * 2 : # Need enough patients for train/test *and* inner CV
        print(f"Error: Insufficient data after preparation. Rows={len(X_full)}, Unique Patients={groups_full.nunique()}. "
              f"Need more data or patients for reliable splitting/CV (minimum {min_patients_for_cv*2} patients recommended for {config.N_SPLITS_CV}-fold CV and test split). Exiting.")
        sys.exit(1)

    print(f"Final data shape prepared for modeling: X={X_full.shape}, y={len(y_full)}, Groups={groups_full.nunique()}")

    return X_full, y_full, groups_full, task_features_map, all_feature_cols