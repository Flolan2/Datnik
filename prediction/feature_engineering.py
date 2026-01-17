# -*- coding: utf-8 -*-
"""
Functions for creating new engineered features.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger('DatnikExperiment')

def create_ratios(X: pd.DataFrame, num_col: str, den_col: str, new_col_name: str) -> pd.DataFrame:
    new_features_df = pd.DataFrame(index=X.index)
    if num_col not in X.columns or den_col not in X.columns:
        missing_cols = []
        if num_col not in X.columns: missing_cols.append(num_col)
        if den_col not in X.columns: missing_cols.append(den_col)
        logger.warning(f"Ratio '{new_col_name}': Missing input columns: {', '.join(missing_cols)}. New column will be all NaN.")
        new_features_df[new_col_name] = np.nan
        return new_features_df
    numerator = pd.to_numeric(X[num_col], errors='coerce')
    denominator = pd.to_numeric(X[den_col], errors='coerce')
    epsilon = 1e-9
    valid_mask = numerator.notna() & denominator.notna() & (np.abs(denominator) > epsilon)
    new_features_df[new_col_name] = np.nan
    new_features_df.loc[valid_mask, new_col_name] = numerator[valid_mask] / denominator[valid_mask]
    num_output_nans = new_features_df[new_col_name].isnull().sum()
    if num_output_nans > 0:
        logger.debug(f"Ratio '{new_col_name}': {num_output_nans}/{len(X)} values are NaN after calculation.")
    return new_features_df

def create_interaction_terms(X: pd.DataFrame, col1: str, col2: str, new_col_name: str) -> pd.DataFrame:
    # ... (this function should remain as it was) ...
    new_features_df = pd.DataFrame(index=X.index)
    if col1 not in X.columns or col2 not in X.columns:
        missing_cols = []
        if col1 not in X.columns: missing_cols.append(col1)
        if col2 not in X.columns: missing_cols.append(col2)
        logger.warning(f"Interaction '{new_col_name}': Missing input columns: {', '.join(missing_cols)}. New column will be all NaN.")
        new_features_df[new_col_name] = np.nan
        return new_features_df
    feat1 = pd.to_numeric(X[col1], errors='coerce')
    feat2 = pd.to_numeric(X[col2], errors='coerce')
    valid_mask = feat1.notna() & feat2.notna()
    new_features_df[new_col_name] = np.nan
    new_features_df.loc[valid_mask, new_col_name] = feat1[valid_mask] * feat2[valid_mask]
    num_output_nans = new_features_df[new_col_name].isnull().sum()
    if num_output_nans > 0:
        logger.debug(f"Interaction '{new_col_name}': {num_output_nans}/{len(X)} values are NaN after calculation.")
    return new_features_df

def create_polynomial_features(X: pd.DataFrame, col: str, degree: int, new_col_prefix: str) -> pd.DataFrame:
    # ... (this function should remain as it was, ensuring it returns only new cols) ...
    new_features_df = pd.DataFrame(index=X.index)
    created_cols = []
    if col not in X.columns:
        logger.warning(f"Polynomial for '{col}': Input column not found. No polynomial features created.")
        return new_features_df
    feature_series = pd.to_numeric(X[col], errors='coerce')
    if degree < 2:
        logger.warning(f"Polynomial for '{col}': Degree must be >= 2. No polynomial features created.")
        return new_features_df
    for d in range(2, degree + 1): # Only create for degree 2 if degree=2 is passed
        poly_col_name = f"{new_col_prefix}_deg{d}"
        created_cols.append(poly_col_name)
        valid_mask = feature_series.notna()
        new_features_df[poly_col_name] = np.nan
        new_features_df.loc[valid_mask, poly_col_name] = feature_series[valid_mask] ** d
        num_output_nans = new_features_df[poly_col_name].isnull().sum()
        if num_output_nans > 0:
            logger.debug(f"Polynomial feature '{poly_col_name}': {num_output_nans}/{len(X)} values are NaN.")
    return new_features_df[created_cols] if created_cols else pd.DataFrame(index=X.index)


def create_log_transform(X: pd.DataFrame, col: str, new_col_name: str) -> pd.DataFrame:
    """
    Creates a log1p transformed feature (log(1+x)).
    Handles non-numeric values by converting to NaN.
    Warns if column contains negative values, as log1p is undefined for x < -1.

    Args:
        X (pd.DataFrame): The input DataFrame.
        col (str): Name of the column to transform.
        new_col_name (str): Name for the new log-transformed feature column.

    Returns:
        pd.DataFrame: DataFrame containing only the new log-transformed feature.
    """
    new_features_df = pd.DataFrame(index=X.index)

    if col not in X.columns:
        logger.warning(f"Log transform for '{col}': Column not found. New column '{new_col_name}' will be all NaN.")
        new_features_df[new_col_name] = np.nan
        return new_features_df

    feature_series = pd.to_numeric(X[col], errors='coerce')
    
    problematic_neg_mask = feature_series < -1
    if problematic_neg_mask.any():
        logger.warning(f"Log transform for '{col}' ('{new_col_name}'): Column contains values < -1. log1p will result in NaN for these entries.")

    new_features_df[new_col_name] = np.log1p(feature_series) # np.log1p handles NaNs in input by returning NaN

    num_input_nans = feature_series.isnull().sum()
    num_output_nans = new_features_df[new_col_name].isnull().sum()

    if num_output_nans > num_input_nans:
        logger.debug(f"Log transform '{new_col_name}': {num_output_nans - num_input_nans} additional NaNs created (likely due to values <= -1 or issues with log). Total NaNs: {num_output_nans}/{len(X)}.")
    elif num_output_nans > 0 :
        logger.debug(f"Log transform '{new_col_name}': Contains {num_output_nans}/{len(X)} NaNs (likely from original data).")
        
    return new_features_df[[new_col_name]]