#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression Comparison Script with Automated Predictor Evaluation

This script reads in the merged_summary.csv file and, for each kinematic outcome
(for both Fingertapping and Hand Movements tasks) and for each imaging region
(e.g., Striatum, Putamen, Caudate), it fits two OLS regression models:
    1. Outcome ~ Contralateral_{Region}_new   (raw imaging measure)
    2. Outcome ~ Contralateral_{Region}_Z     (z-score imaging measure)
It then computes model metrics (R², AIC, BIC, RMSE) for each model and saves
the results. In addition, an extra analysis automatically compares the two models
for each combination by counting which predictor “wins” more metrics.
The final summaries are saved as CSV files.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from math import sqrt

# -------------------------
# 1. Load the Data
# -------------------------
# Determine the script directory and define the input file path.
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, "Input", "merged_summary.csv")
df = pd.read_csv(input_file)

# -------------------------
# 2. Define Variables for Analysis
# -------------------------
# Define tasks (e.g., Fingertapping "ft" and Hand Movements "hm").
tasks = ['ft', 'hm']

# Define the base names of your kinematic measures (without task prefix).
base_kinematic_cols = [
    "meanamplitude", "stdamplitude", "meanspeed", "stdspeed", "meanrmsvelocity",
    "stdrmsvelocity", "meanopeningspeed", "stdopeningspeed", "meanclosingspeed",
    "stdclosingspeed", "meancycleduration", "stdcycleduration", "rangecycleduration",
    "rate", "amplitudedecay", "velocitydecay", "ratedecay", "cvamplitude",
    "cvcycleduration", "cvspeed", "cvrmsvelocity", "cvopeningspeed", "cvclosingspeed"
]

# Define the imaging regions.
imaging_regions = ["Striatum", "Putamen", "Caudate"]

# -------------------------
# 3. Regression Analysis: Raw vs. Z-score Imaging Predictors
# -------------------------
results_list = []

for task in tasks:
    for base_kin in base_kinematic_cols:
        # Construct the full kinematic column name (e.g., "ft_meanamplitude").
        kinematic_col = f"{task}_{base_kin}"
        if kinematic_col not in df.columns:
            print(f"Column {kinematic_col} not found, skipping.")
            continue

        # Convert the outcome column to numeric.
        df[kinematic_col] = pd.to_numeric(
            df[kinematic_col].astype(str).str.strip().str.replace(',', '.'),
            errors='coerce'
        )

        for region in imaging_regions:
            # Define the imaging predictor column names.
            raw_col = f"Contralateral_{region}_new"   # raw imaging measure
            z_col   = f"Contralateral_{region}_Z"       # z-score imaging measure

            if raw_col not in df.columns or z_col not in df.columns:
                print(f"Columns {raw_col} or {z_col} not found for region {region} (kinematic {kinematic_col}). Skipping.")
                continue

            # Convert imaging predictors to numeric.
            df[raw_col] = pd.to_numeric(
                df[raw_col].astype(str).str.strip().str.replace(',', '.'),
                errors='coerce'
            )
            df[z_col] = pd.to_numeric(
                df[z_col].astype(str).str.strip().str.replace(',', '.'),
                errors='coerce'
            )

            # Prepare data for the raw predictor.
            data_raw = df[[kinematic_col, raw_col]].dropna()
            # Prepare data for the z-score predictor.
            data_z = df[[kinematic_col, z_col]].dropna()

            # Ensure there are enough data points.
            if len(data_raw) < 3 or len(data_z) < 3:
                print(f"Not enough data for {kinematic_col} with region {region}. Skipping.")
                continue

            # --- Model 1: Using the raw imaging predictor ---
            X_raw = sm.add_constant(data_raw[raw_col])
            y_raw = data_raw[kinematic_col]
            model_raw = sm.OLS(y_raw, X_raw).fit()
            rmse_raw = sqrt(np.mean(model_raw.resid**2))

            # --- Model 2: Using the z-score imaging predictor ---
            X_z = sm.add_constant(data_z[z_col])
            y_z = data_z[kinematic_col]
            model_z = sm.OLS(y_z, X_z).fit()
            rmse_z = sqrt(np.mean(model_z.resid**2))

            # Append the results to the list.
            results_list.append({
                "Task": task,
                "Kinematic Variable": kinematic_col,
                "Imaging Region": region,
                "Predictor": "raw",
                "R2": model_raw.rsquared,
                "AIC": model_raw.aic,
                "BIC": model_raw.bic,
                "RMSE": rmse_raw,
                "N": len(data_raw)
            })
            results_list.append({
                "Task": task,
                "Kinematic Variable": kinematic_col,
                "Imaging Region": region,
                "Predictor": "zscore",
                "R2": model_z.rsquared,
                "AIC": model_z.aic,
                "BIC": model_z.bic,
                "RMSE": rmse_z,
                "N": len(data_z)
            })

# Convert results to a DataFrame.
results_df = pd.DataFrame(results_list)
results_df.sort_values(["Task", "Kinematic Variable", "Imaging Region", "Predictor"], inplace=True)

# -------------------------
# 4. Save the Regression Comparison Results
# -------------------------
# Create an Output folder one level above the script.
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
output_folder = os.path.join(parent_dir, "Output")
os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(output_folder, "regression_comparison_results.csv")
results_df.to_csv(output_file, index=False)
print(f"Regression comparison results saved to {output_file}")

# -------------------------
# 5. Automated Predictor Comparison Analysis
# -------------------------
def compare_models(group):
    """
    For a given group (a specific Task, Kinematic Variable, and Imaging Region),
    compare the raw and zscore predictors by evaluating four metrics:
    - R2 (higher is better)
    - AIC (lower is better)
    - BIC (lower is better)
    - RMSE (lower is better)
    The function counts how many metrics favor zscore vs. raw and returns a summary.
    """
    raw_row = group[group['Predictor'] == 'raw']
    z_row = group[group['Predictor'] == 'zscore']
    if raw_row.empty or z_row.empty:
        return pd.Series({"Better Predictor": "Incomplete data"})
    
    # Extract metric values.
    raw_r2 = raw_row['R2'].values[0]
    z_r2 = z_row['R2'].values[0]
    raw_aic = raw_row['AIC'].values[0]
    z_aic = z_row['AIC'].values[0]
    raw_bic = raw_row['BIC'].values[0]
    z_bic = z_row['BIC'].values[0]
    raw_rmse = raw_row['RMSE'].values[0]
    z_rmse = z_row['RMSE'].values[0]
    
    # Count wins.
    wins_z = 0
    wins_raw = 0
    
    # R2: higher is better.
    if z_r2 > raw_r2:
        wins_z += 1
    elif raw_r2 > z_r2:
        wins_raw += 1
        
    # AIC: lower is better.
    if z_aic < raw_aic:
        wins_z += 1
    elif raw_aic < z_aic:
        wins_raw += 1
        
    # BIC: lower is better.
    if z_bic < raw_bic:
        wins_z += 1
    elif raw_bic < z_bic:
        wins_raw += 1
        
    # RMSE: lower is better.
    if z_rmse < raw_rmse:
        wins_z += 1
    elif raw_rmse < z_rmse:
        wins_raw += 1
        
    if wins_z > wins_raw:
        better = "zscore"
    elif wins_raw > wins_z:
        better = "raw"
    else:
        better = "tie"
        
    return pd.Series({
        "Better Predictor": better,
        "Wins (zscore)": wins_z,
        "Wins (raw)": wins_raw,
        "Raw R2": raw_r2,
        "Zscore R2": z_r2,
        "Raw AIC": raw_aic,
        "Zscore AIC": z_aic,
        "Raw BIC": raw_bic,
        "Zscore BIC": z_bic,
        "Raw RMSE": raw_rmse,
        "Zscore RMSE": z_rmse,
        "N": raw_row['N'].values[0]  # assume same N for both
    })

# Group by Task, Kinematic Variable, and Imaging Region.
summary_comparison = results_df.groupby(["Task", "Kinematic Variable", "Imaging Region"]).apply(compare_models).reset_index()

# Save the automated comparison summary.
comparison_output_file = os.path.join(output_folder, "predictor_comparison_summary.csv")
summary_comparison.to_csv(comparison_output_file, index=False)
print(f"Predictor comparison summary saved to {comparison_output_file}")

# Optionally, print a sample of the summary.
print(summary_comparison.head(10))
