#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 15:08:22 2025

@author: Lange_L
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# -------------------------
# 1. Load and Process Data
# -------------------------

merged_csv_file = "Input/merged_summary.csv"
df = pd.read_csv(merged_csv_file)

kinematic_cols = [
    "meanamplitude", "stdamplitude", "meanspeed", "stdspeed", "meanrmsvelocity",
    "stdrmsvelocity", "meanopeningspeed", "stdopeningspeed", "meanclosingspeed",
    "stdclosingspeed", "meancycleduration", "stdcycleduration", "rangecycleduration",
    "rate", "amplitudedecay", "velocitydecay", "ratedecay", "cvamplitude",
    "cvcycleduration", "cvspeed", "cvrmsvelocity", "cvopeningspeed", "cvclosingspeed"
]

# Only imaging columns for new software and Z-scores are expected
dat_scan_cols = [col for col in df.columns if col.startswith("Contralateral_")]

# -------------------------
# 2. Correlation Analysis
# -------------------------

results = []
for kinematic in kinematic_cols:
    if kinematic not in df.columns:
        print(f"Kinematic column '{kinematic}' not found. Skipping.")
        continue
    for dat in dat_scan_cols:
        data_pair = df[[kinematic, dat]].copy()
        print(f"Before conversion - {kinematic}: {data_pair[kinematic].head().tolist()}")
        print(f"Before conversion - {dat}: {data_pair[dat].head().tolist()}")
        data_pair[kinematic] = pd.to_numeric(
            data_pair[kinematic].astype(str).str.strip().str.replace(',', '.'),
            errors='coerce'
        )
        data_pair[dat] = pd.to_numeric(
            data_pair[dat].astype(str).str.strip().str.replace(',', '.'),
            errors='coerce'
        )
        print(f"After conversion - {kinematic}: {data_pair[kinematic].head().tolist()}")
        print(f"After conversion - {dat}: {data_pair[dat].head().tolist()}")
        data_pair = data_pair.dropna()
        if len(data_pair) < 3:
            print(f"Not enough data for correlation between '{kinematic}' and '{dat}'. Skipping.")
            continue
        corr_coef, p_value = pearsonr(data_pair[kinematic], data_pair[dat])
        results.append({
            "Kinematic Variable": kinematic,
            "DatScan Variable": dat,
            "Pearson Correlation": corr_coef,
            "P-value": p_value,
            "N": len(data_pair)
        })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="P-value", ascending=True)

data_output_folder = os.path.join("Output", "Data")
os.makedirs(data_output_folder, exist_ok=True)
output_file = os.path.join(data_output_folder, "correlation_results.csv")
results_df.to_csv(output_file, index=False)
print(f"Correlation results saved to {output_file}")

# -------------------------
# 3. Build Paired Significant Findings (Using New Software Values and Z-scores)
# -------------------------

significance_level = 0.001
paired_results = []
regions = ["Striatum", "Putamen", "Caudate"]

for region in regions:
    for kin in kinematic_cols:
        # Use Z-score and new measure columns only
        z_col = f"Contralateral_{region}_Z"
        new_col = f"Contralateral_{region}_new"
        z_result = results_df[
            (results_df["Kinematic Variable"] == kin) &
            (results_df["DatScan Variable"] == z_col)
        ]
        new_result = results_df[
            (results_df["Kinematic Variable"] == kin) &
            (results_df["DatScan Variable"] == new_col)
        ]
        # Filter only significant results
        z_result_sig = z_result[z_result["P-value"] < significance_level]
        new_result_sig = new_result[new_result["P-value"] < significance_level]
        
        if not z_result_sig.empty and not new_result_sig.empty:
            # Choose the most significant (lowest p-value) for each modality
            p_z = z_result_sig["P-value"].min()
            p_new = new_result_sig["P-value"].min()
            p_value_corrected = max(p_z, p_new)
            # Use the minimum sample size among the significant results
            N = min(z_result_sig["N"].min(), new_result_sig["N"].min())
            paired_results.append({
                "Anatomical Region": region,
                "Kinematic Variable": kin,
                "p-value_corrected": p_value_corrected,
                "N": N
            })

paired_significant_df = pd.DataFrame(paired_results)
print("Paired Significant Findings:")
print(paired_significant_df)

# -------------------------
# 4. Dual Scatter Plot for Kinematic vs. DatScan Regions (Using Z-scores and New Measures)
# -------------------------

try:
    from plotting import plot_dual_scatter
except ImportError:
    print("plotting.py module not found. Please ensure it exists in the same directory.")
else:
    # For each paired significant finding, create a dual scatter plot
    for idx, row in paired_significant_df.iterrows():
        region = row["Anatomical Region"]
        kin = row["Kinematic Variable"]
        # Use the Z-score and new measure columns
        z_col = f"Contralateral_{region}_Z"
        new_col = f"Contralateral_{region}_new"
        
        # Grab the best row in results_df for the Z-score column
        z_stats = results_df[
            (results_df["Kinematic Variable"] == kin) &
            (results_df["DatScan Variable"] == z_col)
        ].sort_values(by="P-value", ascending=True).iloc[0]
        
        # Grab the best row in results_df for the new measure column
        new_stats = results_df[
            (results_df["Kinematic Variable"] == kin) &
            (results_df["DatScan Variable"] == new_col)
        ].sort_values(by="P-value", ascending=True).iloc[0]
        
        # Extract the correlation, p-value, and compute R² for both modalities
        r_z = z_stats["Pearson Correlation"]
        p_z = z_stats["P-value"]
        r2_z = r_z**2
        
        r_new = new_stats["Pearson Correlation"]
        p_new = new_stats["P-value"]
        r2_new = r_new**2
        
        # Build a DataFrame containing the kinematic variable and both imaging modalities
        if kin in df.columns and z_col in df.columns and new_col in df.columns:
            plot_data = df[[kin, z_col, new_col]].copy()
            plot_data[kin] = pd.to_numeric(
                plot_data[kin].astype(str).str.strip().str.replace(',', '.'),
                errors='coerce'
            )
            plot_data[z_col] = pd.to_numeric(
                plot_data[z_col].astype(str).str.strip().str.replace(',', '.'),
                errors='coerce'
            )
            plot_data[new_col] = pd.to_numeric(
                plot_data[new_col].astype(str).str.strip().str.replace(',', '.'),
                errors='coerce'
            )
            plot_data.dropna(inplace=True)
            
            if plot_data.shape[0] < 3:
                print(f"Not enough data for dual scatter plot for {region} and {kin}. Skipping.")
                continue
            
            file_name = f"dual_scatter_{region}_{kin}.png"
            print(f"Plotting dual scatter plot for {region} and {kin}...")
            
            # Pass the updated correlation stats and data to the plotting function
            plot_dual_scatter(
                data=plot_data,
                kinematic_col=kin,
                z_col=z_col,
                new_col=new_col,
                r_z=r_z,
                p_z=p_z,
                r2_z=r2_z,
                r_new=r_new,
                p_new=p_new,
                r2_new=r2_new,
                file_name=file_name
            )
        else:
            print(f"Required columns for {region} and {kin} not found. Skipping.")
