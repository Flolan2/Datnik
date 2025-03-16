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

# Determine the script's directory.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define the merged CSV file location relative to the script.
merged_csv_file = os.path.join(script_dir, "Input", "merged_summary.csv")
df = pd.read_csv(merged_csv_file)

# Base kinematic variable names (without task prefix)
base_kinematic_cols = [
    "meanamplitude", "stdamplitude", "meanspeed", "stdspeed", "meanrmsvelocity",
    "stdrmsvelocity", "meanopeningspeed", "stdopeningspeed", "meanclosingspeed",
    "stdclosingspeed", "meancycleduration", "stdcycleduration", "rangecycleduration",
    "rate", "amplitudedecay", "velocitydecay", "ratedecay", "cvamplitude",
    "cvcycleduration", "cvspeed", "cvrmsvelocity", "cvopeningspeed", "cvclosingspeed"
]

# Imaging columns (from DatScan) should start with "Contralateral_"
dat_scan_cols = [col for col in df.columns if col.startswith("Contralateral_")]

# Define the two tasks with their respective prefixes.
tasks = ['ft', 'hm']

# Helper function to extract anatomical region and measure type.
def extract_region_measure(varname):
    # Expected format: "Contralateral_{Region}_{Measure}"
    try:
        _, region, measure = varname.split('_', 2)
    except ValueError:
        region, measure = None, None
    # Optionally, rename "new" to "raw" for clarity.
    if measure == "new":
        measure = "raw"
    return region, measure

# Define the parent directory (one level above the script)
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
# Define an output folder one level above the script.
output_base_folder = os.path.join(parent_dir, "Output")

# -------------------------
# 2. Correlation Analysis & Pivoting
# -------------------------

# Loop over each kinematic task.
for task in tasks:
    print(f"\n=== Processing kinematic task: {task} ===")
    task_results = []  # will store correlation results for this task

    # Loop through each base kinematic column (build the full column name with prefix)
    for base_col in base_kinematic_cols:
        kinematic_col = f"{task}_{base_col}"
        if kinematic_col not in df.columns:
            print(f"Kinematic column '{kinematic_col}' not found for task {task}. Skipping.")
            continue

        for dat in dat_scan_cols:
            data_pair = df[[kinematic_col, dat]].copy()

            # Debug print: show first few values before conversion.
            print(f"\nBefore conversion - {kinematic_col}: {data_pair[kinematic_col].head().tolist()}")
            print(f"Before conversion - {dat}: {data_pair[dat].head().tolist()}")

            # Convert strings with commas to numeric values.
            data_pair[kinematic_col] = pd.to_numeric(
                data_pair[kinematic_col].astype(str).str.strip().str.replace(',', '.'),
                errors='coerce'
            )
            data_pair[dat] = pd.to_numeric(
                data_pair[dat].astype(str).str.strip().str.replace(',', '.'),
                errors='coerce'
            )
            print(f"After conversion - {kinematic_col}: {data_pair[kinematic_col].head().tolist()}")
            print(f"After conversion - {dat}: {data_pair[dat].head().tolist()}")

            # Drop missing values.
            data_pair = data_pair.dropna()
            if len(data_pair) < 3:
                print(f"Not enough data for correlation between '{kinematic_col}' and '{dat}'. Skipping.")
                continue

            # Perform Pearson correlation.
            corr_coef, p_value = pearsonr(data_pair[kinematic_col], data_pair[dat])
            task_results.append({
                "Task": task,
                "Kinematic Variable": kinematic_col,
                "Base Kinematic": base_col,
                "DatScan Variable": dat,
                "Pearson Correlation": corr_coef,
                "P-value": p_value,
                "N": len(data_pair)
            })

    # Convert task results to DataFrame.
    task_results_df = pd.DataFrame(task_results)

    # Add region and measure info from DatScan variable.
    task_results_df["Region"] = task_results_df["DatScan Variable"].apply(lambda x: extract_region_measure(x)[0])
    task_results_df["Measure"] = task_results_df["DatScan Variable"].apply(lambda x: extract_region_measure(x)[1])

    # Pivot so that each base kinematic variable is a single row.
    pivot_corr = task_results_df.pivot(index="Base Kinematic", columns=["Region", "Measure"], values="Pearson Correlation")
    pivot_p    = task_results_df.pivot(index="Base Kinematic", columns=["Region", "Measure"], values="P-value")
    pivot_r2   = task_results_df.pivot(index="Base Kinematic", columns=["Region", "Measure"], values="Pearson Correlation")
    pivot_r2 = pivot_r2.applymap(lambda r: r**2 if pd.notna(r) else r)

    final_df = pd.DataFrame(index=pivot_corr.index)
    for region in pivot_corr.columns.levels[0]:
        for measure in pivot_corr.columns.levels[1]:
            col_corr = f"{region}_{measure}_corr"
            col_p    = f"{region}_{measure}_p"
            col_r2   = f"{region}_{measure}_R2"
            final_df[col_corr] = pivot_corr.get((region, measure))
            final_df[col_p]    = pivot_p.get((region, measure))
            final_df[col_r2]   = pivot_r2.get((region, measure))

    # Optionally, compute a summary statistic (sum of p-values) and sort.
    final_df["p_sum"] = final_df.filter(like="_p").sum(axis=1)
    final_df.sort_values("p_sum", inplace=True)
    final_df.drop(columns=["p_sum"], inplace=True)

    # Save the pivoted correlation results for this task.
    data_output_folder = os.path.join(output_base_folder, "Data")
    os.makedirs(data_output_folder, exist_ok=True)
    output_file = os.path.join(data_output_folder, f"correlation_results_pivot_{task}.csv")
    final_df.to_csv(output_file)
    print(f"Pivoted correlation results for task {task} saved to {output_file}")

    # -------------------------
    # 3. Build Paired Significant Findings (Using Z-scores and Raw Values)
    # -------------------------
    significance_level = 0.000001
    paired_results = []
    regions_of_interest = ["Striatum", "Putamen", "Caudate"]

    for region in regions_of_interest:
        for base_col in base_kinematic_cols:
            # Process both Z-score and raw (new) measures.
            z_col = f"Contralateral_{region}_Z"
            new_col = f"Contralateral_{region}_new"
            z_result = task_results_df[
                (task_results_df["Base Kinematic"] == base_col) &
                (task_results_df["DatScan Variable"] == z_col)
            ]
            new_result = task_results_df[
                (task_results_df["Base Kinematic"] == base_col) &
                (task_results_df["DatScan Variable"] == new_col)
            ]
            z_result_sig = z_result[z_result["P-value"] < significance_level]
            new_result_sig = new_result[new_result["P-value"] < significance_level]

            if not z_result_sig.empty and not new_result_sig.empty:
                p_z = z_result_sig["P-value"].min()
                p_new = new_result_sig["P-value"].min()
                # Use the maximum p-value as a corrected measure.
                p_value_corrected = max(p_z, p_new)
                N = min(z_result_sig["N"].min(), new_result_sig["N"].min())
                paired_results.append({
                    "Task": task,
                    "Anatomical Region": region,
                    "Base Kinematic": base_col,
                    "p-value_corrected": p_value_corrected,
                    "N": N
                })

    paired_significant_df = pd.DataFrame(paired_results)
    print("\nPaired Significant Findings for task", task)
    print(paired_significant_df)

    # -------------------------
    # 4. Dual Scatter Plot for Kinematic vs. DatScan Regions
    #    (Using both Z-scores and Raw Values)
    # -------------------------
    try:
        from plotting import plot_dual_scatter
    except ImportError:
        print("plotting.py module not found. Please ensure it exists in the same directory.")
    else:
        for idx, row in paired_significant_df.iterrows():
            region = row["Anatomical Region"]
            base_col = row["Base Kinematic"]
            # Build the full kinematic column name for this task.
            kinematic_col = f"{task}_{base_col}"
            z_col = f"Contralateral_{region}_Z"
            new_col = f"Contralateral_{region}_new"

            # Grab the best (lowest p-value) result for each imaging modality.
            try:
                z_stats = task_results_df[
                    (task_results_df["Kinematic Variable"] == kinematic_col) &
                    (task_results_df["DatScan Variable"] == z_col)
                ].sort_values(by="P-value", ascending=True).iloc[0]
                new_stats = task_results_df[
                    (task_results_df["Kinematic Variable"] == kinematic_col) &
                    (task_results_df["DatScan Variable"] == new_col)
                ].sort_values(by="P-value", ascending=True).iloc[0]
            except IndexError:
                print(f"Insufficient data for dual scatter plot for task {task}, {region}, {base_col}.")
                continue

            r_z = z_stats["Pearson Correlation"]
            p_z = z_stats["P-value"]
            r2_z = r_z ** 2

            r_new = new_stats["Pearson Correlation"]
            p_new = new_stats["P-value"]
            r2_new = r_new ** 2

            # Prepare data for plotting.
            if kinematic_col in df.columns and z_col in df.columns and new_col in df.columns:
                plot_data = df[[kinematic_col, z_col, new_col]].copy()
                plot_data[kinematic_col] = pd.to_numeric(
                    plot_data[kinematic_col].astype(str).str.strip().str.replace(',', '.'),
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
                    print(f"Not enough data for dual scatter plot for {region} and {kinematic_col}. Skipping.")
                    continue

                # Save plots in an "Plots" subfolder inside the Output folder.
                plots_folder = os.path.join(output_base_folder, "Plots")
                os.makedirs(plots_folder, exist_ok=True)
                file_name = f"dual_scatter_{task}_{region}_{base_col}.png"
                print(f"Plotting dual scatter plot for task {task}, {region}, and {base_col}...")
                plot_dual_scatter(
                    data=plot_data,
                    kinematic_col=kinematic_col,
                    z_col=z_col,
                    new_col=new_col,
                    r_z=r_z,
                    p_z=p_z,
                    r2_z=r2_z,
                    r_new=r_new,
                    p_new=p_new,
                    r2_new=r2_new,
                    output_folder=os.path.join(output_base_folder, "Plots"),
                    file_name=file_name
                )
            else:
                print(f"Required columns for {region} and {kinematic_col} not found. Skipping.")


# End of script.
