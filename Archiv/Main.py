import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import preprocessing
import plotting

# -------------------------------
# Setup and Preprocessing
# -------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(script_dir, 'Input')

# Run the preprocessing pipeline.
# The new preprocessing returns additional long-format DataFrames.
data = preprocessing.preprocess(input_folder)
merged_df = data['merged_df']
all_imaging_cols = data['all_imaging_cols']
kinematic_cols = data['kinematic_cols']
imaging_columns_old = data['imaging_columns_old']
imaging_columns_new = data['imaging_columns_new']
imaging_columns_z = data['imaging_columns_z']
imaging_long = data['imaging_long']
kinematic_long = data['kinematic_long']

print(f"Merged data shape: {merged_df.shape}")
print(f"Imaging long data shape: {imaging_long.shape}")
print(f"Kinematic long data shape: {kinematic_long.shape}")

# -------------------------------
# Create Output Folders
# -------------------------------
outlook_dir = os.path.join(script_dir, "outlook")
outlook_data_dir = os.path.join(outlook_dir, "data")
outlook_plots_dir = os.path.join(outlook_dir, "plots")
os.makedirs(outlook_data_dir, exist_ok=True)
os.makedirs(outlook_plots_dir, exist_ok=True)

# -------------------------------
# Plotting via plotting.py module (Original plots)
# -------------------------------
plotting.plot_patients_kinematic(merged_df, kinematic_cols, outlook_plots_dir)
plotting.plot_correlation_heatmap(merged_df, all_imaging_cols, kinematic_cols, outlook_plots_dir)
plotting.plot_hand_condition(merged_df, outlook_plots_dir)

# -------------------------------
# Statistical Analysis: Contralateral Correlations (Using Long-Format Data)
# -------------------------------
print("\nStarting contralateral correlation analysis (long-format)...")

# Check that kinematic_long has the required "Patient ID" column.
if "Patient ID" not in kinematic_long.columns or kinematic_long.empty:
    raise ValueError("Kinematic long DataFrame is empty or missing 'Patient ID'. "
                     "Please ensure that kinematic columns are correctly labeled (include 'Left' or 'Right').")

# Merge imaging_long and kinematic_long on 'Patient ID'.
# imaging_long has columns: Patient ID, Anatomical Region, Laterality, Modality, Imaging Value.
# kinematic_long has columns: Patient ID, Kinematic Variable, Laterality, Kinematic Value.
merged_long = pd.merge(imaging_long, kinematic_long, on="Patient ID", suffixes=("_img", "_kin"))

# Filter for contralateral pairs (i.e. imaging laterality differs from kinematic laterality).
contralateral_long = merged_long[merged_long["Laterality_img"] != merged_long["Laterality_kin"]].copy()
print("Debug: Contralateral merged long data shape:", contralateral_long.shape)

# Group by Anatomical Region, Imaging Modality, and Kinematic Variable.
grouped = contralateral_long.groupby(["Anatomical Region", "Modality", "Kinematic Variable"])
contralateral_results = []
for (region, modality, kin_var), group in grouped:
    # Only compute correlation if there is sufficient variation and sample size.
    if len(group) > 2 and group["Imaging Value"].nunique() > 1 and group["Kinematic Value"].nunique() > 1:
        corr, pval = stats.pearsonr(group["Imaging Value"], group["Kinematic Value"])
        contralateral_results.append({
            "Anatomical Region": region,
            "Modality": modality,
            "Kinematic Variable": kin_var,
            "Pearson r": corr,
            "p-value": pval,
            "N": len(group)
        })

contralateral_df = pd.DataFrame(contralateral_results)

# -------------------------------
# Significance Testing Without Multiple Comparisons Correction
# -------------------------------
user_alpha = 0.01  # Set your desired significance level.
if not contralateral_df.empty:
    contralateral_df['p-value_corrected'] = contralateral_df['p-value']  # Using raw p-values.
    contralateral_df['reject_null'] = contralateral_df['p-value'] < user_alpha

    print(f"\nContralateral Correlation Results (using significance level {user_alpha}):")
    print(contralateral_df.sort_values(by='p-value').to_string(index=False))
else:
    print("\nNo valid contralateral correlations were computed.")

# Save overall contralateral results.
contralateral_csv_path = os.path.join(outlook_data_dir, "contralateral_imaging_kinematics_correlations.csv")
contralateral_df.to_csv(contralateral_csv_path, index=False)
print(f"Contralateral correlation results saved to: {contralateral_csv_path}")

merged_csv_path = os.path.join(outlook_data_dir, "merged_analysis_results.csv")
merged_df.to_csv(merged_csv_path, index=False)
print(f"Merged analysis results saved to: {merged_csv_path}")

# -------------------------------
# Generate Scatter Plots for Significant Correlations
# -------------------------------
if not contralateral_df.empty:
    sig_pairs = contralateral_df[contralateral_df['p-value_corrected'] < user_alpha]
    # Update plotting function to work with long-format data.
    plotting.plot_scatter_plots(sig_pairs, contralateral_long, outlook_plots_dir)
else:
    print("No contralateral correlation tests were performed.")

# ---------------------------------------------------------
# Analysis: Significant Pairs in Both Imaging Modalities (Old and New)
# ---------------------------------------------------------
paired_significant_list = []
for (region, kin_var), group in contralateral_df.groupby(["Anatomical Region", "Kinematic Variable"]):
    modalities = set(group["Modality"])
    if "old" in modalities and "new" in modalities:
        old_signif = group[group["Modality"] == "old"]["reject_null"].all()
        new_signif = group[group["Modality"] == "new"]["reject_null"].all()
        if old_signif and new_signif:
            combined_entry = {
                "Anatomical Region": region,
                "Kinematic Variable": kin_var,
                "Pearson r (old)": group[group["Modality"]=="old"]["Pearson r"].mean(),
                "p-value (old)": group[group["Modality"]=="old"]["p-value"].mean(),
                "N (old)": group[group["Modality"]=="old"]["N"].mean(),
                "Pearson r (new)": group[group["Modality"]=="new"]["Pearson r"].mean(),
                "p-value (new)": group[group["Modality"]=="new"]["p-value"].mean(),
                "N (new)": group[group["Modality"]=="new"]["N"].mean()
            }
            paired_significant_list.append(combined_entry)

if paired_significant_list:
    significant_pairs_df = pd.DataFrame(paired_significant_list)
    paired_results_csv_path = os.path.join(outlook_data_dir, "paired_contralateral_significant_pairs.csv")
    significant_pairs_df.to_csv(paired_results_csv_path, index=False)
    print(f"Paired significant results saved to: {paired_results_csv_path}")
else:
    print("No paired significant correlations found between old and new modalities.")

# ---------------------------------------------------------
# Create Paired Scatter Plots for Significant Pairs
# ---------------------------------------------------------
if 'significant_pairs_df' in locals() and not significant_pairs_df.empty:
    plotting.plot_paired_scatter_plots(significant_pairs_df, contralateral_long, outlook_plots_dir)
else:
    print("No paired significant scatter plots to generate.")
