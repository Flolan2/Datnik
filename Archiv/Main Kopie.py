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
data = preprocessing.preprocess(input_folder)
merged_df = data['merged_df']
all_imaging_cols = data['all_imaging_cols']
kinematic_cols = data['kinematic_cols']
imaging_columns_old = data['imaging_columns_old']
imaging_columns_new = data['imaging_columns_new']
imaging_columns_z = data['imaging_columns_z']

# Duplicate each patient row for left and right hemibodies.
hemibody_df = pd.concat([
    merged_df.assign(Analyzed_Hand='Left'),
    merged_df.assign(Analyzed_Hand='Right')
]).reset_index(drop=True)
hemibody_df['UniqueID'] = hemibody_df['Patient ID'].astype(str) + '_' + hemibody_df['Analyzed_Hand']

print(f"Merged data shape: {merged_df.shape}")
print(f"Hemibody data shape (should be double): {hemibody_df.shape}")

# -------------------------------
# Create Output Folders
# -------------------------------
outlook_dir = os.path.join(script_dir, "outlook")
outlook_data_dir = os.path.join(outlook_dir, "data")
outlook_plots_dir = os.path.join(outlook_dir, "plots")
os.makedirs(outlook_data_dir, exist_ok=True)
os.makedirs(outlook_plots_dir, exist_ok=True)

# -------------------------------
# Plotting via plotting.py module
# -------------------------------
plotting.plot_patients_kinematic(merged_df, kinematic_cols, outlook_plots_dir)
plotting.plot_correlation_heatmap(merged_df, all_imaging_cols, kinematic_cols, outlook_plots_dir)
plotting.plot_hand_condition(merged_df, outlook_plots_dir)

# -------------------------------
# Statistical Analysis: Contralateral Correlations
# -------------------------------
print("\nStarting contralateral correlation analysis...")
contralateral_results = []

# Loop through all imaging columns.
for im_col in all_imaging_cols:
    # Determine expected contralateral hand.
    if "Right" in im_col:
        expected_hand = "Left"
    elif "Left" in im_col:
        expected_hand = "Right"
    else:
        continue  # Skip non-lateralized imaging metrics (e.g., mean values)

    # Select data for the expected contralateral hand.
    subset_df = hemibody_df[hemibody_df["Analyzed_Hand"].str.lower() == expected_hand.lower()]
    
    # Compare imaging metrics against all kinematic variables.
    for kin_col in kinematic_cols:
        if pd.api.types.is_numeric_dtype(subset_df[im_col]) and pd.api.types.is_numeric_dtype(subset_df[kin_col]):
            valid_data = subset_df[['UniqueID', im_col, kin_col]].dropna()
            if valid_data[im_col].nunique() <= 1 or valid_data[kin_col].nunique() <= 1:
                continue
            if len(valid_data) > 2:
                corr, pval = stats.pearsonr(valid_data[im_col], valid_data[kin_col])
                contralateral_results.append({
                    'Hemibody Count': valid_data['UniqueID'].nunique(),
                    'Imaging Variable': im_col,
                    'Contralateral Hand': expected_hand,
                    'Kinematic Variable': kin_col,
                    'Pearson r': corr,
                    'p-value': pval,
                    'N': len(valid_data)
                })

contralateral_df = pd.DataFrame(contralateral_results)

# -------------------------------
# Significance Testing Without Multiple Comparisons Correction
# -------------------------------
# Set your desired significance level.
user_alpha = 0.01  # Change this value as needed.

if not contralateral_df.empty:
    # Use raw p-values directly.
    contralateral_df['p-value_corrected'] = contralateral_df['p-value']
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
# Generate Scatter Plots for Significant Correlations (Original)
# -------------------------------
if not contralateral_df.empty:
    sig_pairs = contralateral_df[contralateral_df['p-value_corrected'] < 0.01]
    plotting.plot_scatter_plots(sig_pairs, hemibody_df, outlook_plots_dir)
else:
    print("No contralateral correlation tests were performed.")

# ---------------------------------------------------------
# Analysis: Significant Pairs in Both Imaging Modalities (Old and New)
# ---------------------------------------------------------
# We ignore the Z-values here.
def get_modality(im_var):
    if im_var.endswith('_old'):
        return 'old'
    elif im_var.endswith('_new'):
        return 'new'
    else:
        return 'other'

def extract_region(im_var):
    if im_var.endswith('_old') or im_var.endswith('_new'):
        return im_var[:-4]  # remove the last 4 characters (_old or _new)
    else:
        return im_var

# Work only with imaging variables from the old and new modalities.
paired_df = contralateral_df[contralateral_df['Imaging Variable'].str.endswith(('_old', '_new'))].copy()
paired_df['modality'] = paired_df['Imaging Variable'].apply(get_modality)
paired_df['anatomical_region'] = paired_df['Imaging Variable'].apply(extract_region)

# Group by anatomical region and kinematic variable to identify pairs significant in both modalities.
paired_significant_list = []
for (region, kin), group in paired_df.groupby(['anatomical_region', 'Kinematic Variable']):
    modalities = set(group['modality'])
    if 'old' in modalities and 'new' in modalities:
        # Check that all entries in both modalities are flagged significant.
        old_signif = group[group['modality'] == 'old']['reject_null'].all()
        new_signif = group[group['modality'] == 'new']['reject_null'].all()
        if old_signif and new_signif:
            combined_entry = {
                'Anatomical Region': region,
                'Kinematic Variable': kin,
                'Pearson r (old)': group[group['modality']=='old']['Pearson r'].mean(),
                'p-value (old)': group[group['modality']=='old']['p-value'].mean(),
                'N (old)': group[group['modality']=='old']['N'].mean(),
                'Pearson r (new)': group[group['modality']=='new']['Pearson r'].mean(),
                'p-value (new)': group[group['modality']=='new']['p-value'].mean(),
                'N (new)': group[group['modality']=='new']['N'].mean(),
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
# This new plotting function creates paired scatter plots that now include the regression p-values and R² values.
if 'significant_pairs_df' in locals() and not significant_pairs_df.empty:
    plotting.plot_paired_scatter_plots(significant_pairs_df, hemibody_df, outlook_plots_dir)
else:
    print("No paired significant scatter plots to generate.")
