#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 20:41:31 2025

@author: Lange_L
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_patients_kinematic(merged_df, kinematic_cols, outlook_plots_dir):
    """
    Bar plot for count of available kinematic measurements per patient.
    """
    kinematic_counts = merged_df[kinematic_cols].notna().sum(axis=1)
    plt.figure(figsize=(10, 6))
    plt.bar(merged_df['Patient ID'].astype(str), kinematic_counts)
    plt.xlabel('Patient ID')
    plt.ylabel('Count of Available Kinematic Measurements')
    plt.title('Kinematic Data Availability per Patient')
    plt.xticks(rotation=90)
    plt.tight_layout()
    patient_plot_path = os.path.join(outlook_plots_dir, "patients_kinematic_data.png")
    plt.savefig(patient_plot_path)
    plt.close()
    print(f"Patient kinematic data plot saved to: {patient_plot_path}")

def plot_correlation_heatmap(merged_df, all_imaging_cols, kinematic_cols, outlook_plots_dir):
    """
    Heatmap showing correlations between imaging and kinematic variables.
    """
    corr_matrix = pd.DataFrame(index=all_imaging_cols, columns=kinematic_cols)
    for im in all_imaging_cols:
        for kin in kinematic_cols:
            try:
                corr = merged_df[[im, kin]].corr().iloc[0, 1]
            except Exception:
                corr = np.nan
            corr_matrix.loc[im, kin] = corr
    corr_matrix = corr_matrix.astype(float)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, aspect='auto', cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(label='Pearson r')
    plt.xticks(ticks=range(len(kinematic_cols)), labels=kinematic_cols, rotation=90, fontsize=8)
    plt.yticks(ticks=range(len(all_imaging_cols)), labels=all_imaging_cols, fontsize=8)
    plt.title("Correlation Heatmap: Imaging vs. Kinematic Variables")
    heatmap_path = os.path.join(outlook_plots_dir, "correlation_heatmap.png")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Correlation heatmap saved to: {heatmap_path}")

def plot_hand_condition(merged_df, outlook_plots_dir):
    """
    Bar plot for the frequency of hand conditions.
    """
    if "Hand Condition" in merged_df.columns:
        plt.figure()
        merged_df["Hand Condition"].value_counts().plot(kind="bar")
        plt.xlabel("Hand Condition")
        plt.ylabel("Count")
        plt.title("Frequency of Hand Condition")
        handplot_path = os.path.join(outlook_plots_dir, "hand_condition_barplot.png")
        plt.tight_layout()
        plt.savefig(handplot_path)
        plt.close()
        print(f"Hand Condition bar plot saved to: {handplot_path}")
    else:
        print("Hand Condition column not found; skipping hand condition plot.")

def plot_scatter_plots(significant_pairs, hemibody_df, outlook_plots_dir):
    """
    Create scatter plots for significant correlations.
    """
    significance_level = 0.01
    if not significant_pairs.empty:
        print(f"\nGenerating scatter plots for significant pairs (corrected p-value < {significance_level})...")
        for idx, row in significant_pairs.iterrows():
            im_col = row['Imaging Variable']
            kin_col = row['Kinematic Variable']
            expected_hand = row['Contralateral Hand']
            
            subset_df = hemibody_df[hemibody_df["Analyzed_Hand"].str.lower() == expected_hand.lower()]
            plot_data = subset_df[[im_col, kin_col]].dropna()
            if plot_data.empty:
                continue
            
            plt.figure()
            plt.scatter(plot_data[im_col], plot_data[kin_col], label='Data Points')
            plt.xlabel(im_col)
            plt.ylabel(kin_col)
            plt.title(f'{im_col} vs. {kin_col}\n(corrected p-value = {row["p-value_corrected"]:.3g}, N = {row["N"]})')
            
            if len(plot_data) > 2:
                slope, intercept = np.polyfit(plot_data[im_col], plot_data[kin_col], 1)
                x_vals = np.linspace(plot_data[im_col].min(), plot_data[im_col].max(), 100)
                y_vals = slope * x_vals + intercept
                plt.plot(x_vals, y_vals, color='red', label='Fit')
            
            plt.legend()
            plot_filename = f"{im_col}_vs_{kin_col}_scatter.png".replace(" ", "_")
            plot_filepath = os.path.join(outlook_plots_dir, plot_filename)
            plt.tight_layout()
            plt.savefig(plot_filepath)
            plt.close()
            print(f"Scatter plot for {im_col} vs. {kin_col} saved.")
    else:
        print("\nNo significant pairs found for scatter plots.")

def plot_paired_scatter_plots(paired_significant_df, hemibody_df, outlook_plots_dir):
    """
    Create paired scatter plots for significant correlations found in both the old and new imaging modalities.
    Each figure contains two subplots (side by side) with fit lines, and the plots are saved into a new subfolder 'paired'.
    Each subplot is annotated with the regression p-value and R².
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    paired_folder = os.path.join(outlook_plots_dir, "paired")
    os.makedirs(paired_folder, exist_ok=True)
    
    for idx, row in paired_significant_df.iterrows():
        region = row['Anatomical Region']
        kin = row['Kinematic Variable']
        
        # Determine expected contralateral hand based on region laterality.
        if "Right" in region:
            expected_hand = "Left"
        elif "Left" in region:
            expected_hand = "Right"
        else:
            expected_hand = None
        
        if expected_hand is not None:
            subset_df = hemibody_df[hemibody_df["Analyzed_Hand"].str.lower() == expected_hand.lower()]
        else:
            subset_df = hemibody_df
        
        im_old = f"{region}_old"
        im_new = f"{region}_new"
        
        # Only create plot if the required columns exist.
        if im_old not in subset_df.columns or im_new not in subset_df.columns or kin not in subset_df.columns:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # ----------------------
        # Plot for Old Modality
        # ----------------------
        axes[0].scatter(subset_df[im_old], subset_df[kin])
        axes[0].set_title(f"{region} (old)\nKinematic: {kin}")
        axes[0].set_xlabel(im_old)
        axes[0].set_ylabel(kin)
        data_old = subset_df[[im_old, kin]].dropna()
        if len(data_old) > 2:
            # Calculate regression parameters using linregress.
            slope, intercept, r_value, p_value, std_err = stats.linregress(data_old[im_old], data_old[kin])
            r_squared = r_value**2
            x_vals = np.linspace(data_old[im_old].min(), data_old[im_old].max(), 100)
            y_vals = slope * x_vals + intercept
            axes[0].plot(x_vals, y_vals, color='red', label='Fit')
            axes[0].legend()
            # Annotate with p-value and R².
            axes[0].text(0.05, 0.95, f'p = {p_value:.3g}\nR² = {r_squared:.3g}', 
                         transform=axes[0].transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        
        # ----------------------
        # Plot for New Modality
        # ----------------------
        axes[1].scatter(subset_df[im_new], subset_df[kin])
        axes[1].set_title(f"{region} (new)\nKinematic: {kin}")
        axes[1].set_xlabel(im_new)
        axes[1].set_ylabel(kin)
        data_new = subset_df[[im_new, kin]].dropna()
        if len(data_new) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(data_new[im_new], data_new[kin])
            r_squared = r_value**2
            x_vals = np.linspace(data_new[im_new].min(), data_new[im_new].max(), 100)
            y_vals = slope * x_vals + intercept
            axes[1].plot(x_vals, y_vals, color='red', label='Fit')
            axes[1].legend()
            axes[1].text(0.05, 0.95, f'p = {p_value:.3g}\nR² = {r_squared:.3g}', 
                         transform=axes[1].transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        
        fig.suptitle(f"Paired Scatter Plot for {region} vs {kin}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        paired_plot_filename = f"paired_scatter_{region.replace(' ', '_')}_{kin.replace(' ', '_')}.png"
        paired_plot_filepath = os.path.join(paired_folder, paired_plot_filename)
        plt.savefig(paired_plot_filepath)
        plt.close(fig)
        print(f"Paired scatter plot saved to: {paired_plot_filepath}")
