# --- START OF FILE datnik_plot_hm_summary.py ---

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates a summary figure for Hand Movement (HM) OFF-state findings,
combining results from PLS, ElasticNet, Ridge, and Bivariate analyses
based on pre-generated CSV files.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import pearsonr # Ensure this is imported

# Optional: For adjusting text labels in scatter plot
try:
    from adjustText import adjust_text
    ADJUSTTEXT_AVAILABLE = True
except ImportError:
    ADJUSTTEXT_AVAILABLE = False

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "Output", "Data")
OUTPUT_PLOT_DIR = os.path.join(SCRIPT_DIR, "..", "Output", "Plots")
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# --- File Names (Ensure these match your actual files) ---
PLS_RESULTS_FILE = os.path.join(OUTPUT_DATA_DIR, "pls_significant_results_all_tasks_combined_sorted.csv")
ENET_COEFF_FILE = os.path.join(OUTPUT_DATA_DIR, "elasticnet_hm_OFF_coefficients.csv")
RIDGE_COEFF_FILE = os.path.join(OUTPUT_DATA_DIR, "ridge_hm_OFF_coefficients.csv")
BIVAR_COMBINED_FILE = os.path.join(OUTPUT_DATA_DIR, "significant_correlations_Contralateral_Striatum_Z_combined_OFF.csv") # Contains bivariate 'r'

# --- Plotting Parameters ---
PLS_BSR_THRESHOLD = 2.0 # Threshold used in PLS analysis to define reliable features
TOP_N_BARPLOT = 15      # Max features to show in bar plots (Ridge/ElasticNet)
TOP_N_LABEL_SCATTER = 7 # Max features to label in scatter plot
FIG_FILENAME = os.path.join(OUTPUT_PLOT_DIR, "figure_hm_multivariate_summary_OFF_v2.png") # Added v2 to filename

# --- Helper Function & Map (copied from datnik_plotting.py for self-containment) ---
READABLE_KINEMATIC_NAMES = {
    "meanamplitude": "Mean Amp", "stdamplitude": "STD Amp", "meanspeed": "Mean Speed",
    "stdspeed": "STD Speed", "meanrmsvelocity": "Mean RMS Vel", "stdrmsvelocity": "STD RMS Vel",
    "meanopeningspeed": "Mean Open Speed", "stdopeningspeed": "STD Open Speed",
    "meanclosingspeed": "Mean Close Speed", "stdclosingspeed": "STD Close Speed",
    "meancycleduration": "Mean Cycle Dur", "stdcycleduration": "STD Cycle Dur",
    "rangecycleduration": "Range Cycle Dur", "rate": "Rate", "amplitudedecay": "Amp Decay",
    "velocitydecay": "Vel Decay", "ratedecay": "Rate Decay", "cvamplitude": "CV Amp",
    "cvcycleduration": "CV Cycle Dur", "cvspeed": "CV Speed", "cvrmsvelocity": "CV RMS Vel",
    "cvopeningspeed": "CV Open Speed", "cvclosingspeed": "CV Close Speed"
}

def get_base_readable_name(raw_name, name_map=READABLE_KINEMATIC_NAMES):
    """Gets readable base kinematic name (e.g., 'CV Amplitude')."""
    base_name = raw_name.split('_', 1)[-1]
    return name_map.get(base_name, base_name)

# --- Main Function ---
def create_hm_summary_figure():
    """Loads data and creates the summary figure."""
    print("--- Generating HM Multivariate Summary Figure (v2) ---")

    # --- 1. Load Data ---
    try:
        df_pls = pd.read_csv(PLS_RESULTS_FILE, sep=';', decimal='.')
        df_enet = pd.read_csv(ENET_COEFF_FILE, sep=';', decimal='.')
        df_ridge = pd.read_csv(RIDGE_COEFF_FILE, sep=';', decimal='.')
        df_bivar = pd.read_csv(BIVAR_COMBINED_FILE, sep=';', decimal='.')
        print("Successfully loaded all required CSV files.")
    except FileNotFoundError as e:
        print(f"Error: Could not find input file: {e}. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV files: {e}. Exiting.")
        sys.exit(1)

    # --- 2. Prepare PLS Data (Panel A) ---
    df_pls_hm_lv1 = df_pls[(df_pls['Task'] == 'hm') & (df_pls['LV'] == 1)].copy()
    if df_pls_hm_lv1.empty:
        print("Warning: No PLS data found for HM Task, LV 1. Skipping PLS plot.")
        pls_plot_data = None
        pls_sig_features = []
        lv1_corr, lv1_pval = np.nan, np.nan # Default values
    else:
        df_pls_hm_lv1['Abs_BSR'] = df_pls_hm_lv1['Bootstrap_Ratio'].abs()
        pls_plot_data = df_pls_hm_lv1.sort_values(by='Loading', ascending=True)
        pls_plot_data['Readable_Feature'] = pls_plot_data['Kinematic_Variable'].apply(get_base_readable_name)
        pls_sig_features = df_pls_hm_lv1[df_pls_hm_lv1['Abs_BSR'] >= PLS_BSR_THRESHOLD]['Kinematic_Variable'].tolist()
        try:
             lv1_corr = df_pls_hm_lv1['Correlation_LV'].iloc[0]
             lv1_pval = df_pls_hm_lv1['P_value_LV'].iloc[0]
        except IndexError:
             lv1_corr, lv1_pval = np.nan, np.nan


    # --- 3. Prepare ElasticNet Data (Panel B) ---
    if df_enet.empty:
        print("Warning: ElasticNet coefficient data is empty. Skipping ElasticNet plot.")
        enet_plot_data = None
        r2_enet = np.nan # Default value
    else:
        # <<< Extract R2 for ENet >>>
        r2_enet = df_enet['R2_Full_Data'].iloc[0] if 'R2_Full_Data' in df_enet.columns and not df_enet.empty else np.nan
        df_enet_filtered = df_enet[df_enet['Coefficient'].abs() > 1e-9].copy() # Remove zero coeffs
        if df_enet_filtered.empty:
             print("Warning: No non-zero ElasticNet coefficients found.")
             enet_plot_data = None
        else:
            df_enet_filtered['Abs_Coeff'] = df_enet_filtered['Coefficient'].abs()
            enet_plot_data = df_enet_filtered.sort_values(by='Abs_Coeff', ascending=False)\
                                            .head(TOP_N_BARPLOT)\
                                            .sort_values(by='Abs_Coeff', ascending=True)
            enet_plot_data['Readable_Feature'] = enet_plot_data['Feature'].apply(get_base_readable_name)
            enet_plot_data['Is_PLS_Sig'] = enet_plot_data['Feature'].isin(pls_sig_features)


    # --- 4. Prepare Bivariate vs ElasticNet Data (Panel C) ---
    if df_bivar.empty or df_enet.empty:
        print("Warning: Missing Bivariate or ElasticNet data for comparison plot.")
        scatter_plot_data = None
    else:
        df_bivar_hm = df_bivar[df_bivar['Task'] == 'hm'][['Kinematic Variable', 'Pearson Correlation (r)', 'Significant (FDR)']].copy()
        df_bivar_hm = df_bivar_hm.rename(columns={'Kinematic Variable': 'Feature', 'Pearson Correlation (r)': 'Bivar_r'})

        scatter_plot_data = pd.merge(df_enet[['Feature', 'Coefficient']], df_bivar_hm, on='Feature', how='left')
        scatter_plot_data.dropna(subset=['Coefficient', 'Bivar_r'], inplace=True)

        if scatter_plot_data.empty:
            print("Warning: No overlapping data found for Bivariate vs ElasticNet scatter plot.")
            scatter_plot_data = None
        else:
            scatter_plot_data['Abs_Coeff_ENet'] = scatter_plot_data['Coefficient'].abs()
            scatter_plot_data.sort_values(by='Abs_Coeff_ENet', ascending=False, inplace=True)
            scatter_plot_data['Readable_Feature'] = scatter_plot_data['Feature'].apply(get_base_readable_name)
            # <<< Robust handling of boolean column >>>
            scatter_plot_data['Significant (FDR)'] = scatter_plot_data['Significant (FDR)'].fillna(False).astype(bool)


    # --- 5. Prepare Ridge Data (Panel D) ---
    if df_ridge.empty:
        print("Warning: Ridge coefficient data is empty. Skipping Ridge plot.")
        ridge_plot_data = None
        r2_ridge = np.nan # Default value
    else:
        # <<< Extract R2 for Ridge >>>
        r2_ridge = df_ridge['R2_Full_Data'].iloc[0] if 'R2_Full_Data' in df_ridge.columns and not df_ridge.empty else np.nan
        df_ridge['Abs_Coeff'] = df_ridge['Coefficient'].abs()
        ridge_plot_data = df_ridge.sort_values(by='Abs_Coeff', ascending=False)\
                                    .head(TOP_N_BARPLOT)\
                                    .sort_values(by='Abs_Coeff', ascending=True)
        ridge_plot_data['Readable_Feature'] = ridge_plot_data['Feature'].apply(get_base_readable_name)


    # --- 6. Create Figure ---
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Hand Movement (HM) OFF-State Kinematics vs Contralateral Striatum Z-Score", fontsize=16, weight='bold')
    axes = axes.flatten()

    color_pos = '#1f77b4' # Blue
    color_neg = '#d62728' # Red

    # --- Panel A: PLS Loadings ---
    ax = axes[0]
    if pls_plot_data is not None:
        colors = [color_pos if l >= 0 else color_neg for l in pls_plot_data['Loading']]
        edgecolors = ['black' if abs(bsr) >= PLS_BSR_THRESHOLD else 'grey' for bsr in pls_plot_data['Bootstrap_Ratio'].fillna(0)] # Added fillna(0) for safety
        linewidths = [1.2 if abs(bsr) >= PLS_BSR_THRESHOLD else 0.6 for bsr in pls_plot_data['Bootstrap_Ratio'].fillna(0)] # Added fillna(0) for safety

        ax.barh(pls_plot_data['Readable_Feature'], pls_plot_data['Loading'],
                color=colors, edgecolor=edgecolors, linewidth=linewidths, alpha=0.8)
        ax.set_xlabel("PLS Loading on LV1", fontsize=10)
        ax.set_ylabel("Kinematic Feature", fontsize=10)
        # <<< Updated P-value formatting >>>
        pval_str_a = f"p={lv1_pval:.3g}" if pd.notna(lv1_pval) and lv1_pval >= 0.001 else ("p<0.001" if pd.notna(lv1_pval) else "p=N/A")
        ax.set_title(f"A) PLS Loadings (HM LV1: r={lv1_corr:.2f}, {pval_str_a})", fontsize=12, weight='semibold')
        ax.axvline(0, color='black', linestyle='--', linewidth=0.7)
        ax.tick_params(axis='both', which='major', labelsize=9)
        reliable_patch = mpatches.Patch(edgecolor='black', facecolor='grey', linewidth=1.2, label=f'Reliable (|BSR|≥{PLS_BSR_THRESHOLD})')
        unreliable_patch = mpatches.Patch(edgecolor='grey', facecolor='grey', linewidth=0.6, label=f'Unreliable (|BSR|<{PLS_BSR_THRESHOLD})')
        ax.legend(handles=[reliable_patch, unreliable_patch], fontsize=8, loc='lower right')
    else:
        ax.text(0.5, 0.5, "PLS Data Not Available", ha='center', va='center', transform=ax.transAxes)
    ax.text(-0.1, 1.05, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')


    # --- Panel B: ElasticNet Coefficients ---
    ax = axes[1]
    if enet_plot_data is not None:
        colors = [color_pos if c >= 0 else color_neg for c in enet_plot_data['Coefficient']]
        edgecolors = ['black' if pls_sig else 'grey' for pls_sig in enet_plot_data['Is_PLS_Sig']]
        hatches = ['' if pls_sig else '///' for pls_sig in enet_plot_data['Is_PLS_Sig']]

        bars = ax.barh(enet_plot_data['Readable_Feature'], enet_plot_data['Coefficient'],
                       color=colors, edgecolor=edgecolors, hatch=hatches, linewidth=1.0, alpha=0.85)

        ax.set_xlabel("ElasticNet Coefficient", fontsize=10)
        ax.set_ylabel("")
        # <<< Updated Title with R2 >>>
        title_b = f"B) ElasticNet Coefficients (Top {len(enet_plot_data)}, L1={df_enet['Optimal_L1_Ratio'].iloc[0]:.1f}, R²={r2_enet:.2f})"
        ax.set_title(title_b, fontsize=12, weight='semibold')
        ax.axvline(0, color='black', linestyle='--', linewidth=0.7)
        ax.tick_params(axis='both', which='major', labelsize=9)
        solid_patch = mpatches.Patch(facecolor='grey', edgecolor='black', label='PLS Reliable')
        hatched_patch = mpatches.Patch(facecolor='grey', edgecolor='grey', hatch='///', label='PLS Unreliable')
        ax.legend(handles=[solid_patch, hatched_patch], title="Feature Reliability (PLS BSR)", fontsize=8, title_fontsize=9, loc='lower right')
    else:
        ax.text(0.5, 0.5, "ElasticNet Data Not Available", ha='center', va='center', transform=ax.transAxes)
    ax.text(-0.1, 1.05, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')


    # --- Panel C: Bivariate r vs ElasticNet Coefficient ---
    ax = axes[2]
    if scatter_plot_data is not None:
        # <<< Updated scatter plotting call >>>
        markers_c = {"Significant (FDR)": "o"} # Use 'o' for all
        sizes_c = {True: 60, False: 40} # Keep size difference
        palette_c = {True: color_neg, False: color_pos} # Sig Bivar = Red

        sns.scatterplot(
            data=scatter_plot_data, x='Bivar_r', y='Coefficient',
            size='Significant (FDR)', sizes=sizes_c, # Use size
            hue='Significant (FDR)', palette=palette_c, # Use hue
            marker='o', # Set marker directly
            alpha=0.7, legend=False, # Turn off automatic legend
            ax=ax
        )

        # Calculate correlation and format annotation
        corr_val, p_val = np.nan, np.nan
        if len(scatter_plot_data) >= 3:
             try:
                 corr_val, p_val = pearsonr(scatter_plot_data['Bivar_r'], scatter_plot_data['Coefficient'])
             except ValueError: pass
        # <<< Updated Annotation with p-value >>>
        pval_str_c = f", p={p_val:.3g}" if pd.notna(p_val) and p_val >= 0.001 else (", p<0.001" if pd.notna(p_val) else "")
        ax.text(0.03, 0.97, f'r={corr_val:.2f}{pval_str_c}', transform=ax.transAxes, fontsize=9, va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='whitesmoke', alpha=0.7))

        # Label top features
        texts = []
        top_features_scatter = scatter_plot_data.head(TOP_N_LABEL_SCATTER)
        for _, row in top_features_scatter.iterrows():
            texts.append(ax.text(row['Bivar_r'], row['Coefficient'], row['Readable_Feature'], fontsize=8))

        if ADJUSTTEXT_AVAILABLE and texts:
            try:
                adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5), ax=ax)
            except Exception as e_adj:
                print(f"AdjustText warning (Panel C): {e_adj}")
        elif texts:
             print("Note: 'adjustText' not available for Panel C label adjustment.")

        ax.set_xlabel("Bivariate Pearson r", fontsize=10)
        ax.set_ylabel("ElasticNet Coefficient", fontsize=10)
        ax.set_title("C) Bivariate vs ElasticNet Relationship", fontsize=12, weight='semibold')
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.7, alpha=0.7)
        ax.axvline(0, color='grey', linestyle='--', linewidth=0.7, alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=9)

        # <<< Updated Manual Legend Creation for Panel C >>>
        legend_handles_c = []
        # Check which categories actually exist in the data before creating handles
        sig_present = True in scatter_plot_data['Significant (FDR)'].unique()
        nonsig_present = False in scatter_plot_data['Significant (FDR)'].unique()

        if nonsig_present:
            legend_handles_c.append(plt.scatter([],[], marker='o', s=40, color=palette_c[False], label='Bivar Non-Sig (q>0.05)'))
        if sig_present:
            legend_handles_c.append(plt.scatter([],[], marker='o', s=60, color=palette_c[True], label='Bivar Sig (q<=0.05)'))

        if legend_handles_c:
            ax.legend(handles=legend_handles_c, title='Bivariate FDR', title_fontsize='9', fontsize=8, loc='lower right')

    else:
        ax.text(0.5, 0.5, "Comparison Data Not Available", ha='center', va='center', transform=ax.transAxes)
    ax.text(-0.1, 1.05, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

    # --- Panel D: Ridge Coefficients ---
    ax = axes[3]
    if ridge_plot_data is not None:
        colors = [color_pos if c >= 0 else color_neg for c in ridge_plot_data['Coefficient']]
        ax.barh(ridge_plot_data['Readable_Feature'], ridge_plot_data['Coefficient'],
                color=colors, edgecolor='black', linewidth=0.6, alpha=0.85)

        ax.set_xlabel("Ridge Coefficient", fontsize=10)
        ax.set_ylabel("")
        # <<< Updated Title with R2 >>>
        title_d = f"D) Ridge Coefficients (Top {len(ridge_plot_data)}, R²={r2_ridge:.2f})"
        ax.set_title(title_d, fontsize=12, weight='semibold')
        ax.axvline(0, color='black', linestyle='--', linewidth=0.7)
        ax.tick_params(axis='both', which='major', labelsize=9)
    else:
        ax.text(0.5, 0.5, "Ridge Data Not Available", ha='center', va='center', transform=ax.transAxes)
    ax.text(-0.1, 1.05, 'D', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')


    # --- Final Adjustments and Save ---
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    try:
        plt.savefig(FIG_FILENAME, dpi=300, bbox_inches='tight')
        print(f"\nSummary figure saved successfully to: {FIG_FILENAME}")
    except Exception as e:
        print(f"\nError saving figure: {e}")

    plt.close(fig)

# --- Run the function ---
if __name__ == "__main__":
    create_hm_summary_figure()

# --- END OF FILE datnik_plot_hm_summary.py ---