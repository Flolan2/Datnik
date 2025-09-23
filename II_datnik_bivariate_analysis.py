#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performs Bivariate Partial Correlation analysis (controlling for Age)
between OFF-state kinematics (FT & HM tasks) and contralateral Striatum
Z-scores using the hand-centric 'final_merged_data.csv'.

Generates individual plots for significant bivariate findings AND
a combined publication-ready Figure 1 summarizing all bivariate findings.
Saves raw and significant correlation results.
"""

import os
import sys
import pandas as pd
import numpy as np
import pingouin as pg
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import importlib
import traceback # Added for detailed error logging

print("\n" + "="*80)
print("--- RUNNING BIVARIATE ANALYSIS (with Age Control) ---")
print("="*80 + "\n")


try:
    import datnik_plotting
    importlib.reload(datnik_plotting)
    from datnik_plotting import plot_single_bivariate_scatter, get_readable_name, get_base_readable_name

    import datnik_analysis
    importlib.reload(datnik_analysis)
    from datnik_analysis import run_correlation_analysis
    PLOT_AVAILABLE = True
    print("[INFO] Successfully reloaded analysis and plotting modules.")
except ImportError as e:
    print(f"[CRITICAL WARNING] Error importing analysis/plotting functions: {e}")
    PLOT_AVAILABLE = False
    def run_correlation_analysis(*args, **kwargs): print("[ERROR] run_correlation_analysis not imported."); return pd.DataFrame(), {}
    def plot_single_bivariate_scatter(*args, **kwargs): print("[WARNING] plot_single_bivariate_scatter (with ax) not imported.")
    def get_readable_name(name, **kwargs): return name
    def get_base_readable_name(name, **kwargs): return name


# -------------------------
# 1. Load and Process Data
# -------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(script_dir)
processed_data_dir = os.path.join(project_root_dir, "Output", "Data_Processed")
merged_csv_file = os.path.join(processed_data_dir, "final_merged_data.csv")

print(f"--- Datnik Bivariate Analysis for Figure 1 (Age Controlled) ---")
print(f"[INFO] Attempting to load data from: {merged_csv_file}")

try:
    try: df_full = pd.read_csv(merged_csv_file, sep=';', decimal='.')
    except: df_full = pd.read_csv(merged_csv_file, sep=',', decimal='.')
    print(f"[INFO] Original data loaded successfully. Shape: {df_full.shape}")
    if 'Medication Condition' not in df_full.columns or 'Hand_Performed' not in df_full.columns:
        print("[CRITICAL ERROR] Essential columns missing. Exiting."); sys.exit(1)
except FileNotFoundError:
    print(f"[FATAL ERROR] Input file not found at '{merged_csv_file}'. Make sure Script I has been run successfully.")
    sys.exit(1)
except Exception as e:
    print(f"[FATAL ERROR] An unexpected error occurred while loading data: {e}"); sys.exit(1)

df_full['Medication Condition'] = df_full['Medication Condition'].astype(str).str.strip().str.lower()
df = df_full[df_full['Medication Condition'] == 'off'].copy()
if df.empty:
    print("[FATAL ERROR] No data found for Medication Condition == 'off'. Cannot proceed. Exiting."); sys.exit(0)

print(f"[INFO] Filtered for 'OFF' state. New data shape: {df.shape}")

# --- NEW: Age Control Setup ---
CONTROL_FOR_AGE = True
AGE_COL = 'Age'
if CONTROL_FOR_AGE and AGE_COL not in df.columns:
    print("\n" + "#"*60)
    print(f"### WARNING: Age control requested, but '{AGE_COL}' column not found! ###")
    print("### Reverting to standard Pearson correlation. ###")
    print("#"*60 + "\n")
    CONTROL_FOR_AGE = False
elif CONTROL_FOR_AGE:
    print(f"\n[INFO] Age control is ENABLED. Using '{AGE_COL}' as covariate.")
    df.dropna(subset=[AGE_COL], inplace=True)
    print(f"[INFO] Data shape after dropping rows with missing Age: {df.shape}")
else:
    print("\n[INFO] Age control is DISABLED. Running standard Pearson correlation.")

# --- Setup ---
base_kinematic_cols = [
    "meanamplitude", "stdamplitude", "meanspeed", "stdspeed", "meanrmsvelocity",
    "stdrmsvelocity", "meanopeningspeed", "stdopeningspeed", "meanclosingspeed",
    "stdclosingspeed", "meancycleduration", "stdcycleduration", "rangecycleduration",
    "rate", "amplitudedecay", "velocitydecay", "ratedecay", "cvamplitude",
    "cvcycleduration", "cvspeed", "cvrmsvelocity", "cvopeningspeed", "cvclosingspeed"
]
TARGET_IMAGING_BASE = "Contralateral_Striatum"
TARGET_IMAGING_COL = f"{TARGET_IMAGING_BASE}_Z"
tasks = ['ft', 'hm']

output_base_folder = os.path.join(project_root_dir, "Output")
data_output_folder = os.path.join(output_base_folder, "Data")
plots_folder = os.path.join(output_base_folder, "Plots")
os.makedirs(data_output_folder, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)

SIGNIFICANCE_ALPHA = 0.05
TOP_N_FOR_SCATTER_PANEL = 3

all_significant_bivariate_results_dfs = {}
all_raw_bivariate_results_list = []

print(f"\n=== Starting Bivariate Correlation Analysis (Target: {TARGET_IMAGING_COL}) ===")
if TARGET_IMAGING_COL not in df.columns:
    print(f"[FATAL ERROR] Target imaging column '{TARGET_IMAGING_COL}' not found. Skipping all analysis.")
else:
    # --- MODIFIED ANALYSIS LOOP WITH PARTIAL CORRELATION ---
    from statsmodels.stats.multitest import multipletests

    for task_prefix in tasks:
        print(f"\n--- Analyzing Task: {task_prefix.upper()} ---")
        task_results = []
        kinematic_cols_for_task = [f"{task_prefix}_{base}" for base in base_kinematic_cols if f"{task_prefix}_{base}" in df.columns]

        for base_col_name in base_kinematic_cols:
            kinematic_col_name = f"{task_prefix}_{base_col_name}"
            if kinematic_col_name not in df.columns: continue

            cols_for_corr = [kinematic_col_name, TARGET_IMAGING_COL]
            if CONTROL_FOR_AGE:
                cols_for_corr.append(AGE_COL)

            pair_data = df[cols_for_corr].dropna()

            if len(pair_data) < 5: continue # Need sufficient data for partial correlation

            if CONTROL_FOR_AGE:
                pcorr = pg.partial_corr(data=pair_data, x=kinematic_col_name, y=TARGET_IMAGING_COL, covar=AGE_COL)
                corr_coef = pcorr['r'].iloc[0]
                p_value = pcorr['p-val'].iloc[0]
                corr_type = 'Partial'
            else: # Fallback to Pearson
                from scipy.stats import pearsonr
                corr_coef, p_value = pearsonr(pair_data[kinematic_col_name], pair_data[TARGET_IMAGING_COL])
                corr_type = 'Pearson'

            if pd.notna(corr_coef):
                result_dict = {
                    "Task": task_prefix,
                    "Kinematic Variable": kinematic_col_name,
                    "Base Kinematic": base_col_name,
                    f"{corr_type} Correlation (r)": corr_coef,
                    "P-value (uncorrected)": p_value,
                    "N": len(pair_data)
                }
                task_results.append(result_dict)
                all_raw_bivariate_results_list.append(result_dict)

        if task_results:
            task_results_df = pd.DataFrame(task_results)
            p_values = task_results_df["P-value (uncorrected)"]
            reject, q_values, _, _ = multipletests(p_values, alpha=SIGNIFICANCE_ALPHA, method='fdr_bh')
            task_results_df['Q-value (FDR corrected)'] = q_values
            task_results_df['Significant (FDR)'] = reject
            
            significant_results_df = task_results_df[task_results_df['Significant (FDR)']].copy()
            all_significant_bivariate_results_dfs[task_prefix] = significant_results_df
            print(f"Found {len(significant_results_df)} significant age-controlled correlations for {task_prefix.upper()}.")
        else:
            print(f"No valid correlations could be computed for task {task_prefix.upper()}.")
            all_significant_bivariate_results_dfs[task_prefix] = pd.DataFrame()


# --- Final Data Aggregation ---
all_raw_bivariate_results_df = pd.DataFrame(all_raw_bivariate_results_list)
combined_significant_bivariate_df = pd.DataFrame()
if any(not df.empty for df in all_significant_bivariate_results_dfs.values()):
    combined_significant_bivariate_df = pd.concat(
        [df for df in all_significant_bivariate_results_dfs.values() if not df.empty],
        ignore_index=True
    )

# Rename correlation column for consistency in plots
corr_col_name = "Partial Correlation (r)" if CONTROL_FOR_AGE else "Pearson Correlation (r)"
if corr_col_name in combined_significant_bivariate_df.columns:
    combined_significant_bivariate_df.rename(columns={corr_col_name: "Correlation (r)"}, inplace=True)
if corr_col_name in all_raw_bivariate_results_df.columns:
    all_raw_bivariate_results_df.rename(columns={corr_col_name: "Correlation (r)"}, inplace=True)


# ==============================================================================
# --- GENERATE PUBLICATION-READY FIGURE 1 (AGE-CONTROLLED) ---
# ==============================================================================
print("\n--- [INFO] Checking prerequisites for generating Figure 1 ---")
if not PLOT_AVAILABLE:
    print("[INFO] RESULT: SKIPPING FIGURE 1 because PLOT_AVAILABLE is False.")
elif combined_significant_bivariate_df.empty:
    print("[INFO] RESULT: SKIPPING FIGURE 1 because no statistically significant results were found.")
else:
    print(f"[INFO] RESULT: Prerequisites MET. Found {len(combined_significant_bivariate_df)} significant results to plot. Proceeding.")
    try:
        print("\n\n--- Generating Figure 1: Bivariate Findings Summary (Age-Controlled) ---")

        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.2, 1], height_ratios=[1, 1])
        ax_A = fig.add_subplot(gs[0, 0])
        ax_C = fig.add_subplot(gs[0, 1])
        gs_B = gridspec.GridSpecFromSubplotSpec(2, TOP_N_FOR_SCATTER_PANEL, subplot_spec=gs[1, :])
        axes_B = [[fig.add_subplot(gs_B[i, j]) for j in range(TOP_N_FOR_SCATTER_PANEL)] for i in range(2)]

        # --- Panel A: Lollipop Plot of Significant Partial Correlations ---
        panel_A_data = combined_significant_bivariate_df.copy()
        panel_A_data['abs_r'] = panel_A_data['Correlation (r)'].abs()
        sort_order = panel_A_data.groupby('Base Kinematic')['abs_r'].mean().sort_values().index
        panel_A_data_pivot = panel_A_data.pivot_table(index='Base Kinematic', columns='Task', values='Correlation (r)')
        panel_A_data_pivot = panel_A_data_pivot.reindex(sort_order)
        y_labels_readable = [get_base_readable_name(name) for name in panel_A_data_pivot.index]
        y_pos = np.arange(len(y_labels_readable))
        for i, base_kin in enumerate(panel_A_data_pivot.index):
            r_ft = panel_A_data_pivot.loc[base_kin, 'ft']
            r_hm = panel_A_data_pivot.loc[base_kin, 'hm']
            if pd.notna(r_ft) and pd.notna(r_hm):
                ax_A.plot([r_ft, r_hm], [y_pos[i], y_pos[i]], color='grey', alpha=0.5, linewidth=1)
        ax_A.scatter(panel_A_data_pivot['ft'], y_pos, color='#007acc', zorder=3, label='Finger Tapping (FT)')
        ax_A.scatter(panel_A_data_pivot['hm'], y_pos, color='#e85f5f', zorder=3, label='Hand Movements (HM)')
        ax_A.axvline(0, color='black', linestyle='--', linewidth=0.8)
        ax_A.set_yticks(y_pos); ax_A.set_yticklabels(y_labels_readable, fontsize=9)
        ax_A.set_xlabel("Partial Correlation Coefficient (r), controlling for Age", fontsize=11)
        ax_A.set_title("A) Significant Age-Controlled Kinematic-Striatal Correlations", fontsize=13, weight='bold', loc='left')
        ax_A.legend(loc='lower right'); ax_A.grid(axis='x', linestyle=':', alpha=0.6); sns.despine(ax=ax_A, left=True)

        # --- Panel B: Top Scatter Plots of Residuals ---
        y_lim_global = (df[TARGET_IMAGING_COL].min() - 0.5, df[TARGET_IMAGING_COL].max() + 0.5)
        for i, task_prefix in enumerate(tasks):
            task_df_sig = all_significant_bivariate_results_dfs.get(task_prefix, pd.DataFrame())
            if task_df_sig.empty: continue
            
           # Use the correct correlation column name defined earlier
            task_df_sig['abs_r'] = task_df_sig[corr_col_name].abs()
            top_results = task_df_sig.sort_values(by=['Q-value (FDR corrected)', 'abs_r'], ascending=[True, False]).head(TOP_N_FOR_SCATTER_PANEL)
            
            for j, (_, row) in enumerate(top_results.iterrows()):
                ax_b = axes_B[i][j]
                kin_col = row['Kinematic Variable']
                # Use the correct correlation column name to populate the stats dictionary
                stats = {'r': row[corr_col_name], 'q': row['Q-value (FDR corrected)'], 'N': row['N']}


                # --- NEW: Calculate and plot residuals ---
                plot_data = df[[kin_col, TARGET_IMAGING_COL, AGE_COL]].dropna()
                
                # Regress out age from kinematic variable
                kin_model = ols(f"{kin_col} ~ {AGE_COL}", data=plot_data).fit()
                plot_data['kin_resid'] = kin_model.resid
                
                # Regress out age from imaging variable
                img_model = ols(f"{TARGET_IMAGING_COL} ~ {AGE_COL}", data=plot_data).fit()
                plot_data['img_resid'] = img_model.resid
                
                # --- This is the new, correct block ---
                # Corrected line
                sns.regplot(x='kin_resid', y='img_resid', data=plot_data, ax=ax_b,
                            scatter_kws={'alpha': 0.3, 'edgecolor': 'k', 'linewidths': 0.8},
                            line_kws={'color': 'k', 'linestyle': '--'})
                
                # Update labels and title for residuals
                ax_b.set_xlabel(f"{get_readable_name(kin_col)}\n(Residuals vs. Age)", fontsize=9)
                ax_b.set_ylabel(f"{get_readable_name(TARGET_IMAGING_COL)}\n(Residuals vs. Age)", fontsize=9)
                ax_b.set_title(f"{task_prefix.upper()}: {get_base_readable_name(kin_col)}", fontsize=10, weight='bold')

                # Add stats box
                stats_text = f"Partial r = {stats['r']:.2f}\nq = {stats['q']:.4f}\nN = {stats['N']}"
                ax_b.text(0.05, 0.95, stats_text, transform=ax_b.transAxes, fontsize=8,
                          verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='whitesmoke', alpha=0.8))

        fig.text(0.5, 0.48, "B) Top Age-Controlled Kinematic Correlates of Dopaminergic Deficit", ha='center', fontsize=13, weight='bold')

        # --- Panel C: Concordance Plot of Partial Correlations ---
        panel_C_data = all_raw_bivariate_results_df.pivot_table(index='Base Kinematic', columns='Task', values='Correlation (r)')
        panel_C_data.dropna(inplace=True)
        sig_ft = combined_significant_bivariate_df[combined_significant_bivariate_df['Task']=='ft']['Base Kinematic'].unique()
        sig_hm = combined_significant_bivariate_df[combined_significant_bivariate_df['Task']=='hm']['Base Kinematic'].unique()
        
        def get_sig_status(base_kin):
            in_ft = base_kin in sig_ft; in_hm = base_kin in sig_hm
            if in_ft and in_hm: return "Significant in Both"
            if in_ft: return "Significant in FT only"
            if in_hm: return "Significant in HM only"
            return "Not Significant"
        
        panel_C_data['Significance'] = panel_C_data.index.map(get_sig_status)
        size_map = {'Significant in Both': 80, 'Significant in FT only': 60, 'Significant in HM only': 60, 'Not Significant': 30}
        panel_C_data['PointSize'] = panel_C_data['Significance'].map(size_map)
        sns.scatterplot(data=panel_C_data, x='ft', y='hm', hue='Significance', style='Significance',
                        palette={'Significant in Both': 'black', 'Significant in FT only': '#007acc',
                                 'Significant in HM only': '#e85f5f', 'Not Significant': 'lightgrey'},
                        markers={'Significant in Both': 'o', 'Significant in FT only': 's',
                                 'Significant in HM only': '^', 'Not Significant': '.'},
                        size='PointSize', sizes=(30, 100), ax=ax_C, zorder=3)
        ax_C.plot([-1, 1], [-1, 1], color='black', linestyle='--', linewidth=1, alpha=0.7, zorder=1)
        ax_C.set_xlabel("Partial Correlation Strength (r) in FT (Age Controlled)", fontsize=11)
        ax_C.set_ylabel("Partial Correlation Strength (r) in HM (Age Controlled)", fontsize=11)
        ax_C.set_title("C) Concordance of Age-Controlled Correlations", fontsize=13, weight='bold', loc='left')
        ax_C.set_aspect('equal', adjustable='box'); ax_C.grid(True, linestyle=':', alpha=0.6)
        
        # Concordance is now a simple Pearson correlation of the partial correlation values
        concordance_r, concordance_p = pg.corr(panel_C_data['ft'], panel_C_data['hm'])['r'].iloc[0], pg.corr(panel_C_data['ft'], panel_C_data['hm'])['p-val'].iloc[0]
        p_text = f"p < 0.001" if concordance_p < 0.001 else f"p = {concordance_p:.3f}"
        ax_C.text(0.05, 0.95, f"Concordance:\nr = {concordance_r:.2f}\n{p_text}", transform=ax_C.transAxes,
                  fontsize=10, va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='whitesmoke', alpha=0.8))
        
        handles, labels = ax_C.get_legend_handles_labels()
        sig_labels_indices = [i for i, label in enumerate(labels) if label in size_map.keys()]
        ax_C.legend([handles[i] for i in sig_labels_indices], [labels[i] for i in sig_labels_indices],
                    title="Significance (q<0.05)", fontsize=8)

        # --- Final adjustments and saving ---
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.suptitle('Figure 1: Age-Controlled Kinematic Correlates of Contralateral Striatal Dopamine Deficit (OFF State)', fontsize=16, weight='bold')

        figure_1_filename = os.path.join(plots_folder, "Figure1_Bivariate_Findings_Summary_AgeControlled.png")
        plt.savefig(figure_1_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\n--- SUCCESS! Age-controlled Figure 1 saved to: {os.path.abspath(figure_1_filename)} ---")

    except Exception as e:
        print("\n" + "!"*60)
        print("!!! AN UNEXPECTED ERROR OCCURRED DURING FIGURE 1 GENERATION OR SAVING !!!")
        print(f"!!! Error Type: {type(e).__name__}")
        print(f"!!! Error Message: {e}")
        print("!!! Printing detailed traceback:")
        traceback.print_exc()
        print("!"*60 + "\n")

print("\n--- Bivariate analysis script execution finished ---")