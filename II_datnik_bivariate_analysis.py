#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performs Bivariate Partial Correlation analysis (controlling for Age)
between OFF-state kinematics (FT & HM tasks) and BOTH contralateral AND
ipsilateral Striatum Z-scores using the hand-centric 'final_merged_data.csv'.

Generates individual plots for significant bivariate findings AND
a combined publication-ready Figure 1 summarizing all bivariate findings FOR EACH SIDE.
Saves raw and significant correlation results for each side separately.
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
print("--- RUNNING BIVARIATE ANALYSIS (with Age Control) for CONTRALATERAL & IPSILATERAL ---")
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

SIDES_TO_ANALYZE = ['Contralateral', 'Ipsilateral']
tasks = ['ft', 'hm']

output_base_folder = os.path.join(project_root_dir, "Output")
data_output_folder = os.path.join(output_base_folder, "Data")
plots_folder = os.path.join(output_base_folder, "Plots")
os.makedirs(data_output_folder, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)


for side_prefix in SIDES_TO_ANALYZE:
    print("\n" + "#"*80)
    print(f"###   STARTING ANALYSIS FOR: {side_prefix.upper()} SIDE   ###")
    print("#"*80)

    TARGET_IMAGING_BASE = f"{side_prefix}_Putamen"
    TARGET_IMAGING_COL = f"{TARGET_IMAGING_BASE}_Z"

    # --- START: Generate Cohort Summary Statistics for Manuscript ---
    print(f"\n--- Generating Cohort Summary Statistics ({side_prefix}) ---")
    try:
        summary_cohort_df = df.copy()
        IMAGING_COL_SUMMARY = TARGET_IMAGING_COL
        SEX_COL = 'Sex'

        num_hand_observations = summary_cohort_df.shape[0]
        num_unique_patients = summary_cohort_df['Patient ID'].nunique()
        mean_age = summary_cohort_df[AGE_COL].mean()
        std_age = summary_cohort_df[AGE_COL].std()

        percent_male = np.nan
        if SEX_COL in summary_cohort_df.columns:
            male_count = summary_cohort_df[SEX_COL].astype(str).str.strip().str.lower().isin(['m', 'male', '1']).sum()
            total_with_sex = summary_cohort_df[SEX_COL].notna().sum()
            if total_with_sex > 0:
                percent_male = (male_count / total_with_sex) * 100
        else:
            print(f"[INFO] '{SEX_COL}' column not found. Percentage Male will be 'NaN'.")

        mean_z, std_z, min_z, max_z = np.nan, np.nan, np.nan, np.nan
        if IMAGING_COL_SUMMARY in summary_cohort_df.columns:
            imaging_data = summary_cohort_df[IMAGING_COL_SUMMARY].dropna()
            mean_z = imaging_data.mean()
            std_z = imaging_data.std()
            min_z = imaging_data.min()
            max_z = imaging_data.max()
        else:
             print(f"[WARNING] Imaging column '{IMAGING_COL_SUMMARY}' not found. Imaging stats will be 'NaN'.")

        summary_data = {
            'Metric': [
                'Number of unique patients', 'Number of hand observations (N)', 'Mean Age (years)',
                'Std Dev Age (years)', 'Percentage Male (%)', f'Mean {side_prefix} Putamen Z-score',
                f'Std Dev {side_prefix} Putamen Z-score', f'Min {side_prefix} Putamen Z-score', f'Max {side_prefix} Putamen Z-score'
            ],
            'Value': [
                num_unique_patients, num_hand_observations, mean_age, std_age, percent_male,
                mean_z, std_z, min_z, max_z
            ]
        }
        summary_output_df = pd.DataFrame(summary_data)
        summary_output_df['Value'] = summary_output_df['Value'].round(2)

        summary_csv_path = os.path.join(data_output_folder, f"cohort_summary_statistics_{side_prefix.lower()}.csv")
        summary_output_df.to_csv(summary_csv_path, index=False, sep=';', decimal='.')
        print(f"[SUCCESS] Cohort summary for {side_prefix} side saved to: {summary_csv_path}")

    except Exception as e:
        print(f"[ERROR] Could not generate cohort summary statistics for {side_prefix} side. Reason: {e}")
    print(f"--- Finished Cohort Summary Generation ({side_prefix}) ---\n")

    SIGNIFICANCE_ALPHA = 0.05
    TOP_N_FOR_SCATTER_PANEL = 3

    all_significant_bivariate_results_dfs = {}
    all_raw_bivariate_results_list = []

    print(f"\n=== Starting Bivariate Correlation Analysis (Target: {TARGET_IMAGING_COL}) ===")
    if TARGET_IMAGING_COL not in df.columns:
        print(f"[FATAL ERROR] Target imaging column '{TARGET_IMAGING_COL}' not found. Skipping analysis for {side_prefix} side.")
        continue
    else:
        from statsmodels.stats.multitest import multipletests

        for task_prefix in tasks:
            print(f"\n--- Analyzing Task: {task_prefix.upper()} ({side_prefix}) ---")
            task_results = []
            for base_col_name in base_kinematic_cols:
                kinematic_col_name = f"{task_prefix}_{base_col_name}"
                if kinematic_col_name not in df.columns: continue

                cols_for_corr = [kinematic_col_name, TARGET_IMAGING_COL]
                if CONTROL_FOR_AGE:
                    cols_for_corr.append(AGE_COL)

                pair_data = df[cols_for_corr].dropna()
                if len(pair_data) < 5: continue

                if CONTROL_FOR_AGE:
                    pcorr = pg.partial_corr(data=pair_data, x=kinematic_col_name, y=TARGET_IMAGING_COL, covar=AGE_COL)
                    corr_coef = pcorr['r'].iloc[0]
                    p_value = pcorr['p-val'].iloc[0]
                    corr_type = 'Partial'
                else:
                    from scipy.stats import pearsonr
                    corr_coef, p_value = pearsonr(pair_data[kinematic_col_name], pair_data[TARGET_IMAGING_COL])
                    corr_type = 'Pearson'

                if pd.notna(corr_coef):
                    result_dict = {
                        "Task": task_prefix, "Kinematic Variable": kinematic_col_name, "Base Kinematic": base_col_name,
                        f"{corr_type} Correlation (r)": corr_coef, "P-value (uncorrected)": p_value, "N": len(pair_data)
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
                print(f"Found {len(significant_results_df)} significant age-controlled correlations for {task_prefix.upper()} with {side_prefix} Putamen.")
            else:
                print(f"No valid correlations could be computed for task {task_prefix.upper()}.")
                all_significant_bivariate_results_dfs[task_prefix] = pd.DataFrame()

    all_raw_bivariate_results_df = pd.DataFrame(all_raw_bivariate_results_list)
    combined_significant_bivariate_df = pd.DataFrame()
    if any(not df.empty for df in all_significant_bivariate_results_dfs.values()):
        combined_significant_bivariate_df = pd.concat(
            [df for df in all_significant_bivariate_results_dfs.values() if not df.empty],
            ignore_index=True
        ).sort_values(by=['Task', 'Q-value (FDR corrected)'])

    corr_col_name = "Partial Correlation (r)" if CONTROL_FOR_AGE else "Pearson Correlation (r)"
    if corr_col_name in combined_significant_bivariate_df.columns:
        combined_significant_bivariate_df.rename(columns={corr_col_name: "Correlation (r)"}, inplace=True)
    if corr_col_name in all_raw_bivariate_results_df.columns:
        all_raw_bivariate_results_df.rename(columns={corr_col_name: "Correlation (r)"}, inplace=True)

    print(f"\n--- Saving Bivariate Correlation Results ({side_prefix}) ---")
    try:
        raw_results_path = os.path.join(data_output_folder, f"all_raw_bivariate_results_{side_prefix.lower()}.csv")
        all_raw_bivariate_results_df.to_csv(raw_results_path, index=False, sep=';', decimal='.')
        print(f"[SUCCESS] All raw results for {side_prefix} saved to: {raw_results_path}")

        if not combined_significant_bivariate_df.empty:
            sig_results_path = os.path.join(data_output_folder, f"all_significant_bivariate_results_{side_prefix.lower()}.csv")
            combined_significant_bivariate_df.to_csv(sig_results_path, index=False, sep=';', decimal='.')
            print(f"[SUCCESS] Significant results for {side_prefix} saved to: {sig_results_path}")
        else:
            print(f"[INFO] No significant results to save for {side_prefix} side.")
    except Exception as e:
        print(f"[ERROR] Failed to save result CSV files for {side_prefix} side. Reason: {e}")
    print(f"--- Finished Saving Results ({side_prefix}) ---\n")

    print(f"\n--- [INFO] Checking prerequisites for generating Figure 1 for {side_prefix} side ---")
    if not PLOT_AVAILABLE:
        print(f"[INFO] RESULT: SKIPPING FIGURE for {side_prefix} because PLOT_AVAILABLE is False.")
    elif combined_significant_bivariate_df.empty:
        print(f"[INFO] RESULT: SKIPPING FIGURE for {side_prefix} because no statistically significant results were found.")
    else:
        print(f"[INFO] RESULT: Prerequisites MET. Found {len(combined_significant_bivariate_df)} significant results to plot. Proceeding.")
        try:
            print(f"\n\n--- Generating Figure 1: Bivariate Findings Summary ({side_prefix} side) ---")

            fig = plt.figure(figsize=(16, 14))
            gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2], hspace=0.4)
            ax_A, ax_B = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
            gs_C = gridspec.GridSpecFromSubplotSpec(2, TOP_N_FOR_SCATTER_PANEL, subplot_spec=gs[1, :], hspace=0.5, wspace=0.3)
            axes_C = [[fig.add_subplot(gs_C[i, j]) for j in range(TOP_N_FOR_SCATTER_PANEL)] for i in range(2)]

            # Panel A
            panel_A_data = combined_significant_bivariate_df.copy()
            panel_A_data['abs_r'] = panel_A_data['Correlation (r)'].abs()
            sort_order = panel_A_data.groupby('Base Kinematic')['abs_r'].mean().sort_values().index
            panel_A_data_pivot = panel_A_data.pivot_table(index='Base Kinematic', columns='Task', values='Correlation (r)').reindex(sort_order)
            y_labels_readable = [get_base_readable_name(name) for name in panel_A_data_pivot.index]
            y_pos = np.arange(len(y_labels_readable))
            for i, base_kin in enumerate(panel_A_data_pivot.index):
                r_ft, r_hm = panel_A_data_pivot.loc[base_kin, 'ft'], panel_A_data_pivot.loc[base_kin, 'hm']
                if pd.notna(r_ft) and pd.notna(r_hm): ax_A.plot([r_ft, r_hm], [y_pos[i], y_pos[i]], color='grey', alpha=0.5, linewidth=1)
            ax_A.scatter(panel_A_data_pivot['ft'], y_pos, color='#007acc', zorder=3, label='Finger Tapping (FT)')
            ax_A.scatter(panel_A_data_pivot['hm'], y_pos, color='#e85f5f', zorder=3, label='Hand Movements (HM)')
            ax_A.axvline(0, color='black', linestyle='--', linewidth=0.8)
            ax_A.set_yticks(y_pos); ax_A.set_yticklabels(y_labels_readable, fontsize=10)
            ax_A.set_xlabel("Partial Correlation Coefficient (r), controlling for Age", fontsize=11)
            ax_A.set_title(f"A) Significant Kinematic-{side_prefix} Striatal Correlations", fontsize=13, weight='bold', loc='left')
            ax_A.legend(loc='lower right'); ax_A.grid(axis='x', linestyle=':', alpha=0.6); sns.despine(ax=ax_A, left=True)

            # Panel B
            panel_B_data = all_raw_bivariate_results_df.pivot_table(index='Base Kinematic', columns='Task', values='Correlation (r)').dropna()
            sig_ft = combined_significant_bivariate_df[combined_significant_bivariate_df['Task']=='ft']['Base Kinematic'].unique()
            sig_hm = combined_significant_bivariate_df[combined_significant_bivariate_df['Task']=='hm']['Base Kinematic'].unique()
            def get_sig_status_revised(base_kin):
                in_ft, in_hm = base_kin in sig_ft, base_kin in sig_hm
                if in_ft and in_hm: return "Significant: Both Tasks"
                if in_ft: return "Significant: Finger Tapping"
                if in_hm: return "Significant: Hand Movements"
                return "Not Significant"
            panel_B_data['Significance'] = panel_B_data.index.map(get_sig_status_revised)
            palette_map = {'Significant: Both Tasks': 'black', 'Significant: Finger Tapping': '#007acc', 'Significant: Hand Movements': '#e85f5f', 'Not Significant': 'grey'}
            marker_map = {'Significant: Both Tasks': 'o', 'Significant: Finger Tapping': 's', 'Significant: Hand Movements': '^', 'Not Significant': 'o'}
            size_map = {'Significant: Both Tasks': 80, 'Significant: Finger Tapping': 60, 'Significant: Hand Movements': 60, 'Not Significant': 30}
            panel_B_data['PointSize'] = panel_B_data['Significance'].map(size_map)
            sns.regplot(data=panel_B_data, x='ft', y='hm', ax=ax_B, scatter=False, color='black', line_kws={'linewidth': 2, 'alpha': 0.8, 'zorder': 2})
            sns.scatterplot(data=panel_B_data, x='ft', y='hm', hue='Significance', style='Significance', palette=palette_map, markers=marker_map, size='PointSize', sizes=(30, 100), ax=ax_B, zorder=3)
            min_val, max_val = panel_B_data[['ft', 'hm']].min().min(), panel_B_data[['ft', 'hm']].max().max()
            padding = 0.1
            ax_B.set_xlim(min_val - padding, max_val + padding); ax_B.set_ylim(min_val - padding, max_val + padding)
            ax_B.set_xlabel("Partial Correlation Strength (r) in FT (Age Controlled)", fontsize=11)
            ax_B.set_ylabel("Partial Correlation Strength (r) in HM (Age Controlled)", fontsize=11)
            ax_B.set_title(f"B) Concordance of {side_prefix} Correlations", fontsize=13, weight='bold', loc='left')
            ax_B.set_aspect('equal', adjustable='box'); ax_B.grid(True, linestyle=':', alpha=0.6)
            concordance_r, concordance_p = pg.corr(panel_B_data['ft'], panel_B_data['hm'])['r'].iloc[0], pg.corr(panel_B_data['ft'], panel_B_data['hm'])['p-val'].iloc[0]
            p_text = "p < 0.001" if concordance_p < 0.001 else f"p = {concordance_p:.3f}"
            ax_B.text(0.05, 0.95, f"Concordance:\nr = {concordance_r:.2f}\n{p_text}", transform=ax_B.transAxes, fontsize=10, va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='whitesmoke', alpha=0.8))
            handles, labels = ax_B.get_legend_handles_labels()
            sig_labels_indices = [i for i, label in enumerate(labels) if label in size_map.keys()]
            ax_B.legend([handles[i] for i in sig_labels_indices], [labels[i] for i in sig_labels_indices], title="Significance (q<0.05)", fontsize=8)

            # --- Panel C: Top Scatter Plots of Residuals ---
            fig.text(0.5, 0.5, "C) Top Age-Controlled Kinematic Correlates of Dopaminergic Deficit", ha='center', fontsize=14, weight='bold')
            for i, task_prefix in enumerate(tasks):
                task_df_sig = all_significant_bivariate_results_dfs.get(task_prefix, pd.DataFrame())
                if task_df_sig.empty: continue
                
                # --- FIXED: Use the correct column name variable 'corr_col_name' ---
                task_df_sig['abs_r'] = task_df_sig[corr_col_name].abs()
                top_results = task_df_sig.sort_values(by=['Q-value (FDR corrected)', 'abs_r'], ascending=[True, False]).head(TOP_N_FOR_SCATTER_PANEL)
                
                for j, (_, row) in enumerate(top_results.iterrows()):
                    ax_c = axes_C[i][j]
                    kin_col = row['Kinematic Variable']
                    # --- FIXED: Use the correct column name variable 'corr_col_name' ---
                    stats = {'r': row[corr_col_name], 'q': row['Q-value (FDR corrected)'], 'N': row['N']}

                    plot_data = df[[kin_col, TARGET_IMAGING_COL, AGE_COL]].dropna()
                    kin_model = ols(f"{kin_col} ~ {AGE_COL}", data=plot_data).fit()
                    plot_data['kin_resid'] = kin_model.resid
                    img_model = ols(f"{TARGET_IMAGING_COL} ~ {AGE_COL}", data=plot_data).fit()
                    plot_data['img_resid'] = img_model.resid
                    
                    sns.regplot(x='kin_resid', y='img_resid', data=plot_data, ax=ax_c, scatter_kws={'alpha': 0.5, 'edgecolor': 'k', 'linewidths': 0.8}, line_kws={'color': 'k', 'linestyle': '--'})
                    
                    ax_c.set_xlabel(f"{get_readable_name(kin_col)}\n(Residuals vs. Age)", fontsize=9)
                    ax_c.set_ylabel(f"{get_readable_name(TARGET_IMAGING_COL)}\n(Residuals vs. Age)", fontsize=9)
                    task_label = "Finger Tapping" if task_prefix.upper() == 'FT' else "Hand Movements"
                    ax_c.set_title(f"{task_label}: {get_base_readable_name(kin_col)}", fontsize=10, weight='bold')

                    stats_text = f"Partial r = {stats['r']:.2f}\nq = {stats['q']:.4f}\nN = {stats['N']}"
                    ax_c.text(0.05, 0.95, stats_text, transform=ax_c.transAxes, fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='whitesmoke', alpha=0.8))

            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.suptitle(f'Figure 1: Age-Controlled Kinematic Correlates of {side_prefix} Striatal Dopamine Deficit (OFF State)', fontsize=16, weight='bold')

            figure_1_filename = os.path.join(plots_folder, f"Figure1_Bivariate_Findings_{side_prefix}_Summary_AgeControlled.png")
            plt.savefig(figure_1_filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"\n--- SUCCESS! Age-controlled Figure 1 for {side_prefix} side saved to: {os.path.abspath(figure_1_filename)} ---")

        except Exception as e:
            print("\n" + "!"*60)
            print(f"!!! AN UNEXPECTED ERROR OCCURRED DURING FIGURE 1 ({side_prefix}) GENERATION !!!")
            print(f"!!! Error Type: {type(e).__name__}")
            print(f"!!! Error Message: {e}")
            print("!!! Printing detailed traceback:")
            traceback.print_exc()
            print("!"*60 + "\n")

print("\n--- Bivariate analysis script execution finished ---")