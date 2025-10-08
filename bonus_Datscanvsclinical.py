# --- START OF SCRIPT: IV_datnik_clinical_validation.py (FINAL, with professional q-value formatting) ---

import os
import pandas as pd
import numpy as np
import pingouin as pg
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import sys

print("\n" + "="*80)
print("--- RUNNING CLINICAL VALIDATION ANALYSIS ---")
print("="*80 + "\n")

# --- 1. Load and Merge Data (with corrected path logic) ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

project_root_dir = os.path.dirname(script_dir)

processed_data_dir = os.path.join(project_root_dir, "Output", "Data_Processed")
data_output_folder = os.path.join(project_root_dir, "Output", "Data")
plots_folder = os.path.join(project_root_dir, "Output", "Plots")
os.makedirs(data_output_folder, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)

merged_csv_file = os.path.join(processed_data_dir, "final_merged_data.csv")
try:
    df_kinematic = pd.read_csv(merged_csv_file, sep=';', decimal='.')
except FileNotFoundError:
    try:
        df_kinematic = pd.read_csv(merged_csv_file, sep=',', decimal='.')
    except FileNotFoundError:
        print(f"[FATAL ERROR] Kinematic data file not found at: {merged_csv_file}")
        print("Please ensure your project structure is correct (e.g., /Project/Input, /Project/Output, /Project/Scripts).")
        sys.exit()

clinical_csv_file = os.path.join(project_root_dir, "Input", "Clinical_imput_an.csv")
try:
    df_clinical = pd.read_csv(clinical_csv_file, sep=';', decimal=',')
except FileNotFoundError:
    print(f"[FATAL ERROR] Clinical data file not found at: {clinical_csv_file}")
    sys.exit()

df_full = pd.merge(df_kinematic, df_clinical, left_on='Patient ID', right_on='No.', how='left')

# --- 2. Prepare the Data ---
df = df_full[df_full['Medication Condition'].str.lower() == 'off'].copy()
if 'Age_y' in df.columns and 'Age_x' in df.columns:
    df['Age'] = df['Age_y'].fillna(df['Age_x'])
elif 'Age_y' in df.columns: df.rename(columns={'Age_y': 'Age'}, inplace=True)
elif 'Age_x' in df.columns: df.rename(columns={'Age_x': 'Age'}, inplace=True)
else:
    print("[FATAL ERROR] No 'Age' column found.")
    sys.exit()

clinical_rating_cols = ['MedOFF ft left', 'MedOFF ft right', 'MedOFF hm left', 'MedOFF hm right', 'UPDRS', 'H&Y']
df.replace(['NA', 'ND'], np.nan, inplace=True)
for col in clinical_rating_cols:
    if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

df['Clinical_Score'] = np.nan
mask_ft_l = (df['Hand_Performed'] == 'Left') & df['ft_rate'].notna()
mask_ft_r = (df['Hand_Performed'] == 'Right') & df['ft_rate'].notna()
df.loc[mask_ft_l, 'Clinical_Score'] = df.loc[mask_ft_l, 'MedOFF ft left']
df.loc[mask_ft_r, 'Clinical_Score'] = df.loc[mask_ft_r, 'MedOFF ft right']
mask_hm_l = (df['Hand_Performed'] == 'Left') & df['hm_rate'].notna()
mask_hm_r = (df['Hand_Performed'] == 'Right') & df['hm_rate'].notna()
df.loc[mask_hm_l, 'Clinical_Score'] = df.loc[mask_hm_l, 'MedOFF hm left']
df.loc[mask_hm_r, 'Clinical_Score'] = df.loc[mask_hm_r, 'MedOFF hm right']

# --- 3. Run the "Sweeping" Correlation Analysis ---
base_kinematic_cols = [
    "meanamplitude", "stdamplitude", "meanspeed", "stdspeed", "meanrmsvelocity",
    "stdrmsvelocity", "meanopeningspeed", "stdopeningspeed", "meanclosingspeed",
    "stdclosingspeed", "meancycleduration", "stdcycleduration", "rangecycleduration",
    "rate", "amplitudedecay", "velocitydecay", "ratedecay", "cvamplitude",
    "cvcycleduration", "cvspeed", "cvrmsvelocity", "cvopeningspeed", "cvclosingspeed"
]
tasks = ['ft', 'hm']; all_results = []; SIGNIFICANCE_ALPHA = 0.05
for task in tasks:
    task_df = df[df[f'{task}_rate'].notna()].copy()
    for base_kin in base_kinematic_cols:
        kin_col = f"{task}_{base_kin}"
        if kin_col not in task_df.columns: continue
        analysis_data = task_df[[kin_col, 'Clinical_Score', 'Age']].dropna()
        if len(analysis_data) < 10: continue
        pcorr = pg.partial_corr(data=analysis_data, x=kin_col, y='Clinical_Score', covar='Age').iloc[0]
        all_results.append({'Task': task.upper(), 'Kinematic_Variable': kin_col, 'Base_Kinematic': base_kin,
                            'Correlation_r': pcorr['r'], 'P_Value': pcorr['p-val'], 'N': pcorr['n']})

# --- 4. Post-Process and Save Results ---
results_df = pd.DataFrame(all_results)
if not results_df.empty:
    name_map = {
        'meanamplitude': 'MeanAmplitude', 'stdamplitude': 'StdAmplitude', 'meanspeed': 'MeanSpeed',
        'stdspeed': 'StdSpeed', 'meanrmsvelocity': 'MeanRMSVelocity', 'stdrmsvelocity': 'StdRMSVelocity',
        'meanopeningspeed': 'MeanOpeningSpeed', 'stdopeningspeed': 'StdOpeningSpeed',
        'meanclosingspeed': 'MeanClosingSpeed', 'stdclosingspeed': 'StdClosingSpeed',
        'meancycleduration': 'MeanCycleDuration', 'stdcycleduration': 'StdCycleDuration',
        'rangecycleduration': 'RangeCycleDuration', 'rate': 'Rate', 'amplitudedecay': 'AmplitudeDecay',
        'velocitydecay': 'VelocityDecay', 'ratedecay': 'RateDecay', 'cvamplitude': 'CV_Amplitude',
        'cvcycleduration': 'CV_CycleDuration', 'cvspeed': 'CV_Speed', 'cvrmsvelocity': 'CV_RMSVelocity',
        'cvopeningspeed': 'CV_OpeningSpeed', 'cvclosingspeed': 'CV_ClosingSpeed'
    }
    results_df['Feature_Name'] = results_df['Base_Kinematic'].map(name_map)
    reject, q_values, _, _ = multipletests(results_df['P_Value'], alpha=SIGNIFICANCE_ALPHA, method='fdr_bh')
    results_df['Q_Value_FDR'] = q_values
    results_df['Significant_FDR'] = reject
    significant_df = results_df[results_df['Significant_FDR']].sort_values(by=['Task', 'Q_Value_FDR'])

# --- 5. Generate Final Polished Supplementary Figure ---
print("\n--- Generating Final Polished Supplementary Figure ---")
if 'significant_df' in locals() and not significant_df.empty:
    try:
        TOP_N_SCATTERS = 3; fig = plt.figure(figsize=(16, 16))
        gs = gridspec.GridSpec(3, TOP_N_SCATTERS, figure=fig, hspace=0.5, wspace=0.3)
        ax_A = fig.add_subplot(gs[0, :]); panel_A_data = significant_df.copy()
        panel_A_data['abs_r'] = panel_A_data['Correlation_r'].abs()
        sort_order = panel_A_data.groupby('Feature_Name')['abs_r'].mean().sort_values(ascending=True).index
        panel_A_pivot = panel_A_data.pivot_table(index='Feature_Name', columns='Task', values='Correlation_r').reindex(sort_order)
        y_labels = panel_A_pivot.index; y_pos = np.arange(len(y_labels))
        for i, feature_name in enumerate(panel_A_pivot.index):
            r_ft = panel_A_pivot.loc[feature_name, 'FT']; r_hm = panel_A_pivot.loc[feature_name, 'HM']
            if pd.notna(r_ft) and pd.notna(r_hm): ax_A.plot([r_ft, r_hm], [y_pos[i], y_pos[i]], color='grey', alpha=0.7, linewidth=1.5, zorder=1)
        ax_A.scatter(panel_A_pivot['FT'], y_pos, color='#007acc', s=80, zorder=2, label='Finger Tapping (FT)')
        ax_A.scatter(panel_A_pivot['HM'], y_pos, color='#e85f5f', s=80, zorder=2, marker='s', label='Hand Movements (HM)')
        ax_A.axvline(0, color='black', linestyle='--', linewidth=1); ax_A.set_yticks(y_pos); ax_A.set_yticklabels(y_labels, fontsize=11)
        ax_A.set_xlabel("Partial Correlation (r) with Clinical Score (Age Controlled)", fontsize=12); ax_A.set_title("A) Significant Kinematic-Clinical Correlations", fontsize=14, weight='bold', loc='left')
        ax_A.legend(); ax_A.grid(axis='y', linestyle=':', alpha=0.6); ax_A.set_xlim(-0.7, 0.7); sns.despine(ax=ax_A, left=True)
        ax_title_b = fig.add_subplot(gs[1, :]); ax_title_b.set_title("B) Top Finger Tapping Correlates of Clinical Impairment", fontsize=14, weight='bold', pad=20)
        ax_title_b.set_frame_on(False); ax_title_b.get_xaxis().set_visible(False); ax_title_b.get_yaxis().set_visible(False)
        ax_title_c = fig.add_subplot(gs[2, :]); ax_title_c.set_title("C) Top Hand Movement Correlates of Clinical Impairment", fontsize=14, weight='bold', pad=20)
        ax_title_c.set_frame_on(False); ax_title_c.get_xaxis().set_visible(False); ax_title_c.get_yaxis().set_visible(False)
        axes_B = [fig.add_subplot(gs[1, i]) for i in range(TOP_N_SCATTERS)]; axes_C = [fig.add_subplot(gs[2, i]) for i in range(TOP_N_SCATTERS)]
        task_info = {'FT': {'axes': axes_B, 'color': '#007acc'}, 'HM': {'axes': axes_C, 'color': '#e85f5f'}}
        for task, info in task_info.items():
            task_df_sig = significant_df[significant_df['Task'] == task].copy(); task_df_sig['abs_r'] = task_df_sig['Correlation_r'].abs()
            top_results = task_df_sig.sort_values(by=['Q_Value_FDR', 'abs_r'], ascending=[True, False]).head(TOP_N_SCATTERS)
            for j, (_, row) in enumerate(top_results.iterrows()):
                ax = info['axes'][j]; kin_col = row['Kinematic_Variable']; feature_name = row['Feature_Name']
                plot_data = df[[kin_col, 'Clinical_Score']].dropna()
                sns.regplot(x=kin_col, y='Clinical_Score', data=plot_data, ax=ax, scatter_kws={'alpha': 0.6, 'edgecolor': 'w', 'linewidths': 0.5, 'color': info['color']}, line_kws={'color': 'black', 'linestyle': '--'})
                ax.set_title(feature_name, fontsize=12, weight='bold'); ax.set_xlabel("Kinematic Feature Value", fontsize=10); ax.set_ylabel("Clinical Score (0-4)", fontsize=10)
                ax.set_ylim(-0.2, df['Clinical_Score'].dropna().max() + 0.2)
                
                # ### --- FINAL TWEAK: Professional formatting for q-values --- ###
                if row['Q_Value_FDR'] < 0.001:
                    q_text = "q < 0.001"
                else:
                    q_text = f"q = {row['Q_Value_FDR']:.3f}"
                
                stats_text = f"Partial r = {row['Correlation_r']:.2f}\n{q_text}"
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='whitesmoke', alpha=0.8))

        fig.suptitle('Supplementary Figure 1: Clinical Validation of Kinematic Features Against Expert Ratings', fontsize=18, weight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        sfig_path = os.path.join(plots_folder, "SFigure1_Clinical_Validation_Summary_FINAL.png")
        plt.savefig(sfig_path, dpi=300, bbox_inches='tight')
        print(f"\n[SUCCESS] Final polished Supplementary Figure saved to: {sfig_path}")
    except Exception as e:
        import traceback; print(f"\n[ERROR] Could not generate Supplementary Figure. Reason: {e}"); traceback.print_exc()
else:
    print("\n[INFO] Skipping Supplementary Figure generation as no significant results were found.")

print("\n--- Clinical validation script finished ---")