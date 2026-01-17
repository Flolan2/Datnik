# -*- coding: utf-8 -*-
"""
Plotting functions for visualizing experiment results.
UPDATED: Publication-ready styling (Large fonts, Colorblind-friendly, High contrast).
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import matplotlib.gridspec as gridspec
import traceback
import logging
import matplotlib.patheffects as pe

logger = logging.getLogger('DatnikExperiment')

# Map variable names to readable labels
VARIABLE_NAMES_MAP = {
    'meanamplitude': 'Mean Amplitude', 'stdamplitude': 'SD of Amplitude',
    'meanrmsvelocity': 'Mean RMS Velocity', 'stdrmsvelocity': 'SD of RMS Velocity',
    'meanopeningspeed': 'Mean Opening Speed', 'stdopeningspeed': 'SD of Opening Speed',
    'meanclosingspeed': 'Mean Closing Speed', 'stdclosingspeed': 'SD of Closing Speed',
    'meancycleduration': 'Mean Cycle Duration', 'stdcycleduration': 'SD of Cycle Duration',
    'rangecycleduration': 'Range of Cycle Duration', 'amplitudedecay': 'Amplitude Decay',
    'velocitydecay': 'Velocity Decay', 'ratedecay': 'Rate Decay',
    'cvamplitude': 'CV of Amplitude', 'cvcycleduration': 'CV of Cycle Duration',
    'cvrmsvelocity': 'CV of RMS Velocity', 'cvopeningspeed': 'CV of Opening Speed',
    'cvclosingspeed': 'CV of Closing Speed', 'rate': 'Frequency',
    'meanspeed': 'Mean Speed', 'stdspeed': 'SD of Speed', 'cvspeed': 'CV of Speed'
}


def apply_pub_style():
    """Sets consistent large fonts and styling for journal submission."""
    sns.set_context("paper", font_scale=1.6)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 2.5,
        'figure.titlesize': 20,
        'figure.dpi': 300,
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.2
    })

def plot_figure2(summary_df, roc_data, importances_data, output_folder, config):
    """
    Generates the publication-ready 2-panel Prediction Figure (ROC + Feature Sig).
    Labeled as Figure 2.
    """
    try:
        apply_pub_style()

        CONFIG_FT = 'LR_RFE15_FT_OriginalFeats'
        CONFIG_HM = 'LR_RFE15_HM_OriginalFeats'
        COLOR_FT = '#332288'  # Indigo
        COLOR_HM = '#117733'  # Green

        fig = plt.figure(figsize=(18, 8))

        gs = gridspec.GridSpec(
            1, 2,
            figure=fig,
            left=0.06,
            right=0.98,
            bottom=0.08,
            top=0.90,    
            wspace=0.35  
        )

        ax_A = fig.add_subplot(gs[0, 0])
        ax_B = fig.add_subplot(gs[0, 1])

        # --- Panel A: ROC Curves ---
        ax_A.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Chance (AUC = 0.50)')

        for plot_cfg in [
            {'name': CONFIG_FT, 'label': 'Finger Tapping', 'color': COLOR_FT},
            {'name': CONFIG_HM, 'label': 'Hand Movement', 'color': COLOR_HM}
        ]:
            cfg_name = plot_cfg['name']
            roc_runs = roc_data.get(cfg_name, [])
            metrics_row = summary_df[summary_df['Config_Name'] == cfg_name]

            if roc_runs and not metrics_row.empty:
                metrics_row = metrics_row.iloc[0]
                base_fpr = np.linspace(0, 1, 101)
                tprs_interp = [np.interp(base_fpr, fpr, tpr) for fpr, tpr in roc_runs if len(fpr) > 1]

                if tprs_interp:
                    mean_tprs = np.mean(tprs_interp, axis=0)
                    std_tprs = np.std(tprs_interp, axis=0)
                    tprs_upper = np.minimum(mean_tprs + std_tprs, 1)
                    tprs_lower = np.maximum(mean_tprs - std_tprs, 0)

                    auc_val = metrics_row['Mean_ROC_AUC']
                    auc_std = metrics_row['Std_ROC_AUC']
                    auc_label = f"{plot_cfg['label']}\nAUC = {auc_val:.2f} ± {auc_std:.2f}"

                    ax_A.plot(base_fpr, mean_tprs, label=auc_label, color=plot_cfg['color'], lw=3)
                    ax_A.fill_between(base_fpr, tprs_lower, tprs_upper, color=plot_cfg['color'], alpha=0.15)

        ax_A.set_title('A) Prediction Performance', fontsize=16, weight='bold', loc='left')
        ax_A.set_xlabel('False Positive Rate', weight='bold')
        ax_A.set_ylabel('True Positive Rate', weight='bold')
        ax_A.legend(loc='lower right', fontsize=13)
        ax_A.set_aspect('equal', adjustable='box')

        # --- Panel B: Feature Importance (Finger Tapping signature) ---
        ft_coeffs_df = importances_data.get('group', {}).get(CONFIG_FT, {}).get('ft')
        if ft_coeffs_df is not None and not ft_coeffs_df.empty:
            plot_df = (
                ft_coeffs_df
                .reindex(ft_coeffs_df['Mean_Importance'].abs().sort_values(ascending=False).index)
                .head(config.PLOT_TOP_N_FEATURES)
                .sort_values('Mean_Importance', ascending=False)
            )

            base_feature_names = plot_df.index.to_series().str.replace('ft_', '', regex=False)
            plot_df = plot_df.copy()
            plot_df['Readable_Feature'] = base_feature_names.map(VARIABLE_NAMES_MAP).fillna(base_feature_names)


            sns.barplot(
                x='Mean_Importance',
                y='Readable_Feature',
                data=plot_df,
                ax=ax_B,
                edgecolor='black'
            )

            colors = ['#D55E00' if c < 0 else '#0072B2' for c in plot_df['Mean_Importance']]
            for patch, c in zip(ax_B.patches, colors):
                patch.set_facecolor(c)

            ax_B.axvline(0, color='black', linewidth=1.5, linestyle='--')
            ax_B.set_title('B) Kinematic Signature (Finger Tapping)', fontsize=16, weight='bold', loc='left')
            ax_B.set_xlabel('Mean Coefficient (Log-Odds)', weight='bold')
            ax_B.set_ylabel('')
            ax_B.tick_params(axis='y', pad=6)  # small readability boost
        else:
            ax_B.text(0.5, 0.5, "No coefficient data.", ha='center', va='center', transform=ax_B.transAxes)
            ax_B.set_title('B) Kinematic Signature (Finger Tapping)', fontsize=16, weight='bold', loc='left')
            ax_B.set_xlabel('Mean Coefficient (Log-Odds)', weight='bold')
            ax_B.set_ylabel('')

        fig.suptitle('Figure 2: Prediction of Dopaminergic Deficit', fontsize=20, weight='bold', y=0.98)

        figure_filename = os.path.join(output_folder, "Figure 2.pdf")
        plt.savefig(figure_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"--- SUCCESS! Figure 2 saved to: {figure_filename} ---")

    except Exception as e:
        logger.error(f"Error generating Figure 2: {e}")
        traceback.print_exc()


def plot_performance_vs_threshold(summary_df, x_col, y_col, filename, config):
    """
    Generates sensitivity analysis plot.
    """
    apply_pub_style()

    std_col = y_col.replace('Mean_', 'Std_')

    if not {x_col, y_col, std_col, 'Config_Name'}.issubset(summary_df.columns):
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    configs = sorted(summary_df['Config_Name'].unique())
    color_map = {
        'LR_RFE15_FT_OriginalFeats': '#332288',
        'LR_RFE15_HM_OriginalFeats': '#117733',
    }

    for cfg_name in configs:
        df_cfg = summary_df[summary_df['Config_Name'] == cfg_name].copy()
        df_cfg = df_cfg.sort_values(by=x_col).dropna(subset=[x_col, y_col, std_col])

        if df_cfg.empty:
            continue

        x = df_cfg[x_col]
        y_mean = df_cfg[y_col]
        y_std = df_cfg[std_col]

        color = color_map.get(cfg_name, 'gray')
        label = cfg_name.replace('_', ' ').replace('OriginalFeats', '').replace('LR RFE15', '').strip()

        ax.plot(x, y_mean, marker='o', linestyle='-', label=label, color=color, markersize=8, lw=3)
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=0.15)

    ax.axhline(y=0.5, color='black', linestyle='--', label='Chance', lw=2)

    ax.set_title('Sensitivity Analysis: Performance vs. Threshold', fontsize=18, weight='bold', pad=15)
    ax.set_xlabel('DaTscan Z-Score Threshold (More Severe -->)', fontsize=14, weight='bold')
    ax.set_ylabel('Mean ROC AUC', fontsize=14, weight='bold')
    ax.legend(title='Model', loc='best', fontsize=12)
    ax.grid(True, which='both', linestyle=':', linewidth=1)

    ax.invert_xaxis()
    ax.set_ylim(0.40, 0.85)

    sns.despine()
    fig.tight_layout()

    try:
        plt.savefig(filename.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
        logger.info(f"Performance vs. Threshold plot saved to: {filename}")
    except Exception as e:
        logger.error(f"Error saving threshold plot: {e}")
    finally:
        plt.close(fig)
