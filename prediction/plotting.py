# -*- coding: utf-8 -*-
"""
Plotting functions for visualizing experiment results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import matplotlib.gridspec as gridspec
import traceback
import logging

logger = logging.getLogger('DatnikExperiment')

VARIABLE_NAMES_MAP = { 'meanamplitude': 'Mean Amplitude', 'stdamplitude': 'SD of Amplitude', 'meanrmsvelocity': 'Mean RMS Velocity', 'stdrmsvelocity': 'SD of RMS Velocity', 'meanopeningspeed': 'Mean Opening Speed', 'stdopeningspeed': 'SD of Opening Speed', 'meanclosingspeed': 'Mean Closing Speed', 'stdclosingspeed': 'SD of Closing Speed', 'meancycleduration': 'Mean Cycle Duration', 'stdcycleduration': 'SD of Cycle Duration', 'rangecycleduration': 'Range of Cycle Duration', 'amplitudedecay': 'Amplitude Decay', 'velocitydecay': 'Velocity Decay', 'ratedecay': 'Rate Decay', 'cvamplitude': 'CV of Amplitude', 'cvcycleduration': 'CV of Cycle Duration', 'cvrmsvelocity': 'CV of RMS Velocity', 'cvopeningspeed': 'CV of Opening Speed', 'cvclosingspeed': 'CV of Closing Speed', 'rate': 'Frequency', 'meanspeed': 'Mean Speed', 'stdspeed': 'SD of Speed', 'cvspeed': 'CV of Speed' }

def plot_figure3(summary_df, roc_data, importances_data, output_folder, config):
    """
    Generates the publication-ready 2-panel Figure 3.
    Panel A: ROC curves for FT vs. HM models.
    Panel B: Feature signature for the best-performing model.
    """
    try:
        CONFIG_FT = 'LR_RFE15_FT_OriginalFeats'
        CONFIG_HM = 'LR_RFE15_HM_OriginalFeats'
        COLOR_FT = 'indigo'
        COLOR_HM = 'mediumseagreen'

        fig = plt.figure(figsize=(18, 8))
        gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.4, wspace=0.3)
        ax_A = fig.add_subplot(gs[0, 0])
        ax_B = fig.add_subplot(gs[0, 1])
        
        # Panel A: Final Model Performance (ROC Curves)
        ax_A.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
        for plot_cfg in [{'name': CONFIG_FT, 'label': 'Finger Tapping', 'color': COLOR_FT}, {'name': CONFIG_HM, 'label': 'Hand Movement', 'color': COLOR_HM}]:
            cfg_name = plot_cfg['name']
            roc_runs = roc_data.get(cfg_name, [])
            metrics_row = summary_df[summary_df['Config_Name'] == cfg_name]
            if roc_runs and not metrics_row.empty:
                metrics_row = metrics_row.iloc[0]
                base_fpr = np.linspace(0, 1, 101); tprs_interp = [np.interp(base_fpr, fpr, tpr) for fpr, tpr in roc_runs if len(fpr) > 1]
                if tprs_interp:
                    mean_tprs = np.mean(tprs_interp, axis=0); std_tprs = np.std(tprs_interp, axis=0)
                    tprs_upper = np.minimum(mean_tprs + std_tprs, 1); tprs_lower = np.maximum(mean_tprs - std_tprs, 0)
                    auc_label = f"{plot_cfg['label']} (AUC = {metrics_row['Mean_ROC_AUC']:.3f} ± {metrics_row['Std_ROC_AUC']:.3f})"
                    ax_A.plot(base_fpr, mean_tprs, label=auc_label, color=plot_cfg['color'], lw=2)
                    ax_A.fill_between(base_fpr, tprs_lower, tprs_upper, color=plot_cfg['color'], alpha=0.15)
        ax_A.set_title(f'A) Final Model Performance (at Z < {config.ABNORMALITY_THRESHOLD})', fontsize=13, weight='bold', loc='left')
        ax_A.set_xlabel('False Positive Rate (1 - Specificity)'); ax_A.set_ylabel('True Positive Rate (Sensitivity)')
        ax_A.legend(loc='lower right'); ax_A.set_aspect('equal', adjustable='box')

        # Panel B: Predictive Kinematic Signature (Finger Tapping)
        ft_coeffs_df = importances_data.get('group', {}).get(CONFIG_FT, {}).get('ft')
        if ft_coeffs_df is not None and not ft_coeffs_df.empty:
            plot_df = ft_coeffs_df.reindex(ft_coeffs_df['Mean_Importance'].abs().sort_values(ascending=False).index).head(config.PLOT_TOP_N_FEATURES).sort_values('Mean_Importance', ascending=False)
            base_feature_names = plot_df.index.str.replace('ft_', ''); plot_df['Readable_Feature'] = base_feature_names.map(VARIABLE_NAMES_MAP)
            colors = ['crimson' if c < 0 else 'royalblue' for c in plot_df['Mean_Importance']]
            sns.barplot(x=plot_df['Mean_Importance'], y=plot_df['Readable_Feature'], palette=colors, ax=ax_B)
            ax_B.axvline(0, color='k', linewidth=0.8, linestyle='--')
            ax_B.set_title('B) Predictive Kinematic Signature (Finger Tapping)', fontsize=13, weight='bold', loc='left')
            ax_B.set_xlabel('Mean Coefficient (Log-Odds Change)\n<-- Lower Risk of Deficit | Higher Risk of Deficit -->')
            ax_B.set_ylabel('Kinematic Feature')
        else:
            ax_B.text(0.5, 0.5, "Coefficient data could not be generated.", ha='center', va='center'); ax_B.set_title('B) Predictive Kinematic Signature (Finger Tapping)', fontsize=13, weight='bold', loc='left')

        fig.suptitle(f'Figure 2: Prediction of Dopaminergic Deficit', fontsize=16, weight='bold')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        figure_3_filename = os.path.join(output_folder, f"Figure3_Prediction_Summary_FixedThreshold.png")
        plt.savefig(figure_3_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"\n--- SUCCESS! Final Figure 3 saved to: {os.path.abspath(figure_3_filename)} ---")
        
    except Exception as e:
        logger.error("\n" + "!"*60); logger.error("!!! AN UNEXPECTED ERROR OCCURRED DURING FIGURE 3 GENERATION !!!");
        logger.error(f"!!! Error Type: {type(e).__name__} at line {e.__traceback__.tb_lineno}"); logger.error(f"!!! Error Message: {e}");
        traceback.print_exc(); logger.error("!"*60 + "\n")


def plot_performance_vs_threshold(summary_df, x_col, y_col, filename, config):
    """
    Generates a line plot showing a performance metric vs. a changing threshold.
    Includes a shaded region for standard deviation.

    Args:
        summary_df (pd.DataFrame): DataFrame with experiment results from the sweep.
        x_col (str): Column name for the x-axis (e.g., 'Threshold_Value').
        y_col (str): Column name for the y-axis (e.g., 'Mean_ROC_AUC').
        filename (str): Full path to save the plot.
        config (module): The main configuration module.
    """
    std_col = y_col.replace('Mean_', 'Std_')
    required_cols = {x_col, y_col, std_col, 'Config_Name'}
    
    if not required_cols.issubset(summary_df.columns):
        missing = required_cols - set(summary_df.columns)
        logger.warning(f"Required columns for threshold plot not found: {missing}. Skipping plot '{filename}'.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    configs = sorted(summary_df['Config_Name'].unique())
    color_map = {
        'LR_RFE15_FT_OriginalFeats': 'indigo',
        'LR_RFE15_HM_OriginalFeats': 'mediumseagreen',
    }
    default_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(configs)))

    for i, cfg_name in enumerate(configs):
        df_cfg = summary_df[summary_df['Config_Name'] == cfg_name].copy()
        df_cfg = df_cfg.sort_values(by=x_col).dropna(subset=[x_col, y_col, std_col])

        if df_cfg.empty:
            continue

        x = df_cfg[x_col]
        y_mean = df_cfg[y_col]
        y_std = df_cfg[std_col]

        color = color_map.get(cfg_name, default_colors[i])
        label = cfg_name.replace('_', ' ').replace('OriginalFeats', '').strip()
        
        ax.plot(x, y_mean, marker='o', linestyle='-', label=label, color=color, markersize=5)
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=0.15)

    ax.axhline(y=0.5, color='black', linestyle='--', label='Chance Level')

    ax.set_title('Model Performance as a Function of Abnormality Threshold', fontsize=16, weight='bold', pad=20)
    ax.set_xlabel('DaTscan Z-Score Threshold (More Severe -->)', fontsize=12)
    ax.set_ylabel('Mean ROC AUC', fontsize=12)
    ax.legend(title='Model Configuration', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, which='both', linestyle=':', linewidth='0.5')
    
    ax.invert_xaxis()
    ax.set_ylim(0.45, 1.0)
    
    sns.despine()
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Performance vs. Threshold plot saved to: {filename}")
    except Exception as e:
        logger.error(f"Error saving performance vs. threshold plot {filename}: {e}")
    finally:
        plt.close(fig)