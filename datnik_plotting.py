import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Mapping for Readable Kinematic Names ---
READABLE_KINEMATIC_NAMES = {
    "meanamplitude": "Mean Amplitude",
    "stdamplitude": "STD Amplitude",
    "meanspeed": "Mean Speed",
    "stdspeed": "STD Speed",
    "meanrmsvelocity": "Mean RMS Velocity",
    "stdrmsvelocity": "STD RMS Velocity",
    "meanopeningspeed": "Mean Opening Speed",
    "stdopeningspeed": "STD Opening Speed",
    "meanclosingspeed": "Mean Closing Speed",
    "stdclosingspeed": "STD Closing Speed",
    "meancycleduration": "Mean Cycle Duration",
    "stdcycleduration": "STD Cycle Duration",
    "rangecycleduration": "Range Cycle Duration",
    "rate": "Rate",
    "amplitudedecay": "Amplitude Decay",
    "velocitydecay": "Velocity Decay",
    "ratedecay": "Rate Decay",
    "cvamplitude": "CV Amplitude",
    "cvcycleduration": "CV Cycle Duration",
    "cvspeed": "CV Speed",
    "cvrmsvelocity": "CV RMS Velocity",
    "cvopeningspeed": "CV Opening Speed",
    "cvclosingspeed": "CV Closing Speed"
}

# --- Helper Function to Get Readable Name ---
def get_readable_name(raw_name, name_map=READABLE_KINEMATIC_NAMES):
    """
    Converts a raw variable name (e.g., 'ft_cvamplitude') to a
    readable format (e.g., 'FT CV Amplitude' or 'Contralateral Striatum Z-Score').
    Falls back to the original name if not found or pattern doesn't match.
    """
    parts = raw_name.split('_', 1)
    prefix = ""
    base_name = raw_name

    # Check for known kinematic prefixes
    if len(parts) == 2 and parts[0].lower() in ['ft', 'hm']:
        prefix = parts[0].upper() # "FT" or "HM"
        base_name = parts[1]
    # Check for imaging variable pattern
    elif base_name.startswith("Contralateral_") and base_name.endswith("_Z"):
         img_parts = base_name.split('_')
         if len(img_parts) == 3: # Expect Contralateral_Region_Z
             return f"Contralateral {img_parts[1]} Z-Score"

    # Look up the base name in the map
    readable_base = name_map.get(base_name, base_name) # Fallback to original base_name

    if prefix:
        return f"{prefix} {readable_base}"
    else:
        return readable_base # Return base name or formatted imaging name

# --- Helper function to get ONLY the base readable name ---
def get_base_readable_name(raw_name, name_map=READABLE_KINEMATIC_NAMES):
    """
    Extracts the base part of a kinematic name (e.g., from 'ft_cvamplitude')
    and returns its readable version (e.g., 'CV Amplitude').
    """
    base_name = raw_name.split('_', 1)[-1] # Get part after first underscore, or full name if no underscore
    return name_map.get(base_name, base_name) # Lookup in map, fallback to base_name


# --------------------------------------------------------------------------
# Function for Bivariate Task Comparison Plotting (Updated X-Labels)
# --------------------------------------------------------------------------
def plot_task_comparison_scatter(
    data: pd.DataFrame,
    ft_kinematic_col: str,
    hm_kinematic_col: str,
    imaging_col: str,
    ft_stats: dict,
    hm_stats: dict,
    output_folder: str = "Output/Plots",
    file_name: str = "task_comparison_scatter.png"
):
    """
    Creates a styled dual scatter plot comparing two tasks for the same kinematic concept
    against the same imaging variable using Seaborn styling and BASE readable names on X-axis.
    """
    # --- Style setup ---
    sns.set_style("whitegrid")
    ft_color = sns.color_palette("Set1")[1]; hm_color = sns.color_palette("Set1")[0]
    reg_line_color = '#555555'; annotation_facecolor = 'whitesmoke'; annotation_edgecolor = 'grey'
    # --- Get Base Readable Names for X-axis ---
    readable_ft_base_name = get_base_readable_name(ft_kinematic_col)
    readable_hm_base_name = get_base_readable_name(hm_kinematic_col)
    readable_imaging_name = get_readable_name(imaging_col) # Keep full name for imaging

    os.makedirs(output_folder, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    sns.despine(fig=fig)

    # --- Left Subplot: Finger Tapping vs. Imaging ---
    ax0 = axes[0]
    ft_plot_data = data[[ft_kinematic_col, imaging_col]].dropna()
    if not ft_plot_data.empty and len(ft_plot_data) >= 3:
        ax0.scatter(ft_plot_data[ft_kinematic_col], ft_plot_data[imaging_col],
                    color=ft_color, alpha=0.6, edgecolor='dimgray', linewidth=0.5, s=60, label='Finger Tap')
        try:
            m_ft, b_ft = np.polyfit(ft_plot_data[ft_kinematic_col], ft_plot_data[imaging_col], 1)
            x_vals_ft = np.array([ft_plot_data[ft_kinematic_col].min(), ft_plot_data[ft_kinematic_col].max()])
            ax0.plot(x_vals_ft, m_ft * x_vals_ft + b_ft, color=reg_line_color, linewidth=2, linestyle='--')
        except Exception as e: print(f"Note: Could not plot regression line for FT ({ft_kinematic_col}): {e}")
        annotation_text_ft = (f"Finger Tap\nr = {ft_stats.get('r', np.nan):.2f}\n"
                              f"$R^2$ = {ft_stats.get('r2', np.nan):.2f}\np = {ft_stats.get('p', np.nan):.3g}\nN = {ft_stats.get('N', 0)}")
        ax0.text(0.05, 0.95, annotation_text_ft, transform=ax0.transAxes, fontsize=11, va='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=annotation_facecolor, edgecolor=annotation_edgecolor, alpha=0.8))
        ax0.margins(0.05)
    else: ax0.text(0.5, 0.5, "Insufficient FT Data", ha='center', va='center', transform=ax0.transAxes, fontsize=12)

    # Use Base Readable Name for X-Label
    ax0.set_title(f"Finger Tapping vs. {readable_imaging_name}", fontsize=14, weight='bold')
    ax0.set_xlabel(readable_ft_base_name, fontsize=12) # Use base name
    ax0.set_ylabel(readable_imaging_name, fontsize=12)
    ax0.tick_params(axis='both', which='major', labelsize=10)

    # --- Right Subplot: Hand Movement vs. Imaging ---
    ax1 = axes[1]
    hm_plot_data = data[[hm_kinematic_col, imaging_col]].dropna()
    if not hm_plot_data.empty and len(hm_plot_data) >= 3:
        ax1.scatter(hm_plot_data[hm_kinematic_col], hm_plot_data[imaging_col],
                    color=hm_color, alpha=0.6, edgecolor='dimgray', linewidth=0.5, s=60, label='Hand Movement')
        try:
            m_hm, b_hm = np.polyfit(hm_plot_data[hm_kinematic_col], hm_plot_data[imaging_col], 1)
            x_vals_hm = np.array([hm_plot_data[hm_kinematic_col].min(), hm_plot_data[hm_kinematic_col].max()])
            ax1.plot(x_vals_hm, m_hm * x_vals_hm + b_hm, color=reg_line_color, linewidth=2, linestyle='--')
        except Exception as e: print(f"Note: Could not plot regression line for HM ({hm_kinematic_col}): {e}")
        annotation_text_hm = (f"Hand Movement\nr = {hm_stats.get('r', np.nan):.2f}\n"
                              f"$R^2$ = {hm_stats.get('r2', np.nan):.2f}\np = {hm_stats.get('p', np.nan):.3g}\nN = {hm_stats.get('N', 0)}")
        ax1.text(0.05, 0.95, annotation_text_hm, transform=ax1.transAxes, fontsize=11, va='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=annotation_facecolor, edgecolor=annotation_edgecolor, alpha=0.8))
        ax1.margins(0.05)
    else: ax1.text(0.5, 0.5, "Insufficient HM Data", ha='center', va='center', transform=ax1.transAxes, fontsize=12)

    # Use Base Readable Name for X-Label
    ax1.set_title(f"Hand Movement vs. {readable_imaging_name}", fontsize=14, weight='bold')
    ax1.set_xlabel(readable_hm_base_name, fontsize=12) # Use base name
    ax1.tick_params(axis='both', which='major', labelsize=10)

    # --- Overall Figure Title using Base Readable Name ---
    try:
        base_kinematic_name = ft_kinematic_col.split('_', 1)[1]
        readable_base_name = READABLE_KINEMATIC_NAMES.get(base_kinematic_name, base_kinematic_name)
    except IndexError: readable_base_name = "Kinematic Variable"

    fig.suptitle(f"Task Comparison: '{readable_base_name}' vs. {readable_imaging_name}", fontsize=16, weight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plot
    output_path = os.path.join(output_folder, file_name)
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Task comparison plot saved to {output_path}")
    except Exception as e:
        print(f"Error saving task comparison plot {output_path}: {e}")
    finally:
        plt.close(fig)

# --------------------------------------------------------------------------
# Function for PLS Results Plotting (Updated Y-axis Labels)
# --------------------------------------------------------------------------
def plot_pls_results(
    pls_results_lv: dict,
    lv_index: int,
    output_folder: str = "Output/Plots",
    file_name_base: str = "pls_results",
    bsr_threshold: float = 2.0
):
    """
    Generates PLS plots for a specific significant LV using BASE readable names on Y-axis.
    """
    sns.set_style("whitegrid")
    task = pls_results_lv.get('task', 'unknown_task')
    kinematic_variables = pls_results_lv.get('kinematic_variables') # Raw names: e.g., ['hm_meanamplitude', ...]

    # --- 1. Plot X Loadings with Bootstrap Ratios ---
    x_loadings = pls_results_lv.get('x_loadings')
    bootstrap_ratios = pls_results_lv.get('bootstrap_ratios')

    if kinematic_variables is None:
        print(f"Warning: Kinematic variable list missing for task {task}, LV{lv_index}. Cannot plot loadings.")
        return

    if x_loadings is not None and isinstance(x_loadings, pd.Series):
        try: # Reindex safely
            x_loadings = x_loadings.reindex(kinematic_variables)
            if bootstrap_ratios is not None and isinstance(bootstrap_ratios, pd.Series):
                 bootstrap_ratios = bootstrap_ratios.reindex(kinematic_variables)
            else:
                 print(f"Note: Bootstrap ratios not available for task {task}, LV{lv_index}.")
                 bootstrap_ratios = pd.Series(np.nan, index=kinematic_variables)
        except Exception as e:
            print(f"Error reindexing loadings/BSR for task {task}, LV{lv_index}: {e}. Skipping loadings plot.")
            x_loadings = None

        if x_loadings is not None and len(kinematic_variables) == len(x_loadings):
            fig_load, ax_load = plt.subplots(figsize=(11, max(6, len(x_loadings) * 0.35)))
            try:
                # Sort by loading value (direction) using raw names for indexing
                sorted_idx = np.argsort(x_loadings.values)
                sorted_loadings = x_loadings.iloc[sorted_idx]
                sorted_bsr = bootstrap_ratios.iloc[sorted_idx]
                sorted_vars_raw = sorted_loadings.index.tolist() # Still raw names here

                # Create BASE Readable Labels for Y-axis
                readable_y_labels = [get_base_readable_name(var) for var in sorted_vars_raw]

                # Create colors based on BSR
                colors = ['#d62728' if abs(bsr) >= bsr_threshold else '#7f7f7f' for bsr in sorted_bsr.fillna(0)]

                # Plot using BASE Readable Labels
                bars = ax_load.barh(readable_y_labels, sorted_loadings.values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

                ax_load.set_xlabel(f"X Loadings on LV{lv_index} (Kinematic Variables)", fontsize=12)
                ax_load.set_title(f"PLS X Loadings (LV{lv_index}) - Task {task.upper()}\n"
                                  f"(Colored if |BSR| >= {bsr_threshold})", fontsize=14, weight='bold')
                ax_load.axvline(0, color='black', linewidth=0.8, linestyle='--')
                ax_load.grid(axis='x', linestyle='--', alpha=0.6)
                ax_load.tick_params(axis='y', labelsize=10)
                sns.despine(ax=ax_load)

                plt.tight_layout()
                loadings_filename = f"{file_name_base}_loadings_{task}_LV{lv_index}.png"
                loadings_path = os.path.join(output_folder, loadings_filename)
                try:
                    plt.savefig(loadings_path, dpi=300, bbox_inches='tight')
                    print(f"  PLS loadings plot saved to {loadings_path}")
                except Exception as e_save: print(f"  ERROR saving PLS loadings plot {loadings_path}: {type(e_save).__name__} - {e_save}")
            except Exception as e_plot: print(f"  ERROR during PLS loadings plot generation: {type(e_plot).__name__} - {e_plot}")
            finally: plt.close(fig_load)

    # --- 2. Plot LV Scores ---
    x_scores = pls_results_lv.get('x_scores'); y_scores = pls_results_lv.get('y_scores')
    lv_correlation = pls_results_lv.get('correlation', np.nan); lv_p_value = pls_results_lv.get('p_value', np.nan)
    n_samples = pls_results_lv.get('n_samples_pls', 'N/A')

    if x_scores is not None and y_scores is not None and isinstance(x_scores, np.ndarray) and isinstance(y_scores, np.ndarray):
        if len(x_scores) != len(y_scores):
             print(f"Warning: Mismatch in length of X/Y scores. Skipping scores plot.")
             return

        fig_score, ax_score = plt.subplots(figsize=(7, 6))
        sns.despine(fig=fig_score)
        try:
            sns.scatterplot(x=x_scores, y=y_scores, alpha=0.7, edgecolor='dimgray', s=50, ax=ax_score)
            # Regression line
            try:
                valid_mask = np.isfinite(x_scores) & np.isfinite(y_scores)
                if np.sum(valid_mask) >= 2 :
                    x_plot = x_scores[valid_mask]; y_plot = y_scores[valid_mask]
                    m, b = np.polyfit(x_plot, y_plot, 1)
                    line_x = np.array([np.min(x_plot), np.max(x_plot)])
                    ax_score.plot(line_x, m*line_x + b, color='red', linewidth=2, linestyle='--')
            except Exception as e_reg: print(f"Note: Could not plot regression line for PLS scores: {e_reg}")

            ax_score.set_xlabel(f"X Scores (Kinematics LV{lv_index})", fontsize=12)
            ax_score.set_ylabel(f"Y Scores (Imaging LV{lv_index})", fontsize=12)
            ax_score.set_title(f"PLS Latent Variable Scores (LV{lv_index}) - Task {task.upper()}", fontsize=14, weight='bold')
            ax_score.grid(True, linestyle='--', alpha=0.6)
            ax_score.tick_params(axis='both', which='major', labelsize=10)
            # Annotation
            annotation_text = (f"r = {lv_correlation:.3f}\np = {lv_p_value:.4g}\nN = {n_samples}")
            ax_score.text(0.05, 0.95, annotation_text, transform=ax_score.transAxes, fontsize=11, va='top',
                          bbox=dict(boxstyle='round,pad=0.4', facecolor='whitesmoke', edgecolor='grey', alpha=0.8))

            plt.tight_layout()
            scores_filename = f"{file_name_base}_scores_{task}_LV{lv_index}.png"
            scores_path = os.path.join(output_folder, scores_filename)
            try:
                plt.savefig(scores_path, dpi=300, bbox_inches='tight')
                print(f"  PLS scores plot saved to {scores_path}")
            except Exception as e_save: print(f"  ERROR saving PLS scores plot {scores_path}: {type(e_save).__name__} - {e_save}")
        except Exception as e_plot: print(f"  ERROR during PLS scores plot generation: {type(e_plot).__name__} - {e_plot}")
        finally: plt.close(fig_score)