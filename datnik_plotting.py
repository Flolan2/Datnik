import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

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


def plot_single_bivariate_scatter(
    data: pd.DataFrame,
    kinematic_col: str,
    imaging_col: str,
    stats_dict: dict,
    ax: plt.Axes, # <<< THIS MUST BE PRESENT
    title_prefix: str = "" # <<< THIS MUST BE PRESENT
):
    """
    Creates a styled scatter plot on a given matplotlib Axes.
    Args:
        data (pd.DataFrame): DataFrame containing the two columns to plot.
        kinematic_col (str): Name of the kinematic variable column.
        imaging_col (str): Name of the imaging variable column.
        stats_dict (dict): Dictionary containing statistics like 'r', 'p', 'q', 'N'.
        ax (plt.Axes): The matplotlib Axes object to draw on.
        title_prefix (str): Prefix for the plot title (e.g., "A) ").
    """
    sns.set_style("whitegrid") # Apply style within function if called standalone
    point_color = sns.color_palette("pastel")[0] # Example color
    reg_line_color = '#555555'
    annotation_facecolor = 'whitesmoke'
    annotation_edgecolor = 'grey'

    readable_kinematic_name = get_readable_name(kinematic_col)
    readable_imaging_name = get_readable_name(imaging_col) # Usually Contralateral Striatum Z-Score

    plot_data_clean = data[[kinematic_col, imaging_col]].dropna()
    if plot_data_clean.empty or len(plot_data_clean) < 3:
        ax.text(0.5, 0.5, "Insufficient Data", ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.set_title(f"{title_prefix}{readable_kinematic_name}", fontsize=10, weight='bold')
        return

    ax.scatter(plot_data_clean[kinematic_col], plot_data_clean[imaging_col],
               color=point_color, alpha=0.6, edgecolor='dimgray', linewidth=0.5, s=40) # Smaller points for multi-panel

    try:
        m, b = np.polyfit(plot_data_clean[kinematic_col], plot_data_clean[imaging_col], 1)
        x_vals = np.array([plot_data_clean[kinematic_col].min(), plot_data_clean[kinematic_col].max()])
        ax.plot(x_vals, m * x_vals + b, color=reg_line_color, linewidth=1.5, linestyle='--')
    except Exception: # Broad except for plotting
        pass # Silently skip regression line if it fails

    r_val = stats_dict.get('r', np.nan)
    p_val = stats_dict.get('p', np.nan)
    q_val = stats_dict.get('q', np.nan)
    n_val = stats_dict.get('N', 0)

    p_str = f"p={p_val:.2g}" if pd.notna(p_val) and p_val >= 0.001 else ("p<0.001" if pd.notna(p_val) else "p=N/A")
    q_str = f"q={q_val:.2g}" if pd.notna(q_val) and q_val >= 0.001 else ("q<0.001" if pd.notna(q_val) else "q=N/A")

    annotation_text = (f"r={r_val:.2f}\n{p_str}\n{q_str}\nN={n_val}") # Compact
    ax.text(0.05, 0.95, annotation_text, transform=ax.transAxes, fontsize=8, va='top', # Smaller font
            bbox=dict(boxstyle='round,pad=0.3', facecolor=annotation_facecolor, edgecolor=annotation_edgecolor, alpha=0.7))
    ax.margins(0.05)

    ax.set_title(f"{title_prefix}{readable_kinematic_name}", fontsize=10, weight='bold') # Title is just the kinematic var
    ax.set_xlabel(readable_kinematic_name, fontsize=9)
    ax.set_ylabel(readable_imaging_name, fontsize=9) # Y-label only on first column
    ax.tick_params(axis='both', which='major', labelsize=8)
    sns.despine(ax=ax)


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
# Function for PLS Results Plotting (Updated Y-axis Labels and R2 in Scores Plot)
# --------------------------------------------------------------------------
def plot_pls_results(
    pls_results_lv: dict,
    lv_index: int,
    output_folder: str = "Output/Plots",
    file_name_base: str = "pls_results",
    bsr_threshold: float = 2.0
):
    """
    Generates PLS plots for a specific significant LV using BASE readable names on Y-axis
    for loadings, and includes R-squared in the scores plot.
    """
    sns.set_style("whitegrid")
    task = pls_results_lv.get('task', 'unknown_task')
    kinematic_variables = pls_results_lv.get('kinematic_variables') # Raw names: e.g., ['hm_meanamplitude', ...]

    # --- 1. Plot X Loadings with Bootstrap Ratios ---
    x_loadings = pls_results_lv.get('x_loadings')
    bootstrap_ratios = pls_results_lv.get('bootstrap_ratios')

    if kinematic_variables is None:
        print(f"Warning: Kinematic variable list missing for task {task}, LV{lv_index}. Cannot plot loadings.")
        # Decide if you want to return or continue to scores plot if scores are available
    elif x_loadings is not None and isinstance(x_loadings, pd.Series):
        try: # Reindex safely
            x_loadings = x_loadings.reindex(kinematic_variables) # Ensure it's indexed by the provided kinematic_variables
            if bootstrap_ratios is not None and isinstance(bootstrap_ratios, pd.Series):
                 bootstrap_ratios = bootstrap_ratios.reindex(kinematic_variables) # Align BSRs too
            else:
                 print(f"Note: Bootstrap ratios not available or not a Series for task {task}, LV{lv_index}.")
                 # Create a NaN series if BSRs are missing, to avoid errors in color generation
                 bootstrap_ratios = pd.Series(np.nan, index=kinematic_variables)
        except Exception as e:
            print(f"Error reindexing loadings/BSR for task {task}, LV{lv_index}: {e}. Skipping loadings plot.")
            x_loadings = None # Prevent further processing of loadings

        if x_loadings is not None and not x_loadings.empty and len(kinematic_variables) == len(x_loadings):
            fig_load, ax_load = plt.subplots(figsize=(11, max(6, len(x_loadings) * 0.35)))
            try:
                # Sort by loading value (direction) using raw names for indexing
                # Ensure x_loadings is a Series for .values and .iloc
                if not isinstance(x_loadings, pd.Series): x_loadings = pd.Series(x_loadings, index=kinematic_variables)
                if not isinstance(bootstrap_ratios, pd.Series): bootstrap_ratios = pd.Series(bootstrap_ratios, index=kinematic_variables)


                sorted_idx = np.argsort(x_loadings.values)
                sorted_loadings = x_loadings.iloc[sorted_idx]
                sorted_bsr = bootstrap_ratios.iloc[sorted_idx] # bootstrap_ratios should now be a Series
                sorted_vars_raw = sorted_loadings.index.tolist()

                readable_y_labels = [get_base_readable_name(var) for var in sorted_vars_raw]
                colors = ['#d62728' if abs(bsr) >= bsr_threshold else '#7f7f7f' for bsr in sorted_bsr.fillna(0)]

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
        elif x_loadings is None or x_loadings.empty :
            print(f"  Skipping PLS loadings plot for task {task}, LV{lv_index}: x_loadings is None, empty or not a Series after processing.")

    # --- 2. Plot LV Scores ---
    x_scores = pls_results_lv.get('x_scores'); y_scores = pls_results_lv.get('y_scores')
    lv_correlation = pls_results_lv.get('correlation', np.nan); lv_p_value = pls_results_lv.get('p_value', np.nan)
    n_samples = pls_results_lv.get('n_samples_pls', 'N/A') # Assuming 'n_samples_pls' is in the dict

    if x_scores is not None and y_scores is not None and isinstance(x_scores, np.ndarray) and isinstance(y_scores, np.ndarray):
        if len(x_scores) != len(y_scores):
             print(f"Warning: Mismatch in length of X/Y scores for task {task}, LV{lv_index}. Skipping scores plot.")
             return # Or decide to proceed if one is a scalar for some reason (unlikely for scores)

        fig_score, ax_score = plt.subplots(figsize=(7, 6))
        sns.despine(fig=fig_score)
        try:
            sns.scatterplot(x=x_scores, y=y_scores, alpha=0.7, edgecolor='dimgray', s=50, ax=ax_score)
            try:
                valid_mask = np.isfinite(x_scores) & np.isfinite(y_scores)
                if np.sum(valid_mask) >= 2 : # Need at least 2 points for polyfit
                    x_plot = x_scores[valid_mask]; y_plot = y_scores[valid_mask]
                    m, b = np.polyfit(x_plot, y_plot, 1)
                    line_x = np.array([np.min(x_plot), np.max(x_plot)])
                    ax_score.plot(line_x, m*line_x + b, color='red', linewidth=2, linestyle='--')
            except Exception as e_reg: print(f"Note: Could not plot regression line for PLS scores task {task}, LV{lv_index}: {e_reg}")

            ax_score.set_xlabel(f"X Scores (Kinematics LV{lv_index})", fontsize=12)
            ax_score.set_ylabel(f"Y Scores (Imaging LV{lv_index})", fontsize=12)
            ax_score.set_title(f"PLS Latent Variable Scores (LV{lv_index}) - Task {task.upper()}", fontsize=14, weight='bold')
            ax_score.grid(True, linestyle='--', alpha=0.6)
            ax_score.tick_params(axis='both', which='major', labelsize=10)

            # --- MODIFICATION FOR R-SQUARED START ---
            if pd.notna(lv_correlation):
                r_squared = lv_correlation**2
                r_squared_str = f"{r_squared:.3f}" # Format R-squared
            else:
                r_squared_str = "N/A"

            # Updated Annotation text to include R-squared
            # Using $R^2$ for a nice superscript 2
            annotation_text = (f"r = {lv_correlation:.3f}\n"
                               f"$R^2$ = {r_squared_str}\n"
                               f"p = {lv_p_value:.4g}\n" # .4g for p-value to handle small values
                               f"N = {n_samples}")
            # --- MODIFICATION FOR R-SQUARED END ---

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
    else:
        print(f"  Skipping PLS scores plot for task {task}, LV{lv_index}: x_scores or y_scores are None or not ndarrays.")

        
def plot_ridge_coefficients(
    ridge_results_task: dict,
    top_n: int = 20, # Plot top N features by absolute coefficient
    output_folder: str = "Output/Plots",
    file_name_base: str = "ridge_coefficients"
):
    """
    Generates a horizontal bar plot for Ridge Regression coefficients for a specific task.

    Args:
        ridge_results_task (dict): The dictionary returned by run_ridge_analysis for one task.
        top_n (int): Number of top features to display based on absolute coefficient value.
        output_folder (str): Folder path to save the plot.
        file_name_base (str): Base name for the output PNG file.
    """
    task = ridge_results_task.get('task', 'unknown_task')
    coeffs_series = ridge_results_task.get('coefficients')
    optimal_alpha = ridge_results_task.get('optimal_alpha', np.nan)
    r2_full = ridge_results_task.get('r2_full_data', np.nan)
    n_samples = ridge_results_task.get('n_samples_ridge', 'N/A')
    imaging_var = ridge_results_task.get('imaging_variable', 'Target')

    if coeffs_series is None or not isinstance(coeffs_series, pd.Series) or coeffs_series.empty:
        print(f"  Skipping Ridge coefficients plot for task {task}: No coefficient data found.")
        return

    # Sort by absolute coefficient value
    coeffs_sorted = coeffs_series.reindex(coeffs_series.abs().sort_values(ascending=False).index)
    plot_data = coeffs_sorted.head(top_n).iloc[::-1] # Select top N and reverse for plotting

    if plot_data.empty:
         print(f"  Skipping Ridge coefficients plot for task {task}: No coefficients left after filtering/sorting.")
         return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, max(5, len(plot_data) * 0.35))) # Adjust height dynamically

    colors = ['#d62728' if c < 0 else '#1f77b4' for c in plot_data.values] # Red for negative, Blue for positive

    bars = ax.barh(plot_data.index, plot_data.values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.7)

    ax.set_xlabel("Ridge Coefficient Value", fontsize=12)
    ax.set_ylabel("Kinematic Feature", fontsize=12)
    title = (f"Ridge Coefficients predicting {imaging_var} - Task {task.upper()} (OFF Data)\n"
             f"Top {len(plot_data)} Features (Optimal Alpha={optimal_alpha:.2f}, Full Data R²={r2_full:.3f}, N={n_samples})")
    ax.set_title(title, fontsize=14, weight='bold')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.tick_params(axis='y', labelsize=10)
    sns.despine(ax=ax, left=True, bottom=False) # Remove left spine for clarity

    # Add coefficient values as text labels (optional, can be cluttered)
    # for bar in bars:
    #     width = bar.get_width()
    #     label_x_pos = width + (0.01 * ax.get_xlim()[1]) if width >= 0 else width - (0.01 * ax.get_xlim()[1])
    #     ax.text(label_x_pos, bar.get_y() + bar.get_height()/2., f'{width:.2f}',
    #             va='center', ha='left' if width >= 0 else 'right', fontsize=8)


    plt.tight_layout()
    plot_filename = f"{file_name_base}_{task}_OFF.png"
    plot_path = os.path.join(output_folder, plot_filename)
    try:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Ridge coefficients plot saved to {plot_path}")
    except Exception as e_save:
        print(f"  ERROR saving Ridge coefficients plot {plot_path}: {type(e_save).__name__} - {e_save}")
    finally:
        plt.close(fig)


# Attempt to import adjustText, but make it optional
try:
    from adjustText import adjust_text
    ADJUSTTEXT_AVAILABLE = True
except ImportError:
    ADJUSTTEXT_AVAILABLE = False

# --- Keep other functions like get_readable_name, plot_task_comparison_scatter etc. ---

def plot_bivariate_vs_ridge_scatter(
    bivariate_results_df: pd.DataFrame, # DataFrame from all_raw_bivariate_results_list
    ridge_results_task: dict,          # Single task results from all_ridge_results_dict
    significant_bivariate_df: pd.DataFrame, # DF with FDR significant bivariate results
    task_prefix: str,                  # 'ft' or 'hm'
    top_n_label: int = 5,              # How many top Ridge features to label
    output_folder: str = "Output/Plots",
    file_name_base: str = "bivar_vs_ridge_scatter"
):
    """
    Creates a scatter plot comparing bivariate Pearson r and Ridge coefficients.

    Args:
        bivariate_results_df (pd.DataFrame): DF containing raw bivariate results for all tasks,
                                             MUST contain 'Task', 'Kinematic Variable', and
                                             'Pearson Correlation (r)' columns.
        ridge_results_task (dict): Ridge results for the specific task.
        significant_bivariate_df (pd.DataFrame): Filtered DF containing only FDR significant
                                                 bivariate results. Must contain 'Task' and
                                                 'Kinematic Variable' columns if not empty.
        task_prefix (str): The task prefix ('ft' or 'hm').
        top_n_label (int): Number of top features (by abs Ridge coeff) to label.
        output_folder (str): Folder path to save the plot.
        file_name_base (str): Base name for the output PNG file.
    """
    print(f"\n--- Generating Bivariate vs Ridge Scatter for Task: {task_prefix.upper()} ---")

    # --- Prepare Data ---
    # Get Ridge coefficients
    coeffs_series = ridge_results_task.get('coefficients')
    if coeffs_series is None or not isinstance(coeffs_series, pd.Series) or coeffs_series.empty:
        print(f"  Skipping plot for task {task_prefix}: Missing Ridge coefficients.")
        return
    ridge_df = coeffs_series.reset_index()
    ridge_df.columns = ['Feature', 'Ridge_Coefficient'] # Feature column has full name like 'ft_...'

    # Get Bivariate results for this task
    bivar_task_df = bivariate_results_df[bivariate_results_df['Task'] == task_prefix].copy()
    if bivar_task_df.empty:
        print(f"  Skipping plot for task {task_prefix}: Missing Bivariate results for this task.")
        return

    # Check for required column from bivariate results
    if 'Kinematic Variable' not in bivar_task_df.columns:
         print(f"  Skipping plot for task {task_prefix}: 'Kinematic Variable' column missing in bivariate results df.")
         return
    if 'Pearson Correlation (r)' not in bivar_task_df.columns:
         print(f"  Skipping plot for task {task_prefix}: 'Pearson Correlation (r)' column missing in bivariate results df.")
         return

    # Select and rename columns for merging
    bivar_task_df = bivar_task_df[['Kinematic Variable', 'Pearson Correlation (r)']].rename(
        columns={'Kinematic Variable': 'Feature', 'Pearson Correlation (r)': 'Bivariate_r'}
    )

    # Get list of FDR significant features for this task
    sig_bivar_features = []
    # Check if the significant df exists, is not empty, and has the necessary columns
    if significant_bivariate_df is not None and not significant_bivariate_df.empty and \
       'Task' in significant_bivariate_df.columns and 'Kinematic Variable' in significant_bivariate_df.columns:
        sig_rows = significant_bivariate_df[significant_bivariate_df['Task'] == task_prefix]
        if not sig_rows.empty:
             sig_bivar_features = sig_rows['Kinematic Variable'].tolist()

    # Merge Ridge and Bivariate results based on the full feature name
    merged_df = pd.merge(bivar_task_df, ridge_df, on='Feature', how='inner')
    if merged_df.empty:
        print(f"  Skipping plot for task {task_prefix}: No common features after merging Bivariate and Ridge results.")
        return

    # Add significance flag based on the extracted list
    merged_df['Significant_Bivariate'] = merged_df['Feature'].isin(sig_bivar_features)

    # Drop rows with NaN correlation or coefficient values before plotting/calculating corr
    merged_df.dropna(subset=['Bivariate_r', 'Ridge_Coefficient'], inplace=True)
    if merged_df.empty:
        print(f"  Skipping plot for task {task_prefix}: No valid data points after dropping NaNs.")
        return

    # --- Create Plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9, 8))

    # Define point styles based on significance
    markers = {True: "X", False: "o"} # X for significant, o for non-significant
    sizes = {True: 80, False: 50}

    sns.scatterplot(
        data=merged_df,
        x='Bivariate_r',
        y='Ridge_Coefficient',
        style='Significant_Bivariate', # Use style for significance
        markers=markers,
        size='Significant_Bivariate', # Use size for significance
        sizes=sizes,
        alpha=0.7, # Overall alpha
        hue='Significant_Bivariate', # Use hue for significance
        palette={True: '#d62728', False: '#1f77b4'}, # Red=Sig, Blue=Non-sig
        legend='brief', # Control legend display
        ax=ax
    )

    # Add quadrant lines
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)

    # --- Annotations & Labels ---
    # Calculate correlation between the plotted coefficients
    corr_val, p_val = pearsonr(merged_df['Bivariate_r'], merged_df['Ridge_Coefficient'])
    ax.text(0.05, 0.95, f'r(Bivar, Ridge) = {corr_val:.2f}\np = {p_val:.3g}',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='whitesmoke', alpha=0.8))

    # Label top N features by Ridge coefficient magnitude
    top_features = merged_df.reindex(merged_df['Ridge_Coefficient'].abs().sort_values(ascending=False).index).head(top_n_label)
    texts = []
    for i, row in top_features.iterrows():
         # Simple label using base name (remove task prefix)
         base_name = row['Feature'].replace(f"{task_prefix}_", "")
         texts.append(ax.text(row['Bivariate_r'], row['Ridge_Coefficient'], base_name, fontsize=8))

    # Adjust text labels to prevent overlap if library is available
    if ADJUSTTEXT_AVAILABLE and texts:
        try:
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        except Exception as e_adjust:
             print(f"  Warning: adjustText failed: {e_adjust}")
    elif not ADJUSTTEXT_AVAILABLE:
        print("  Note: 'adjustText' library not found. Labels might overlap.")

    ax.set_xlabel("Bivariate Pearson r (Feature vs Z-Score)", fontsize=12)
    ax.set_ylabel("Ridge Regression Coefficient", fontsize=12)
    imaging_var = ridge_results_task.get('imaging_variable', 'Target')
    n_points = len(merged_df)
    ax.set_title(f"Bivariate Correlation vs Ridge Coefficient - Task {task_prefix.upper()} (OFF Data, N={n_points})\nPredicting {imaging_var}", fontsize=14, weight='bold')

    # Customize legend
    handles, labels = ax.get_legend_handles_labels()
    # Ensure both True and False are present for consistent legend creation
    try:
        # Create labels based on boolean values directly if possible
        unique_styles = merged_df['Significant_Bivariate'].unique()
        if True in unique_styles and False in unique_styles:
            true_idx = labels.index('True'); false_idx = labels.index('False')
            ax.legend([handles[false_idx], handles[true_idx]],
                      ['Bivar Non-Sig (q>0.05)', 'Bivar Sig (q<=0.05)'],
                      title='Bivariate FDR', title_fontsize='10', fontsize='9', loc='lower right')
        elif True in unique_styles: # Only significant points plotted
             ax.legend([handles[labels.index('True')]], ['Bivar Sig (q<=0.05)'], title='Bivariate FDR', title_fontsize='10', fontsize='9', loc='lower right')
        elif False in unique_styles: # Only non-significant points plotted
             ax.legend([handles[labels.index('False')]], ['Bivar Non-Sig (q>0.05)'], title='Bivariate FDR', title_fontsize='10', fontsize='9', loc='lower right')
        else: # No points plotted or legend failed
             ax.legend_.remove() if ax.legend_ else None
    except (ValueError, IndexError, AttributeError):
        ax.legend_.remove() if ax.legend_ else None # Fallback: remove legend if error

    plt.tight_layout()
    plot_filename = f"{file_name_base}_{task_prefix}_OFF.png"
    plot_path = os.path.join(output_folder, plot_filename)
    try:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Bivariate vs Ridge scatter plot saved to {plot_path}")
    except Exception as e_save:
        print(f"  ERROR saving Bivariate vs Ridge plot {plot_path}: {type(e_save).__name__} - {e_save}")
    finally:
        plt.close(fig)
        
        
        # --- START OF ADDITION to datnik_plotting.py ---

# Add this import if you don't have it already
import matplotlib.patches as mpatches

def plot_elasticnet_coefficients_with_significance(
    enet_results_task: dict,
    pls_significant_features: list, # List of features significant in PLS
    top_n: int = 20, # Plot top N non-zero features by absolute coefficient
    output_folder: str = "Output/Plots",
    file_name_base: str = "elasticnet_coefficients_sig"
):
    """
    Generates a horizontal bar plot for non-zero ElasticNet coefficients,
    highlighting features also found significant via PLS BSRs.

    Args:
        enet_results_task (dict): The dictionary from run_elasticnet_analysis for one task.
        pls_significant_features (list): A list of full kinematic variable names
                                         (e.g., 'hm_stdamplitude') that were significant
                                         based on PLS BSR threshold for this task.
        top_n (int): Max number of non-zero features to display based on absolute coefficient value.
        output_folder (str): Folder path to save the plot.
        file_name_base (str): Base name for the output PNG file.
    """
    task = enet_results_task.get('task', 'unknown_task')
    coeffs_series = enet_results_task.get('coefficients')
    optimal_alpha = enet_results_task.get('optimal_alpha', np.nan)
    optimal_l1_ratio = enet_results_task.get('optimal_l1_ratio', np.nan)
    r2_full = enet_results_task.get('r2_full_data', np.nan)
    n_samples = enet_results_task.get('n_samples_enet', 'N/A')
    n_selected = enet_results_task.get('n_selected_features', 'N/A')
    imaging_var = enet_results_task.get('imaging_variable', 'Target')

    if coeffs_series is None or not isinstance(coeffs_series, pd.Series) or coeffs_series.empty:
        print(f"  Skipping ElasticNet plot for task {task}: No coefficient data found.")
        return

    # Filter for non-zero coefficients
    non_zero_coeffs = coeffs_series[coeffs_series.abs() > 1e-9].copy() # Use tolerance for float comparison
    if non_zero_coeffs.empty:
        print(f"  Skipping ElasticNet plot for task {task}: No non-zero coefficients found.")
        return

    # Sort by absolute coefficient value and select top N
    coeffs_sorted = non_zero_coeffs.reindex(non_zero_coeffs.abs().sort_values(ascending=False).index)
    plot_data = coeffs_sorted.head(top_n).iloc[::-1] # Select top N and reverse for plotting

    if plot_data.empty:
         print(f"  Skipping ElasticNet plot for task {task}: No coefficients left after filtering/sorting.")
         return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(11, max(5, len(plot_data) * 0.35))) # Adjust height

    # --- Color and Style Logic ---
    colors = []
    edge_colors = []
    hatches = []
    feature_labels = plot_data.index.tolist()

    # Base colors for positive/negative
    color_pos = '#1f77b4' # Blue
    color_neg = '#d62728' # Red
    # Style for PLS significant features
    pls_sig_edgecolor = 'black'
    pls_sig_hatch = '' # No hatch for PLS significant
    # Style for PLS non-significant features
    pls_nonsig_edgecolor = 'grey'
    pls_nonsig_hatch = '///' # Add diagonal hatching

    for feature, coeff in plot_data.items():
        is_pls_significant = feature in pls_significant_features
        base_color = color_pos if coeff >= 0 else color_neg

        colors.append(base_color)
        edge_colors.append(pls_sig_edgecolor if is_pls_significant else pls_nonsig_edgecolor)
        hatches.append(pls_sig_hatch if is_pls_significant else pls_nonsig_hatch)
    # --- End Color and Style Logic ---

    bars = ax.barh(
        feature_labels,
        plot_data.values,
        color=colors,
        edgecolor=edge_colors, # Apply edge colors
        hatch=hatches,         # Apply hatching
        linewidth=1.0,         # Slightly thicker edge for visibility
        alpha=0.85
    )

    ax.set_xlabel("ElasticNet Coefficient Value", fontsize=12)
    ax.set_ylabel("Kinematic Feature", fontsize=12)
    title = (f"ElasticNet Coefficients predicting {imaging_var} - Task {task.upper()} (OFF Data)\n"
             f"Top {len(plot_data)} Non-Zero Feats (Alpha={optimal_alpha:.3f}, L1 Ratio={optimal_l1_ratio:.2f}, R²={r2_full:.3f}, N={n_samples}, Selected={n_selected})")
    ax.set_title(title, fontsize=14, weight='bold')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.tick_params(axis='y', labelsize=10)
    sns.despine(ax=ax, left=True, bottom=False)

    # --- Add Legend for Significance ---
    solid_patch = mpatches.Patch(facecolor='grey', edgecolor='black', label='PLS BSR Significant')
    hatched_patch = mpatches.Patch(facecolor='grey', edgecolor='grey', hatch='///', label='PLS BSR Non-Significant')
    ax.legend(handles=[solid_patch, hatched_patch], title="Feature Reliability (PLS BSR)", fontsize=9, title_fontsize=10, loc='lower right')
    # --- End Legend ---

    plt.tight_layout()
    plot_filename = f"{file_name_base}_{task}_OFF.png"
    plot_path = os.path.join(output_folder, plot_filename)
    try:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  ElasticNet coefficients plot with significance saved to {plot_path}")
    except Exception as e_save:
        print(f"  ERROR saving ElasticNet coefficients plot {plot_path}: {type(e_save).__name__} - {e_save}")
    finally:
        plt.close(fig)

# --- START OF ADDITION to datnik_plotting.py ---

