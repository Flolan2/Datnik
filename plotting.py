import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_dual_scatter(
    data: pd.DataFrame,
    kinematic_col: str,
    z_col: str,
    new_col: str,
    r_z: float,
    p_z: float,
    r2_z: float,
    r_new: float,
    p_new: float,
    r2_new: float,
    output_folder: str = "Output/Plots",
    file_name: str = "dual_scatter.png"
):
    """
    Creates a dual scatter plot with two subplots:
      - Left: scatter plot of the kinematic variable vs. the Z-scores of the DatScan modality.
      - Right: scatter plot of the kinematic variable vs. the new DatScan raw values.

    Parameters:
        data (pd.DataFrame): DataFrame containing the data (already filtered, if desired).
        kinematic_col (str): Column name for the kinematic variable.
        z_col (str): Column name for the Z-scores of the DatScan modality.
        new_col (str): Column name for the new DatScan raw values.
        r_z (float): Precomputed Pearson's r for the Z-scores.
        p_z (float): Precomputed p-value for the Z-scores.
        r2_z (float): Precomputed R² (r_z**2) for the Z-scores.
        r_new (float): Precomputed Pearson's r for the new modality.
        p_new (float): Precomputed p-value for the new modality.
        r2_new (float): Precomputed R² (r_new**2) for the new modality.
        output_folder (str): Folder where the plot will be saved.
        file_name (str): Name of the output image file.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Create figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Set a grid for better readability
    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.6)
    
    # Define common x-values for regression lines
    x_vals = np.linspace(data[kinematic_col].min(), data[kinematic_col].max(), 100)
    
    # --- Left Subplot: Z-Scores vs. Kinematic Variable ---
    ax0 = axes[0]
    ax0.scatter(data[kinematic_col], data[z_col], color='black', alpha=0.7, edgecolor='w', s=50)
    
    # Regression line for the Z-scores (based on the local data)
    m_z, b_z = np.polyfit(data[kinematic_col], data[z_col], 1)
    ax0.plot(x_vals, m_z * x_vals + b_z, color='red', linewidth=2)
    
    # Set title and labels
    ax0.set_title(f"Z-Scores vs. {kinematic_col}", fontsize=12)
    ax0.set_xlabel(kinematic_col, fontsize=10)
    ax0.set_ylabel(z_col, fontsize=10)
    
    # Add annotation box with the passed-in p-value and R²
    annotation_text_z = f"r = {r_z:.2f}\n$R^2$ = {r2_z:.2f}\np = {p_z:.3g}"
    ax0.text(
        0.05, 0.95,
        annotation_text_z,
        transform=ax0.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    )
    
    # --- Right Subplot: New Raw Values vs. Kinematic Variable ---
    ax1 = axes[1]
    ax1.scatter(data[kinematic_col], data[new_col], color='black', alpha=0.7, edgecolor='w', s=50)
    
    # Regression line for the new modality
    m_new, b_new = np.polyfit(data[kinematic_col], data[new_col], 1)
    ax1.plot(x_vals, m_new * x_vals + b_new, color='red', linewidth=2)
    
    # Set title and labels
    ax1.set_title(f"Raw Values vs. {kinematic_col}", fontsize=12)
    ax1.set_xlabel(kinematic_col, fontsize=10)
    ax1.set_ylabel(new_col, fontsize=10)
    
    # Add annotation box with the passed-in p-value and R²
    annotation_text_new = f"r = {r_new:.2f}\n$R^2$ = {r2_new:.2f}\np = {p_new:.3g}"
    ax1.text(
        0.05, 0.95,
        annotation_text_new,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    )
    
    plt.tight_layout()
    output_path = os.path.join(output_folder, file_name)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Dual scatter plot saved to {output_path}")
