# plotting.py
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_dual_scatter(
    data: pd.DataFrame,
    kinematic_col: str,
    old_col: str,
    new_col: str,
    r_old: float,
    p_old: float,
    r2_old: float,
    r_new: float,
    p_new: float,
    r2_new: float,
    output_folder: str = "Output/Plots",
    file_name: str = "dual_scatter.png"
):
    """
    Creates a dual scatter plot with two subplots:
      - Left: scatter plot of the kinematic variable vs. the old Datscan values.
      - Right: scatter plot of the kinematic variable vs. the new Datscan values.

    Parameters:
        data (pd.DataFrame): DataFrame containing the data (already filtered, if desired).
        kinematic_col (str): Column name for the kinematic variable.
        old_col (str): Column name for the old Datscan modality.
        new_col (str): Column name for the new Datscan modality.
        r_old (float): Precomputed Pearson's r for the old modality.
        p_old (float): Precomputed p-value for the old modality.
        r2_old (float): Precomputed R² (r_old**2) for the old modality.
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
    x_vals_old = np.linspace(data[kinematic_col].min(), data[kinematic_col].max(), 100)
    x_vals_new = np.linspace(data[kinematic_col].min(), data[kinematic_col].max(), 100)
    
    # --- Left Subplot: Old Datscan vs. Kinematic Variable ---
    ax0 = axes[0]
    ax0.scatter(data[kinematic_col], data[old_col], color='black', alpha=0.7, edgecolor='w', s=50)
    
    # Regression line for the old modality (based on the local data)
    m_old, b_old = np.polyfit(data[kinematic_col], data[old_col], 1)
    ax0.plot(x_vals_old, m_old * x_vals_old + b_old, color='red', linewidth=2)
    
    # Set title and labels
    ax0.set_title(f"Z-Scores vs. {kinematic_col}", fontsize=12)
    ax0.set_xlabel(kinematic_col, fontsize=10)
    ax0.set_ylabel(old_col, fontsize=10)
    
    # Add annotation box with the *passed-in* p-value and R²
    annotation_text_old = f"r = {r_old:.2f}\n$R^2$ = {r2_old:.2f}\np = {p_old:.3g}"
    ax0.text(
        0.05, 0.95,
        annotation_text_old,
        transform=ax0.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    )
    
    # --- Right Subplot: New Datscan vs. Kinematic Variable ---
    ax1 = axes[1]
    ax1.scatter(data[kinematic_col], data[new_col], color='black', alpha=0.7, edgecolor='w', s=50)
    
    # Regression line for the new modality
    m_new, b_new = np.polyfit(data[kinematic_col], data[new_col], 1)
    ax1.plot(x_vals_new, m_new * x_vals_new + b_new, color='red', linewidth=2)
    
    # Set title and labels
    ax1.set_title(f"Raw Values vs. {kinematic_col}", fontsize=12)
    ax1.set_xlabel(kinematic_col, fontsize=10)
    ax1.set_ylabel(new_col, fontsize=10)
    
    # Add annotation box with the *passed-in* p-value and R²
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
