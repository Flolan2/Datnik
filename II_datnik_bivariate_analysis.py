#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bivariate (partial) correlation analysis between OFF-state kinematics
(FT & HM) and side-specific Putamen-Z-Scores (DaT-SPECT).

Updates:
- Reduced gap between Row 1 (A-C) and Row 2 (Scatter Plots)
- Maintained nested grid structure
"""

import os
import sys
import unicodedata
import difflib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines

from scipy.stats import t as t_dist
from scipy.stats import pearsonr
from statsmodels.formula.api import ols

# -------------------------------------------------------------------------
# Configuration & Constants
# -------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

INPUT_CSV = os.path.join(
    PROJECT_ROOT, "Output", "Data_Processed", "final_merged_data.csv"
)

DATA_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Output", "Data")
PLOTS_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Output", "Plots")
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)

OFF_LABEL = "off"
ANALYSIS_MODES = ["all"]
DIAGNOSE_COL = "Diagnose"
DIAGNOSE_PD_PATTERNS = ["pd", "parkinson", "morbus parkinson"]
PD_EXCLUDE_PATTERNS = ["pd+"]

CONTROL_FOR_AGE = True
AGE_COL = "Age"
SIDES_TO_ANALYZE = ["Contralateral"]

TASK_PREFIXES = ["ft", "hm"]
ALPHA_FDR = 0.05

DOMAIN_COLORS = {
    "Variability": "#d62728",  # Red
    "Amplitude": "#1f77b4",    # Blue
    "Speed": "#2ca02c",        # Green
    "Rhythm": "#9467bd"        # Purple
}

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def identify_domain(feature_name: str) -> str:
    name = feature_name.lower()
    if any(x in name for x in ["cv", "std", "range", "regularity", "decay"]):
        return "Variability"
    elif any(x in name for x in ["speed", "velocity", "rate"]):
        return "Speed"
    elif "amplitude" in name:
        return "Amplitude"
    else:
        return "Other"

def format_feature_label(col_name: str) -> str:
    if "_" in col_name:
        task, base = col_name.split("_", 1)
    else:
        task, base = "", col_name
    task = task.upper()
    clean_base = base
    
    if base.startswith("mean"): clean_base = "Mean " + base[4:].capitalize()
    elif base.startswith("std"): clean_base = base[3:].capitalize() + " (SD)"
    elif base.startswith("cv"): clean_base = base[2:].capitalize() + " (CV)"
    
    lower_base = clean_base.lower()
    if "rmsvelocity" in lower_base: clean_base = clean_base.replace("rmsvelocity", "RMS Velocity").replace("Rmsvelocity", "RMS Velocity")
    if "cycleduration" in lower_base: clean_base = clean_base.replace("cycleduration", "Cycle Duration")
    if "openingspeed" in lower_base: clean_base = clean_base.replace("openingspeed", "Opening Speed")
    if "closingspeed" in lower_base: clean_base = clean_base.replace("closingspeed", "Closing Speed")
    if "amplitudedecay" in lower_base: clean_base = clean_base.replace("amplitudedecay", "Amplitude Decay")
        
    return f"{task} {clean_base}".strip()

def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    qvals = np.full(n, np.nan, dtype=float)
    if n == 0: return qvals
    order = np.argsort(pvals)
    ranked = pvals[order]
    prev_q = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1.0
        p = ranked[i]
        q = p * n / rank
        if q > prev_q: q = prev_q
        prev_q = q
        qvals[order[i]] = q
    return qvals

def pearson_or_partial(df: pd.DataFrame, x_col: str, y_col: str, z_col: str = None, control_for_age: bool = True):
    use_cols = [x_col, y_col]
    if control_for_age and z_col: use_cols.append(z_col)
    data = df[use_cols].dropna()
    n = data.shape[0]
    if n < 3: return np.nan, np.nan, n
    if control_for_age and z_col:
        r_xy = data[x_col].corr(data[y_col])
        r_xz = data[x_col].corr(data[z_col])
        r_yz = data[y_col].corr(data[z_col])
        denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        if denom == 0 or np.isnan(denom): return np.nan, np.nan, n
        r = (r_xy - r_xz * r_yz) / denom
        dfree = n - 3
        if abs(r) >= 1: p = np.nan
        else:
            t_val = r * np.sqrt(dfree / (1 - r**2))
            p = 2 * t_dist.sf(np.abs(t_val), dfree)
    else:
        r, p = pearsonr(data[x_col], data[y_col])
    return float(r), float(p), int(n)

def _age_residuals(df: pd.DataFrame, x_col: str, age_col: str) -> pd.Series:
    model = ols(f"{x_col} ~ {age_col}", data=df.dropna(subset=[x_col, age_col])).fit()
    return model.resid.reindex(df.index)

def _normalize_diag_text(text):
    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if ch.isalnum() or ch.isspace())

def is_probable_pd_diagnosis(text):
    raw = str(text).strip().lower()
    if raw == "" or raw == "nan": return False
    for neg in PD_EXCLUDE_PATTERNS:
        if neg in raw: return False
    norm_text = _normalize_diag_text(text)
    for pat in DIAGNOSE_PD_PATTERNS:
        if _normalize_diag_text(pat) in norm_text: return True
    return False

def build_task_kinematic_cols(df: pd.DataFrame) -> dict:
    task_cols = {}
    for task in TASK_PREFIXES:
        cols = [c for c in df.columns if c.startswith(f"{task}_")]
        task_cols[task] = cols
    return task_cols

def compute_taskwise_correlations_for_subset(df_subset, target_col, task_kin_cols):
    rows = []
    for task, cols in task_kin_cols.items():
        for kin_col in cols:
            if kin_col not in df_subset.columns: continue
            r, p, n = pearson_or_partial(
                df_subset, kin_col, target_col, 
                z_col=AGE_COL if CONTROL_FOR_AGE else None, 
                control_for_age=CONTROL_FOR_AGE
            )
            rows.append({"Task": task, "Kinematic_Variable": kin_col, "r": r, "p": p, "N": n})
    return pd.DataFrame(rows)

def format_p_value(p):
    if pd.isna(p): return "n.s."
    if p < 0.001: return "p<0.001"
    return f"p={p:.3f}"

# -------------------------------------------------------------------------
# PLOTTING: Figure 1 (Enhanced 3x3 with Nested Grid)
# -------------------------------------------------------------------------

def plot_figure1_enhanced(
    sig_df: pd.DataFrame,
    res_df: pd.DataFrame,
    df_all: pd.DataFrame,
    side: str,
    target_imaging_col: str,
    out_dir: str,
    task_kin_cols: dict
):
    plot_df = sig_df.copy()
    if plot_df.empty: plot_df = res_df.copy()
    
    plot_df["Abs_r"] = plot_df["r"].abs()
    plot_df["Domain"] = plot_df["Kinematic_Variable"].apply(identify_domain)
    plot_df["Label"] = plot_df["Kinematic_Variable"].apply(format_feature_label)
    plot_df = plot_df.sort_values("Abs_r", ascending=True) 
    
    # --- LAYOUT ---
    fig = plt.figure(figsize=(20, 18))
    
    # FIX: Reduced hspace from 0.45 to 0.3 to bring Row 1 and Row 2 closer
    gs_main = fig.add_gridspec(2, 1, height_ratios=[1, 2.1], hspace=0.3)
    
    # --- TOP SECTION (Panels A, B, C) ---
    gs_top = gs_main[0].subgridspec(1, 3, wspace=0.3)
    axA = fig.add_subplot(gs_top[0])
    axB = fig.add_subplot(gs_top[1])
    axC = fig.add_subplot(gs_top[2])
    
    # --- BOTTOM SECTION (Panels D-I) ---
    gs_bot = gs_main[1].subgridspec(2, 3, hspace=0.25, wspace=0.3)
    
    # --- Panel A: Lollipop Chart ---
    if len(plot_df) > 15: plot_data_A = plot_df.tail(15)
    else: plot_data_A = plot_df

    y_range = range(len(plot_data_A))
    colors = [DOMAIN_COLORS.get(d, "gray") for d in plot_data_A["Domain"]]
    
    xlims = (-0.7, 0.7)
    axA.axvspan(0, xlims[1], color='blue', alpha=0.05)
    axA.axvspan(xlims[0], 0, color='red', alpha=0.05)
    axA.text(xlims[1]*0.9, 0, "Link to\nPreserved DaT", ha='right', va='bottom', 
             fontsize=9, color='navy', alpha=0.6, fontweight='bold')
    axA.text(xlims[0]*0.9, 0, "Link to\nDaT Deficit", ha='left', va='bottom', 
             fontsize=9, color='darkred', alpha=0.6, fontweight='bold')
    
    axA.hlines(y=y_range, xmin=0, xmax=plot_data_A["r"], color=colors, alpha=0.6, linewidth=2)
    axA.scatter(plot_data_A["r"], y_range, color=colors, s=70, zorder=3)
    axA.set_yticks(y_range)
    axA.set_yticklabels(plot_data_A["Label"], fontsize=10)
    axA.axvline(0, color="black", linewidth=0.8)
    axA.set_xlabel(f"Partial Correlation ($r$)\n(adjusted for {AGE_COL})", fontsize=11)
    axA.set_title("A) Strongest Kinematic Links to Dopamine", loc="left", fontweight="bold", fontsize=12)
    axA.set_xlim(xlims)
    
    handles = [mlines.Line2D([], [], color=color, marker='o', linestyle='-', label=domain) 
               for domain, color in DOMAIN_COLORS.items() if domain in plot_data_A["Domain"].unique()]
    axA.legend(handles=handles, loc="lower right", fontsize=9, title="Kinematic Domain", framealpha=0.9)

    # --- Panel B: Consistency (All vs PD Only) ---
    if DIAGNOSE_COL in df_all.columns:
        pd_mask = df_all[DIAGNOSE_COL].apply(is_probable_pd_diagnosis)
        df_pd = df_all[pd_mask].copy()
        if len(df_pd) > 5:
            pd_res = compute_taskwise_correlations_for_subset(df_pd, target_imaging_col, task_kin_cols)
            merged_pd = pd.merge(
                res_df[["Kinematic_Variable", "r"]].rename(columns={"r": "r_all"}),
                pd_res[["Kinematic_Variable", "r"]].rename(columns={"r": "r_pd"}),
                on="Kinematic_Variable"
            )
            merged_pd["Domain"] = merged_pd["Kinematic_Variable"].apply(identify_domain)
            cols_pd = [DOMAIN_COLORS.get(d, "gray") for d in merged_pd["Domain"]]
            axB.scatter(merged_pd["r_all"], merged_pd["r_pd"], c=cols_pd, alpha=0.7, s=45, edgecolor='white')
            
            sns.regplot(x=merged_pd["r_all"], y=merged_pd["r_pd"], ax=axB, 
                        scatter=False, color="gray", line_kws={'linestyle': '--', 'linewidth': 1.5})
            
            r_cons, p_cons = pearsonr(merged_pd["r_all"], merged_pd["r_pd"])
            stats_txt = f"Consistency:\n$r = {r_cons:.2f}$\n{format_p_value(p_cons)}"
            axB.text(0.05, 0.95, stats_txt, transform=axB.transAxes, 
                     va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            
            axB.set_xlabel("Correlation ($r$) - All Patients", fontsize=10)
            axB.set_ylabel("Correlation ($r$) - PD Only", fontsize=10)
            axB.set_title("B) Consistency: Cohort Selection", loc="left", fontweight="bold", fontsize=12)
        else:
            axB.text(0.5, 0.5, "Insufficient PD-only data", ha='center')
    else:
        axB.text(0.5, 0.5, "Diagnosis column missing", ha='center')

    # --- Panel C: Consistency (Cross-Task) ---
    res_df["Base"] = res_df["Kinematic_Variable"].apply(lambda x: x.split("_",1)[1] if "_" in x else x)
    pivot_task = res_df.pivot(index="Base", columns="Task", values="r")
    if "ft" in pivot_task.columns and "hm" in pivot_task.columns:
        pivot_task = pivot_task.dropna()
        domains_c = [identify_domain(x) for x in pivot_task.index]
        colors_c = [DOMAIN_COLORS.get(d, "gray") for d in domains_c]
        axC.scatter(pivot_task["ft"], pivot_task["hm"], c=colors_c, alpha=0.7, s=45, edgecolor='white')
        
        sns.regplot(x=pivot_task["ft"], y=pivot_task["hm"], ax=axC, 
                    scatter=False, color="gray", line_kws={'linestyle': '--', 'linewidth': 1.5})
        
        r_cross, p_cross = pearsonr(pivot_task["ft"], pivot_task["hm"])
        stats_txt_c = f"Cross-Task:\n$r = {r_cross:.2f}$\n{format_p_value(p_cross)}"
        axC.text(0.05, 0.95, stats_txt_c, transform=axC.transAxes,
                 va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        axC.set_xlabel("Finger Tapping Correlation ($r$)", fontsize=10)
        axC.set_ylabel("Hand Movements Correlation ($r$)", fontsize=10)
        axC.set_title("C) Consistency: Cross-Task", loc="left", fontweight="bold", fontsize=12)
        for ax in [axB, axC]:
            ax.grid(True, linestyle=':', alpha=0.4)
            sns.despine(ax=ax)

    # --- Panels D-I: 6 Scatter Plots (Using gs_bot) ---
    top_features = res_df.assign(abs_r=res_df["r"].abs()).sort_values("abs_r", ascending=False).head(6)
    
    for i, (_, row) in enumerate(top_features.iterrows()):
        if i >= 6: break
        grid_row = i // 3 # 0 or 1
        grid_col = i % 3
        ax = fig.add_subplot(gs_bot[grid_row, grid_col])
        
        kin_col = row["Kinematic_Variable"]
        cols = [kin_col, target_imaging_col, AGE_COL]
        sub_data = df_all[cols].dropna()
        
        x_resid = _age_residuals(sub_data, kin_col, AGE_COL)
        y_resid = _age_residuals(sub_data, target_imaging_col, AGE_COL)
        
        x_z = (x_resid - x_resid.mean()) / x_resid.std()
        y_z = (y_resid - y_resid.mean()) / y_resid.std()
        
        dom = identify_domain(kin_col)
        col = DOMAIN_COLORS.get(dom, "black")
        
        ax.scatter(x_z, y_z, alpha=0.5, s=25, color=col, edgecolor='none')
        
        sns.regplot(x=x_z, y=y_z, ax=ax, scatter=False, color="black", 
                    line_kws={'linewidth':1.5}, truncate=True)
        
        pretty_name = format_feature_label(kin_col)
        pretty_name = pretty_name.replace("Finger Tapping", "FT").replace("Hand Movements", "HM")
        
        p_val = row['p']
        title_str = f"{pretty_name}\n$r={row['r']:.2f}$, {format_p_value(p_val)}"
        ax.set_title(title_str, fontsize=10)
        
        if i == 0:
            ax.text(-0.15, 1.2, "D) Representative Correlations", transform=ax.transAxes, 
                    fontsize=12, fontweight='bold', va='bottom', ha='left')

        ax.margins(x=0.1, y=0.1)
        
        if grid_col == 0:
            ax.set_ylabel("DaT Binding (Z)", fontsize=10)
        else:
            ax.set_ylabel("")
            
        if grid_row == 1:
            ax.set_xlabel("Kinematics (Z)", fontsize=10)
        else:
            ax.set_xlabel("")
            
        sns.despine(ax=ax)

    plt.subplots_adjust(top=0.91)
    fig.suptitle(f"Figure 1: Kinematic signatures of Striatal Dopamine Deficit ({side} Putamen)", 
                 fontsize=18, fontweight="bold", y=0.98)
    
    filename = f"Figure1.pdf"
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Enhanced Figure 1 saved to: {out_path}")

def main():
    print("Loading Data...")
    if not os.path.exists(INPUT_CSV): print(f"Error: {INPUT_CSV} not found."); sys.exit(1)
    try: df_full = pd.read_csv(INPUT_CSV, sep=";", decimal=".")
    except: df_full = pd.read_csv(INPUT_CSV, sep=",", decimal=".")

    df = df_full[df_full["Medication Condition"].astype(str).str.lower() == OFF_LABEL].copy()
    if CONTROL_FOR_AGE: df = df.dropna(subset=[AGE_COL])
    
    # Quick check for Disease Duration stats
    # --- FIX FOR DISEASE DURATION ---
    # Convert 'Date of Visit' to datetime to extract the year
    df_full['Visit_DT'] = pd.to_datetime(df_full['Date of Visit'], dayfirst=True, errors='coerce')
    df_full['Visit_Year'] = df_full['Visit_DT'].dt.year
    
    # Check if "Disease_Duration" looks like a year (e.g., > 1900)
    is_year_format = df_full['Disease_Duration'] > 1900
    
    if is_year_format.sum() > 0:
        print("Detected 'Year of Diagnosis' instead of Duration. Recalculating...")
        
        # Calculate Duration = Visit Year - Diagnosis Year
        # We store it in a new column or overwrite the old one
        df_full.loc[is_year_format, 'Disease_Duration'] = (
            df_full.loc[is_year_format, 'Visit_Year'] - df_full.loc[is_year_format, 'Disease_Duration']
        )
    
    # Now print the corrected stats
    valid_dur = df_full['Disease_Duration'].dropna()
    # Filter out negative numbers just in case of typos (e.g. Diagnosis Year > Visit Year)
    valid_dur = valid_dur[valid_dur >= 0]
    
    print(f"\n--- CORRECTED DISEASE DURATION ---")
    print(f"N = {len(valid_dur)}")
    print(f"Duration: {valid_dur.mean():.2f} ± {valid_dur.std():.2f} years")
    print(f"Range: {valid_dur.min():.1f} - {valid_dur.max():.1f} years")
    
    task_kin_cols = build_task_kinematic_cols(df)
    
    for side in SIDES_TO_ANALYZE:
        target_col = f"{side}_Putamen_Z"
        if target_col not in df.columns: continue
        
        print(f"Processing Side: {side}")
        res_df = compute_taskwise_correlations_for_subset(df, target_col, task_kin_cols)
        
        res_df["q"] = np.nan
        for t_name, grp in res_df.groupby("Task"):
            res_df.loc[grp.index, "q"] = fdr_bh(grp["p"].values)
            
        sig_df = res_df[res_df["q"] < ALPHA_FDR]
        plot_figure1_enhanced(sig_df, res_df, df, side, target_col, PLOTS_OUTPUT_DIR, task_kin_cols)

if __name__ == "__main__":
    main()