#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bivariate (partial) correlation analysis (Figure 1).
UPDATED: Strict 3-Domain Consistency (Speed, Amplitude, Variability).
"""

import os
import sys
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
from scipy.stats import t as t_dist
from scipy.stats import pearsonr
from statsmodels.formula.api import ols

# -------------------------------------------------------------------------
# Configuration & Constants
# -------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
INPUT_CSV = os.path.join(PROJECT_ROOT, "Output", "Data_Processed", "final_merged_data.csv")
DATA_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Output", "Data")
PLOTS_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Output", "Plots")
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)

# Global Plot Settings
sns.set_context("paper", font_scale=1.6)
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 2.5,
    'figure.dpi': 300,
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.0
})

# STRICT 3-COLOR PALETTE (Speed, Amplitude, Variability)
DOMAIN_COLORS = {
    "Variability": "#CC79A7",  # Reddish Purple
    "Amplitude": "#0072B2",    # Blue
    "Speed": "#009E73",        # Bluish Green
}

OFF_LABEL = "off"
CONTROL_FOR_AGE = True
AGE_COL = "Age"
SIDES_TO_ANALYZE = ["Contralateral"]
TASK_PREFIXES = ["ft", "hm"]
ALPHA_FDR = 0.05
DIAGNOSE_COL = "Diagnose"
DIAGNOSE_PD_PATTERNS = ["pd", "parkinson", "morbus parkinson"]
PD_EXCLUDE_PATTERNS = ["pd+"]

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def identify_domain(feature_name: str) -> str:
    """
    Categorizes features into 3 domains: Variability, Amplitude, Speed.
    Rhythm/Rate/Duration are merged into Speed.
    """
    name = feature_name.lower()
    
    # 1. Variability (Precedence)
    if any(x in name for x in ["cv", "std", "range", "regularity", "decay"]):
        return "Variability"
    
    # 2. Amplitude
    elif "amplitude" in name:
        return "Amplitude"
        
    # 3. Speed (includes duration, rate)
    elif any(x in name for x in ["speed", "velocity", "rate", "duration", "frequency"]):
        return "Speed"
        
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

def fdr_bh(pvals):
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

def pearson_or_partial(df, x_col, y_col, z_col=None, control_for_age=True):
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
        t_val = r * np.sqrt(dfree / (1 - r**2)) if abs(r) < 1 else 0
        p = 2 * t_dist.sf(np.abs(t_val), dfree)
    else:
        r, p = pearsonr(data[x_col], data[y_col])
    return float(r), float(p), int(n)

def _age_residuals(df, x_col, age_col):
    model = ols(f"{x_col} ~ {age_col}", data=df.dropna(subset=[x_col, age_col])).fit()
    return model.resid.reindex(df.index)

def is_probable_pd_diagnosis(text):
    raw = str(text).strip().lower()
    if raw == "" or raw == "nan": return False
    for neg in PD_EXCLUDE_PATTERNS:
        if neg in raw: return False
    norm_text = "".join(ch for ch in raw if ch.isalnum() or ch.isspace())
    for pat in DIAGNOSE_PD_PATTERNS:
        if "".join(ch for ch in pat if ch.isalnum() or ch.isspace()) in norm_text: return True
    return False

def build_task_kinematic_cols(df):
    task_cols = {}
    for task in TASK_PREFIXES:
        task_cols[task] = [c for c in df.columns if c.startswith(f"{task}_")]
    return task_cols

def compute_taskwise_correlations_for_subset(df_subset, target_col, task_kin_cols):
    rows = []
    for task, cols in task_kin_cols.items():
        for kin_col in cols:
            if kin_col not in df_subset.columns: continue
            r, p, n = pearson_or_partial(df_subset, kin_col, target_col, z_col=AGE_COL if CONTROL_FOR_AGE else None, control_for_age=CONTROL_FOR_AGE)
            rows.append({"Task": task, "Kinematic_Variable": kin_col, "r": r, "p": p, "N": n})
    return pd.DataFrame(rows)

def format_p_value(p):
    if pd.isna(p): return "n.s."
    if p < 0.001: return "p<0.001"
    return f"p={p:.3f}"

# -------------------------------------------------------------------------
# PLOTTING: Figure 1 (Enhanced)
# -------------------------------------------------------------------------

def plot_figure1_enhanced(sig_df, res_df, df_all, side, target_imaging_col, out_dir, task_kin_cols):
    plot_df = sig_df.copy()
    if plot_df.empty: plot_df = res_df.copy()
    
    plot_df["Abs_r"] = plot_df["r"].abs()
    plot_df["Domain"] = plot_df["Kinematic_Variable"].apply(identify_domain)
    plot_df["Label"] = plot_df["Kinematic_Variable"].apply(format_feature_label)
    plot_df = plot_df.sort_values("Abs_r", ascending=True) 
    
    fig = plt.figure(figsize=(20, 18))
    gs_main = fig.add_gridspec(2, 1, height_ratios=[1, 2.2], hspace=0.35)
    gs_top = gs_main[0].subgridspec(1, 3, wspace=0.3)
    axA, axB, axC = fig.add_subplot(gs_top[0]), fig.add_subplot(gs_top[1]), fig.add_subplot(gs_top[2])
    gs_bot = gs_main[1].subgridspec(2, 3, hspace=0.55, wspace=0.3)
    
    # --- Panel A: Lollipop ---
    if len(plot_df) > 15: plot_data_A = plot_df.tail(15)
    else: plot_data_A = plot_df

    y_range = range(len(plot_data_A))
    colors = [DOMAIN_COLORS.get(d, "gray") for d in plot_data_A["Domain"]]
    
    xlims = (-0.7, 0.7)
    axA.axvspan(0, xlims[1], color='#0072B2', alpha=0.05)
    axA.axvspan(xlims[0], 0, color='#D55E00', alpha=0.05)
    
    axA.text(0.02, 0.98, "← Link to DaT Deficit", transform=axA.transAxes, ha='left', va='top', fontsize=11, color='#8c3e00', weight='bold')
    axA.text(0.98, 0.98, "Link to Preserved DaT →", transform=axA.transAxes, ha='right', va='top', fontsize=11, color='#004d7a', weight='bold')
    
    axA.hlines(y=y_range, xmin=0, xmax=plot_data_A["r"], color=colors, alpha=0.8, linewidth=3)
    axA.scatter(plot_data_A["r"], y_range, color=colors, s=90, zorder=3, edgecolor='white', linewidth=1)
    axA.set_yticks(y_range)
    axA.set_yticklabels(plot_data_A["Label"], fontsize=12)
    axA.axvline(0, color="black", linewidth=1.5)
    axA.set_xlabel(f"Partial Correlation ($r$)\n(adjusted for {AGE_COL})", weight='bold')
    axA.set_title("A) Strongest Kinematic Links", loc="left", pad=10)
    axA.set_xlim(xlims)
    
    handles = [mlines.Line2D([], [], color=color, marker='o', linestyle='-', label=domain) 
               for domain, color in DOMAIN_COLORS.items() if domain in plot_data_A["Domain"].unique()]
    axA.legend(handles=handles, loc="lower right", fontsize=11, framealpha=0.95, edgecolor='gray')

    # --- Panel B: Consistency (Cohort) ---
    if DIAGNOSE_COL in df_all.columns:
        pd_mask = df_all[DIAGNOSE_COL].apply(is_probable_pd_diagnosis)
        df_pd = df_all[pd_mask].copy()
        if len(df_pd) > 5:
            pd_res = compute_taskwise_correlations_for_subset(df_pd, target_imaging_col, task_kin_cols)
            merged_pd = pd.merge(res_df[["Kinematic_Variable", "r"]].rename(columns={"r": "r_all"}), pd_res[["Kinematic_Variable", "r"]].rename(columns={"r": "r_pd"}), on="Kinematic_Variable")
            merged_pd["Domain"] = merged_pd["Kinematic_Variable"].apply(identify_domain)
            cols_pd = [DOMAIN_COLORS.get(d, "gray") for d in merged_pd["Domain"]]
            
            axB.scatter(merged_pd["r_all"], merged_pd["r_pd"], c=cols_pd, alpha=0.8, s=70, edgecolor='white')
            sns.regplot(x=merged_pd["r_all"], y=merged_pd["r_pd"], ax=axB, scatter=False, color="gray", line_kws={'linestyle': '--', 'linewidth': 2})
            r_cons, p_cons = pearsonr(merged_pd["r_all"], merged_pd["r_pd"])
            axB.text(0.05, 0.95, f"Consistency:\n$r = {r_cons:.2f}$\n{format_p_value(p_cons)}", transform=axB.transAxes, va='top', fontsize=12, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))
            axB.set_xlabel("Correlation ($r$) - All Patients", weight='bold')
            axB.set_ylabel("Correlation ($r$) - PD Only", weight='bold')
            axB.set_title("B) Consistency: Cohort Selection", loc="left", pad=10)
        else: axB.text(0.5, 0.5, "Insufficient PD-only data", ha='center')
    else: axB.text(0.5, 0.5, "Diagnosis missing", ha='center')

    # --- Panel C: Consistency (Cross-Task) ---
    res_df["Base"] = res_df["Kinematic_Variable"].apply(lambda x: x.split("_",1)[1] if "_" in x else x)
    pivot_task = res_df.pivot(index="Base", columns="Task", values="r")
    if "ft" in pivot_task.columns and "hm" in pivot_task.columns:
        pivot_task = pivot_task.dropna()
        colors_c = [DOMAIN_COLORS.get(identify_domain(x), "gray") for x in pivot_task.index]
        axC.scatter(pivot_task["ft"], pivot_task["hm"], c=colors_c, alpha=0.8, s=70, edgecolor='white')
        sns.regplot(x=pivot_task["ft"], y=pivot_task["hm"], ax=axC, scatter=False, color="gray", line_kws={'linestyle': '--', 'linewidth': 2})
        r_cross, p_cross = pearsonr(pivot_task["ft"], pivot_task["hm"])
        axC.text(0.05, 0.95, f"Cross-Task:\n$r = {r_cross:.2f}$\n{format_p_value(p_cross)}", transform=axC.transAxes, va='top', fontsize=12, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))
        axC.set_xlabel("Finger Tapping ($r$)", weight='bold')
        axC.set_ylabel("Hand Movements ($r$)", weight='bold')
        axC.set_title("C) Consistency: Cross-Task", loc="left", pad=10)

    for ax in [axB, axC]: ax.grid(True, linestyle=':', alpha=0.6); sns.despine(ax=ax)

    # --- Panels D-I: Scatter Plots ---
    top_features = res_df.assign(abs_r=res_df["r"].abs()).sort_values("abs_r", ascending=False).head(6)
    for i, (_, row) in enumerate(top_features.iterrows()):
        if i >= 6: break
        ax = fig.add_subplot(gs_bot[i // 3, i % 3])
        
        kin_col = row["Kinematic_Variable"]
        sub_data = df_all[[kin_col, target_imaging_col, AGE_COL]].dropna()
        x_resid = _age_residuals(sub_data, kin_col, AGE_COL)
        y_resid = _age_residuals(sub_data, target_imaging_col, AGE_COL)
        x_z = (x_resid - x_resid.mean()) / x_resid.std()
        y_z = (y_resid - y_resid.mean()) / y_resid.std()
        
        col = DOMAIN_COLORS.get(identify_domain(kin_col), "black")
        ax.scatter(x_z, y_z, alpha=0.6, s=50, color=col, edgecolor='white', linewidth=0.5)
        sns.regplot(x=x_z, y=y_z, ax=ax, scatter=False, color="black", line_kws={'linewidth': 2.5}, truncate=True)
        
        pretty_name = format_feature_label(kin_col).replace("Finger Tapping", "FT").replace("Hand Movements", "HM")
        t_txt = ax.set_title(f"{pretty_name}\n$r={row['r']:.2f}$, {format_p_value(row['p'])}", fontsize=12, pad=20)
        t_txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground='white')])
        
        if i == 0: ax.text(-0.15, 1.2, "D) Representative Correlations", transform=ax.transAxes, fontsize=16, fontweight='bold', va='bottom', ha='left')
        
        if i % 3 == 0: ax.set_ylabel("DaT Binding (Z)", weight='bold')
        else: ax.set_ylabel("")
        if i >= 3: ax.set_xlabel("Kinematics (Z)", weight='bold')
        else: ax.set_xlabel("")
        sns.despine(ax=ax)

    plt.subplots_adjust(top=0.91)
    fig.suptitle(f"Figure 1: Kinematic signatures of Striatal Dopamine Deficit", fontsize=22, fontweight="bold", y=0.98)
    
    filename = f"Figure1_Bivariate_Enhanced.pdf"
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Figure 1 saved to: {out_path}")

def main():
    print("Loading Data for Figure 1...")
    try: df_full = pd.read_csv(INPUT_CSV, sep=";", decimal=".")
    except: df_full = pd.read_csv(INPUT_CSV, sep=",", decimal=".")

    df = df_full[df_full["Medication Condition"].astype(str).str.lower() == OFF_LABEL].copy()
    if CONTROL_FOR_AGE: df = df.dropna(subset=[AGE_COL])
    
    # Disease Duration Fix
    if 'Date of Visit' in df_full.columns:
        df_full['Visit_DT'] = pd.to_datetime(df_full['Date of Visit'], dayfirst=True, errors='coerce')
        if 'Disease_Duration' in df_full.columns:
            is_year = df_full['Disease_Duration'] > 1900
            if is_year.sum() > 0:
                df_full.loc[is_year, 'Disease_Duration'] = df_full.loc[is_year, 'Visit_DT'].dt.year - df_full.loc[is_year, 'Disease_Duration']
    
    task_kin_cols = build_task_kinematic_cols(df)
    for side in SIDES_TO_ANALYZE:
        target = f"{side}_Putamen_Z"
        if target in df.columns:
            res_df = compute_taskwise_correlations_for_subset(df, target, task_kin_cols)
            res_df["q"] = np.nan
            for t, g in res_df.groupby("Task"): res_df.loc[g.index, "q"] = fdr_bh(g["p"].values)
            plot_figure1_enhanced(res_df[res_df["q"] < ALPHA_FDR], res_df, df, side, target, PLOTS_OUTPUT_DIR, task_kin_cols)

if __name__ == "__main__":
    main()