#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
levodopa_ols_analysis_robust.py

Purpose
-------
Use baseline-adjusted regression models with robust statistical methods to assess
the relationship between dopaminergic deficit and levodopa response (Δ).

Key Improvements from Previous Version:
----------------------------------------
1) Same-Row Residualization: Ensures that for each model, the residualization
   and final regression are performed on the exact same set of complete observations,
   satisfying the Frisch-Waugh-Lovell theorem.
2) Cluster-Robust Standard Errors: Accounts for potential non-independence of
   observations from the same patient (e.g., left and right hands) by clustering
   standard errors at the patient level.
3) Pre-Pivot Averaging: Explicitly handles repeated visits by averaging kinematic
   data per patient-hand before pivoting, preventing data mixing.
4) Fully Standardized Betas: The outcome variable (Δ_resid) is now also z-scored,
   so the reported beta coefficients are fully standardized (SD change in Δ per
   SD change in predictor).
5) Robust Domain Aggregation: Domain scores are calculated by averaging the
   standardized scores of available features for each observation.
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from scipy.stats import norm
import traceback

# --- Configuration (Unchanged) ---
AGE_COL = "Age"
DATSCAN_COL = "Contralateral_Putamen_Z"
PATIENT_ID_CANDIDATES = ["Patient ID", "Patient_ID", "Subject_ID", "patient_id", "ID"]
HAND_CANDIDATES = ["Hand_Performed", "Hand", "hand", "Side"]
TASKS = ["ft", "hm"]
BASE_FEATURES = [
    "meanamplitude", "stdamplitude", "meanspeed", "stdspeed", "meanrmsvelocity",
    "stdrmsvelocity", "meanopeningspeed", "stdopeningspeed", "meanclosingspeed",
    "stdclosingspeed", "meancycleduration", "stdcycleduration", "rangecycleduration",
    "rate", "amplitudedecay", "velocitydecay", "ratedecay", "cvamplitude",
    "cvcycleduration", "cvspeed", "cvrmsvelocity", "cvopeningspeed", "cvclosingspeed"
]
POLARITY: Dict[str, int] = {
    "meanamplitude": +1, "meanspeed": +1, "meanrmsvelocity": +1, "meanopeningspeed": +1, "meanclosingspeed": +1, "rate": +1,
    "stdamplitude": -1, "stdspeed": -1, "stdrmsvelocity": -1, "stdopeningspeed": -1, "stdclosingspeed": -1,
    "cvamplitude": -1, "cvcycleduration": -1, "cvspeed": -1, "cvrmsvelocity": -1, "cvopeningspeed": -1, "cvclosingspeed": -1,
    "meancycleduration": -1, "stdcycleduration": -1, "rangecycleduration": -1, "amplitudedecay": -1, "velocitydecay": -1, "ratedecay": -1,
}
DOMAINS: Dict[str, List[str]] = {
    "Amplitude": ["meanamplitude", "amplitudedecay"],
    "Speed": ["meanspeed", "meanrmsvelocity", "meanopeningspeed", "meanclosingspeed", "velocitydecay", "ratedecay"],
    "Variability": [
        "stdamplitude","stdspeed","stdrmsvelocity","stdopeningspeed","stdclosingspeed",
        "cvamplitude","cvcycleduration","cvspeed","cvrmsvelocity","cvopeningspeed","cvclosingspeed",
        "meancycleduration","stdcycleduration","rangecycleduration"
    ],
}

# --- Utilities (Unchanged) ---
def pick_first_present(d: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in d.columns: return c
    return ""

def zscore(s: pd.Series) -> pd.Series:
    mu, sd = np.nanmean(s.values), np.nanstd(s.values, ddof=1) # Use ddof=1 for sample std dev
    if sd == 0 or np.isnan(sd): return pd.Series(np.zeros_like(s), index=s.index)
    return (s - mu) / sd

def ci_from_beta_se(beta: float, se: float, alpha: float = 0.05) -> Tuple[float, float]:
    z = norm.ppf(1 - alpha / 2.0)
    return beta - z * se, beta + z * se

def ensure_dirs(outdir: str):
    data_dir, plots_dir = os.path.join(outdir, "Data"), os.path.join(outdir, "Plots")
    os.makedirs(data_dir, exist_ok=True); os.makedirs(plots_dir, exist_ok=True)
    return data_dir, plots_dir

# ---------------------------
# Core modelling (UPDATED FOR ROBUSTNESS)
# ---------------------------

def run_ols_robust(df_model: pd.DataFrame, outcome: str, predictors: List[str], patient_col: str) -> Dict:
    """Fits an OLS model with cluster-robust standard errors and fully standardized variables."""
    d = df_model[[outcome] + predictors + [patient_col]].dropna().copy()
    
    base_results = {"n": d.shape[0], "beta_dat_std": np.nan, "se_dat": np.nan, "p_dat": np.nan, "beta_off_std": np.nan, "p_off": np.nan, "rsquared_adj": np.nan, "error": "None"}
    if d.shape[0] < 10:
        base_results["error"] = "Too few samples"
        return base_results

    # Z-score outcome and predictors for standardized betas
    d[f"z_{outcome}"] = zscore(d[outcome])
    for pred in predictors:
        d[f"z_{pred}"] = zscore(d[pred])

    formula = f"z_{outcome} ~ z_{predictors[0]} + z_{predictors[1]}"
    try:
        model = smf.ols(formula, d)
        # Use cluster-robust standard errors
        res = model.fit(cov_type="cluster", cov_kwds={"groups": d[patient_col]})
        
        beta_dat, se_dat, p_dat = res.params.get(f"z_{predictors[1]}", np.nan), res.bse.get(f"z_{predictors[1]}", np.nan), res.pvalues.get(f"z_{predictors[1]}", np.nan)
        beta_off, p_off = res.params.get(f"z_{predictors[0]}", np.nan), res.pvalues.get(f"z_{predictors[0]}", np.nan)
        rsquared_adj = res.rsquared_adj
        
        base_results.update({
            "beta_dat_std": beta_dat, "se_dat": se_dat, "p_dat": p_dat,
            "beta_off_std": beta_off, "p_off": p_off, "rsquared_adj": rsquared_adj
        })
    except Exception as e:
        base_results["error"] = str(e)
    return base_results

# ---------------------------
# Main Execution Block
# ---------------------------

def main():
    @dataclass
    class Args: input: str; outdir: str; make_plots: bool
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args = Args(input=os.path.join(project_root, "Output", "Data_Processed", "final_merged_data.csv"),
                outdir=os.path.join(project_root, "Output"), make_plots=True)
    print(f"[INFO] Running in Spyder mode. Input: {args.input}")
    print(f"[INFO] Output will be saved to: {args.outdir}")

    data_dir, plots_dir = ensure_dirs(args.outdir)
    try: df_long = pd.read_csv(args.input, sep=';', decimal='.')
    except: df_long = pd.read_csv(args.input, sep=',', decimal='.')

    patient_col = pick_first_present(df_long, PATIENT_ID_CANDIDATES)
    hand_col = pick_first_present(df_long, HAND_CANDIDATES)
    kinematic_cols = [f"{t}_{f}" for t in TASKS for f in BASE_FEATURES if f"{t}_{f}" in df_long.columns]
    
    # --- NEW: Handle repeated visits by pre-averaging ---
    print("\n[INFO] Pre-processing: Averaging repeated visits per patient-hand...")
    agg_dict = {col: 'mean' for col in kinematic_cols}
    agg_dict[AGE_COL] = 'first' # Assuming Age/DaTscan are stable per patient
    agg_dict[DATSCAN_COL] = 'first'
    df_long_avg = df_long.groupby([patient_col, hand_col, 'Medication Condition']).agg(agg_dict).reset_index()
    print(f"[SUCCESS] Data aggregated. Shape changed from {df_long.shape} to {df_long_avg.shape}")

    print("\n[INFO] Pre-processing: Pivoting data from long to wide format...")
    try:
        df_long_avg['Medication Condition'] = df_long_avg['Medication Condition'].astype(str).str.strip().str.lower()
        df_on_off = df_long_avg[df_long_avg['Medication Condition'].isin(['on', 'off'])].copy()
        
        index_cols = [patient_col, hand_col, AGE_COL, DATSCAN_COL]
        df = df_on_off.pivot_table(index=index_cols, columns='Medication Condition', values=kinematic_cols)
        df.columns = [f'{col[0]}_{col[1].upper()}' for col in df.columns]; df.reset_index(inplace=True)
        print(f"[SUCCESS] Data pivoted. Analysis will proceed on wide data (Shape: {df.shape})")
    except Exception as e:
        print(f"[FATAL ERROR] Failed to pivot data."); traceback.print_exc(); return

    for task in TASKS:
        present_feats = [f for f in BASE_FEATURES if f"{task}_{f}_OFF" in df.columns and f"{task}_{f}_ON" in df.columns]
        if not present_feats: print(f"\n[INFO] No features for task '{task}'. Skipping."); continue
        
        print(f"\n[INFO] Processing Task: {task.upper()}")
        
        # --- FEATURE-LEVEL ANALYSIS ---
        feature_rows = []
        df_task_residuals = pd.DataFrame(index=df.index) # To store residuals for domain analysis

        for feat in present_feats:
            # --- NEW: Same-Row Residualization (FWL) ---
            cols_needed = [f"{task}_{feat}_ON", f"{task}_{feat}_OFF", AGE_COL, DATSCAN_COL, patient_col]
            df_feat = df[cols_needed].dropna().copy()
            
            if df_feat.shape[0] < 10: continue

            # Harmonize, calculate delta
            sign = POLARITY.get(feat, +1)
            df_feat[f"{feat}_off_h"] = sign * df_feat[f"{task}_{feat}_OFF"]
            df_feat[f"{feat}_on_h"] = sign * df_feat[f"{task}_{feat}_ON"]
            df_feat[f"{feat}_delta"] = df_feat[f"{feat}_on_h"] - df_feat[f"{feat}_off_h"]

            # Residualize all variables against Age on the SAME subset of data
            df_feat[f"{feat}_delta_resid"] = smf.ols(f"Q('{feat}_delta') ~ Q('{AGE_COL}')", data=df_feat).fit().resid
            df_feat[f"{feat}_off_h_resid"] = smf.ols(f"Q('{feat}_off_h') ~ Q('{AGE_COL}')", data=df_feat).fit().resid
            df_feat['dat_resid'] = smf.ols(f"Q('{DATSCAN_COL}') ~ Q('{AGE_COL}')", data=df_feat).fit().resid
            
            # Store residuals for later domain aggregation
            df_task_residuals = df_task_residuals.join(df_feat[[f"{feat}_off_h_resid", f"{feat}_delta_resid"]])

            # --- UPDATED: Call robust OLS function ---
            stats = run_ols_robust(
                df_feat,
                outcome=f"{feat}_delta_resid",
                predictors=[f"{feat}_off_h_resid", 'dat_resid'],
                patient_col=patient_col
            )
            stats.update({"task": task, "feature": feat})
            
            if not np.isnan(stats["beta_dat_std"]) and not np.isnan(stats["se_dat"]):
                lo, hi = ci_from_beta_se(stats["beta_dat_std"], stats["se_dat"])
            else:
                lo = hi = np.nan
            stats["ci_lo_dat"], stats["ci_hi_dat"] = lo, hi
            feature_rows.append(stats)
        
        feat_df = pd.DataFrame(feature_rows)
        if feat_df["p_dat"].notna().any():
            rej, q, *_ = multipletests(feat_df["p_dat"].fillna(1.0).values, alpha=0.05, method="fdr_bh")
            feat_df["q_dat"], feat_df["sig_dat_q<0.05"] = q, rej
        else:
            feat_df["q_dat"], feat_df["sig_dat_q<0.05"] = np.nan, False
        out_csv_feat = os.path.join(data_dir, f"ols_robust_features_{task}.csv"); feat_df.to_csv(out_csv_feat, index=False)
        print(f"  [OK] Saved feature-level robust OLS results: {out_csv_feat}")

        # --- DOMAIN-LEVEL ANALYSIS ---
        # First, calculate DaTscan residuals on the maximal available data for domains
        df_task_residuals['dat_resid'] = smf.ols(f"Q('{DATSCAN_COL}') ~ Q('{AGE_COL}')", data=df.dropna(subset=[DATSCAN_COL, AGE_COL])).fit().resid

        domain_rows = []
        for domain, feats in DOMAINS.items():
            off_resid_cols = [f"{f}_off_h_resid" for f in feats if f"{f}_off_h_resid" in df_task_residuals.columns]
            delta_resid_cols = [f"{f}_delta_resid" for f in feats if f"{f}_delta_resid" in df_task_residuals.columns]
            
            if not off_resid_cols or not delta_resid_cols: continue
            
            # Z-score each feature residual before averaging
            z_off_resids = df_task_residuals[off_resid_cols].apply(zscore)
            z_delta_resids = df_task_residuals[delta_resid_cols].apply(zscore)

            # Create domain score by averaging available (non-NaN) z-scored features for each row
            dom_df = pd.DataFrame(index=df.index)
            dom_df[f"{domain}_off_dom_resid"] = z_off_resids.mean(axis=1)
            dom_df[f"{domain}_delta_dom_resid"] = z_delta_resids.mean(axis=1)
            dom_df['dat_resid'] = df_task_residuals['dat_resid']
            dom_df[patient_col] = df[patient_col]

            stats = run_ols_robust(
                dom_df,
                outcome=f"{domain}_delta_dom_resid",
                predictors=[f"{domain}_off_dom_resid", 'dat_resid'],
                patient_col=patient_col
            )
            stats["domain"] = domain
            domain_rows.append(stats)
            
        dom_res_df = pd.DataFrame(domain_rows)
        if not dom_res_df.empty and dom_res_df["p_dat"].notna().any():
            rej, q, *_ = multipletests(dom_res_df["p_dat"].fillna(1.0).values, alpha=0.05, method="fdr_bh")
            dom_res_df["q_dat"], dom_res_df["sig_dat_q<0.05"] = q, rej
        out_csv_dom = os.path.join(data_dir, f"ols_robust_domains_{task}.csv"); dom_res_df.to_csv(out_csv_dom, index=False)
        print(f"  [OK] Saved domain-level robust OLS results: {out_csv_dom}")

    print("\n[DONE] Robust multiple regression (OLS) analysis complete.")

if __name__ == "__main__":
    main()