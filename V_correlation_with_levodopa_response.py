#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved levodopa correlation analysis (Z-score version)
-------------------------------------------------------
- Keeps only DaT predictors containing "_Z"
- Harmonizes polarity for all features
- Orthogonalizes OFF vs contra DaT
- Global FDR across all tests
- Contra vs Ipsi comparison
"""

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm
import traceback

# -------------------
# CONFIG
# -------------------
AGE_COL = "Age"
PATIENT_ID_CANDIDATES = ["Patient ID", "Patient_ID", "Subject_ID", "patient_id", "ID"]
HAND_CANDIDATES = ["Hand_Performed", "Hand", "hand", "Side"]
TASKS = ["ft", "hm"]
EXPECTED_SIGN = -1   # biologically: lower DaT -> more improvement

# Feature polarity (higher = worse, negative Δ = improvement)
POLARITY = {
    "meanamplitude": +1, "meanspeed": +1, "meanrmsvelocity": +1,
    "meanopeningspeed": +1, "meanclosingspeed": +1, "rate": +1,
    "stdamplitude": -1, "stdspeed": -1, "stdrmsvelocity": -1,
    "stdopeningspeed": -1, "stdclosingspeed": -1,
    "cvamplitude": -1, "cvcycleduration": -1, "cvspeed": -1,
    "cvrmsvelocity": -1, "cvopeningspeed": -1, "cvclosingspeed": -1,
    "meancycleduration": -1, "stdcycleduration": -1, "rangecycleduration": -1,
    "amplitudedecay": -1, "velocitydecay": -1, "ratedecay": -1,
}

# -------------------
# UTILITIES
# -------------------
def pick_first_present(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"No column from {candidates} found in dataframe")

def ci_from_beta_se(beta, se, alpha=0.05):
    z = norm.ppf(1 - alpha/2)
    return beta - z*se, beta + z*se

def run_ols(df, outcome, predictors, patient_col):
    try:
        f = f"{outcome} ~ {' + '.join(predictors)}"
        res = smf.ols(f, df).fit(cov_type="cluster", cov_kwds={"groups": df[patient_col]})
        return dict(beta=res.params, se=res.bse, p=res.pvalues,
                    rsq=res.rsquared_adj, n=len(res.resid))
    except Exception as e:
        return dict(error=str(e))

# -------------------
# MAIN
# -------------------
def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(project_root, "Output", "Data_Processed", "final_merged_data.csv")
    outdir = os.path.join(project_root, "Output")
    os.makedirs(outdir, exist_ok=True)

    print(f"[INFO] Loading: {input_path}")
    try:
        df_long = pd.read_csv(input_path, sep=';', decimal='.')
        
    except Exception:
        df_long = pd.read_csv(input_path, sep=',', decimal='.')

    patient_col = pick_first_present(df_long, PATIENT_ID_CANDIDATES)
    hand_col = pick_first_present(df_long, HAND_CANDIDATES)
    
    # --- NEW: Count patients with ON recordings
    med_col = "Medication Condition"
    if med_col in df_long.columns:
        medon_patients = df_long.loc[
            df_long[med_col].astype(str).str.lower().str.strip() == "on"
        ]
        n_on_unique = medon_patients[patient_col].nunique()
        print(f"[INFO] Unique patients with ON-medication recordings: {n_on_unique}")
    else:
        print("[WARN] No 'Medication Condition' column found.")


    # --- keep only Z-score DaT predictors
    # --- keep only Z-score DaT predictors from the Putamen
    dat_cols = [c for c in df_long.columns if "_Z" in c and "Putamen" in c]
    print(f"[INFO] Found {len(dat_cols)} DaT Z predictors.")

    # --- DaT correlation matrix
    corr = df_long[dat_cols].corr()
    corr.to_csv(os.path.join(outdir,"DaT_Z_predictor_correlations.csv"))
    print("[INFO] Saved DaT Z correlation matrix.")

    # --- Safe pivot
    df_long["Medication Condition"] = df_long["Medication Condition"].astype(str).str.lower().str.strip()
    index_cols = [patient_col, hand_col, AGE_COL] + dat_cols
    numeric_cols = df_long.select_dtypes(include=[np.number]).columns.tolist()
    keep_cols = index_cols + ["Medication Condition"]
    df_long_pivotable = df_long[keep_cols + [c for c in numeric_cols if c not in index_cols]]
    print(f"[INFO] Pivoting with {len(numeric_cols)} numeric columns.")

    try:
        df_pivot = (
            df_long_pivotable[df_long_pivotable["Medication Condition"].isin(["on","off"])]
            .pivot_table(index=index_cols, columns="Medication Condition")
            .pipe(lambda d: d.set_axis([f"{a}_{b.upper()}" for a,b in d.columns], axis=1))
        )
        df_pivot.reset_index(inplace=True)
        print(f"[SUCCESS] Pivot complete: {df_pivot.shape}")
    except Exception:
        print("[FATAL] Pivot failed.")
        traceback.print_exc()
        return

    # -------------------
    # MAIN ANALYSIS
    # -------------------
    all_results=[]
    for task in TASKS:
        feat_cols=[c for c in df_pivot.columns if c.startswith(f"{task}_") and c.endswith("_OFF")]
        if not feat_cols: 
            continue
        print(f"[INFO] Task {task.upper()} — {len(feat_cols)} features")

        for feat_off in feat_cols:
            feat_on = feat_off.replace("_OFF","_ON")
            if feat_on not in df_pivot: 
                continue
            feat = feat_off.split(f"{task}_")[-1].replace("_OFF","")
            sign = POLARITY.get(feat, +1)

            for dat_contra in [c for c in dat_cols if "Contralateral" in c]:
                dat_ipsi = dat_contra.replace("Contralateral","Ipsilateral")
                use_cols=[patient_col,AGE_COL,feat_off,feat_on,dat_contra]
                if dat_ipsi in df_pivot.columns: use_cols.append(dat_ipsi)
                df=df_pivot[use_cols].dropna().copy()
                if df.shape[0]<10: 
                    continue

                # Harmonized feature space
                df["off_h"] = sign*df[feat_off]
                df["on_h"]  = sign*df[feat_on]
                df["delta"] = df["on_h"] - df["off_h"]  # negative = improvement

                # Residualization
                df["off_resid"]   = smf.ols(f"Q('off_h') ~ Q('{AGE_COL}')",df).fit().resid
                df["delta_resid"] = smf.ols(f"Q('delta') ~ Q('{AGE_COL}')",df).fit().resid
                df["dat_contra_resid"] = smf.ols(f"Q('{dat_contra}') ~ Q('{AGE_COL}')",df).fit().resid
                if dat_ipsi in df:
                    df["dat_ipsi_resid"] = smf.ols(f"Q('{dat_ipsi}') ~ Q('{AGE_COL}')",df).fit().resid

                # Orthogonalize OFF
                df["off_only"] = smf.ols("off_resid ~ dat_contra_resid",df).fit().resid

                # =============================
                # EXTENDED MODEL SET
                # =============================
                
                # --- Base linear terms (already residualized)
                r1 = run_ols(df, "delta_resid", ["off_only", "dat_contra_resid"], patient_col)
                
                # --- Add ipsilateral predictor (original Model 2)
                r2 = run_ols(df, "delta_resid",
                             ["off_only", "dat_contra_resid", "dat_ipsi_resid"],
                             patient_col) if "dat_ipsi_resid" in df else None
                
                # --- Model 3: Quadratic term (nonlinearity test)
                try:
                    df["dat_contra_resid_sq"] = df["dat_contra_resid"] ** 2
                    r3 = run_ols(df, "delta_resid",
                                 ["off_only", "dat_contra_resid", "dat_contra_resid_sq"],
                                 patient_col)
                except Exception as e:
                    r3 = {"error": str(e)}
                
                # --- Model 4: Log-transformed DaT (saturating test)
                try:
                    eps = 1e-6  # avoid log(0)
                    df["dat_contra_resid_log"] = np.log(np.abs(df["dat_contra_resid"]) + eps)
                    r4 = run_ols(df, "delta_resid",
                                 ["off_only", "dat_contra_resid_log"],
                                 patient_col)
                except Exception as e:
                    r4 = {"error": str(e)}
                
                # --- Collect all models
                for m_id, res in enumerate([r1, r2, r3, r4], start=1):
                    if not res or "error" in res:
                        continue
                    # For nonlinear models, report the main DaT term (linear or log)
                    key = "dat_contra_resid"
                    if m_id == 3:
                        key = "dat_contra_resid_sq" if key not in res["beta"] else key
                    elif m_id == 4:
                        key = "dat_contra_resid_log"
                    b = res["beta"].get(key, np.nan)
                    se = res["se"].get(key, np.nan)
                    lo, hi = ci_from_beta_se(b, se)
                    wrong_sign = np.sign(b) != EXPECTED_SIGN
                    all_results.append(dict(
                        task=task,
                        feature=feat,
                        dat_contra=dat_contra,
                        model=m_id,
                        beta=b,
                        se=se,
                        ci_lo=lo,
                        ci_hi=hi,
                        p=res["p"].get(key, np.nan),
                        rsq=res["rsq"],
                        n=res["n"],
                        wrong_sign=wrong_sign
                    ))

                for m_id,res in enumerate([r1,r2],start=1):
                    if not res or "error" in res: 
                        continue
                    b=res["beta"].get("dat_contra_resid",np.nan)
                    se=res["se"].get("dat_contra_resid",np.nan)
                    lo,hi=ci_from_beta_se(b,se)
                    wrong_sign = np.sign(b)!=EXPECTED_SIGN
                    all_results.append(dict(
                        task=task, feature=feat, dat_contra=dat_contra, model=m_id,
                        beta=b, se=se, ci_lo=lo, ci_hi=hi,
                        p=res["p"].get("dat_contra_resid",np.nan),
                        rsq=res["rsq"], n=res["n"], wrong_sign=wrong_sign
                    ))

    # -------------------
    # POSTPROCESSING
    # -------------------
    print("[INFO] Consolidating results...")
    res_df = pd.DataFrame(all_results)
    if res_df.empty:
        print("[WARN] No valid models fit.")
        return
    
    # FDR correction
    rej, q, _, _ = multipletests(res_df["p"].fillna(1.0), alpha=0.05, method="fdr_bh")
    res_df["q"] = q
    res_df["sig_q<0.05"] = rej
    
    # ============================================================
    # 🔧 FILTER: Keep only Model 1 (contralateral-only)
    #     and restrict to Putamen DaT predictors
    # ============================================================
    # res_df = res_df[
    #     (res_df["model"] == 1) &
    #     (res_df["dat_contra"].str.contains("Putamen", case=False, na=False))
    # ]
    
    # Sort for readability
    res_df.sort_values(by="q", inplace=True)
    
    # Save main summary CSV (filtered)
    out_csv = os.path.join(outdir, "summary_ols_robust_improved_Z.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"[DONE] Results saved: {out_csv}")
    print(f"[INFO] Total models after filtering: {len(res_df)}")
    
    # ============================================================
    # 📄 SUPPLEMENTARY TABLE 2 (DaT ↔ Levodopa)
    # ============================================================
    supp2 = res_df.loc[res_df["q"] < 0.05, [
        "task", "feature", "beta", "p", "q", "ci_lo", "ci_hi", "n"
    ]].copy()
    
    # Format table values
    supp2["beta"] = supp2["beta"].round(2)
    supp2["p"] = supp2["p"].apply(lambda x: "< 0.001" if x < 0.001 else f"{x:.3f}")
    supp2["q"] = supp2["q"].apply(lambda x: "< 0.001" if x < 0.001 else f"{x:.3f}")
    supp2["ci_lo"] = supp2["ci_lo"].round(2)
    supp2["ci_hi"] = supp2["ci_hi"].round(2)
    supp2.rename(columns={
        "task": "TASK",
        "feature": "KINEMATIC_FEATURE",
        "beta": "STANDARDIZED_BETA",
        "p": "P_VALUE",
        "q": "Q_VALUE_FDR",
        "ci_lo": "CI_LOW",
        "ci_hi": "CI_HIGH",
        "n": "N"
    }, inplace=True)
    
    supp2.sort_values(["TASK", "Q_VALUE_FDR"], inplace=True)
    
    # Export
    supp2_path = os.path.join(outdir, "Supplementary_Table_2_DaT_vs_Levodopa.csv")
    supp2.to_csv(supp2_path, index=False)
    print(f"[INFO] Supplementary Table 2 saved: {supp2_path}")
    print(f"[INFO] Significant results (q<0.05): {len(supp2)} rows")
    
    # Optional: preview top few results in console
    print(supp2.head(10))


if __name__=="__main__":
    main()
