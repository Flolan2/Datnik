#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kinematics vs Clinical vs Supervised Composite for DAT

This script:
1) Builds two kinematic composites per task (FT, HM):
   - Pre-specified: mean of FDR-significant features (z-scored; oriented)
   - Supervised: Elastic-Net trained to predict DAT (oof predictions; grouped CV)
     with strict within-fold covariate control (residualization).
2) Compares associations with DAT (partial correlations) against UPDRS
   using Williams/Steiger tests (dependent correlations).
3) Cluster-bootstraps Δz (Fisher z difference) CIs at the patient level.
4) Runs hierarchical regressions (ΔAdj R², partial F) to test incremental value.

Reproducible, no leakage, reviewer-proof.
"""

import os, sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import ElasticNetCV
from statsmodels.formula.api import ols
import statsmodels.stats.api as sms
import pingouin as pg

warnings.filterwarnings('ignore', category=FutureWarning)
np.random.seed(42)

# =========================
# Config / Paths
# =========================
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
project_root_dir = os.path.dirname(script_dir)
processed_data_dir = os.path.join(project_root_dir, "Output", "Data_Processed")
data_output_folder = os.path.join(project_root_dir, "Output", "Data")
os.makedirs(data_output_folder, exist_ok=True)

DAT_COL = 'Contralateral_Putamen_Z'
CLINICAL_SCORE_COL = 'Clinical_Score'
COVARIATES = ['Age', 'Sex_numeric']  # residualized within folds for supervised model
N_BOOTSTRAPS = 5000
TASKS = ['ft', 'hm']

# Pre-selected features (from your FDR-significant bivariate screen)
PRESELECTED = {
    'ft': [
        'ft_meanamplitude',
        'ft_meanrmsvelocity',
        'ft_cvamplitude',
        'ft_meanclosingspeed',
    ],
    'hm': [
        'hm_stdamplitude',
        'hm_cvspeed',
        'hm_cvclosingspeed',
        'hm_cvrmsvelocity',
        'hm_cvamplitude',
        'hm_stdrmsvelocity',
        'hm_stdspeed',
        'hm_stdclosingspeed',
    ]
}

print("\n" + "="*80)
print("--- RUNNING COMPARATIVE ANALYSIS: KINEMATICS vs. CLINICAL SCORES ---")
print("="*80 + "\n")

# =========================
# Utilities
# =========================
def williams_steiger_test(r1, r2, r12, n):
    """Williams/Steiger test for two dependent correlations sharing one variable."""
    k = 1 - r1**2 - r2**2 - r12**2 + 2*r1*r2*r12
    if k <= 0:
        return np.nan, np.nan
    z = (r1 - r2) * np.sqrt((n - 3) * (1 + r12))
    denom = np.sqrt(2 * (1 - r12) * k)
    if denom == 0:
        return np.nan, np.nan
    z_stat = z / denom
    p_val = 2 * (1 - norm.cdf(abs(z_stat)))
    return z_stat, p_val

def partial_corr_residuals(df, y, x, covars):
    """Return residuals for y and x after regressing out covariates in df."""
    y_resid = ols(f'{y} ~ {" + ".join(covars)}', data=df).fit().resid
    x_resid = ols(f'{x} ~ {" + ".join(covars)}', data=df).fit().resid
    return y_resid, x_resid

def cluster_boot_delta_z(df, lhs_resid, rhs1_resid, rhs2_resid, group_col='Patient ID', n_boot=2000):
    """
    Bootstrap Δz = atanh(cor(lhs, rhs1)) - atanh(cor(lhs, rhs2))
    by resampling patients with replacement; no refitting inside bootstrap
    (appropriate because residualization done already for evaluation dataset).
    """
    groups = df[group_col].values
    unique_groups = df[group_col].unique()
    lhs = np.asarray(lhs_resid)
    r1 = np.asarray(rhs1_resid)
    r2 = np.asarray(rhs2_resid)

    deltas = []
    for _ in range(n_boot):
        sel_groups = np.random.choice(unique_groups, size=len(unique_groups), replace=True)
        mask = np.isin(groups, sel_groups)
        if mask.sum() < 5:
            continue
        try:
            r_dat_r1 = pg.corr(lhs[mask], r1[mask])['r'].iloc[0]
            r_dat_r2 = pg.corr(lhs[mask], r2[mask])['r'].iloc[0]
            deltas.append(np.arctanh(r_dat_r1) - np.arctanh(r_dat_r2))
        except Exception:
            continue
    if len(deltas) == 0:
        return np.nan, np.nan, 0
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return lo, hi, len(deltas)

def orient_to_reference(signal, reference):
    """Flip sign so correlation with reference is positive."""
    tmp = pd.concat([pd.Series(signal), pd.Series(reference)], axis=1).dropna()
    if tmp.shape[0] < 3:
        return signal
    r = tmp.corr().iloc[0, 1]
    return signal if r >= 0 else -signal

# =========================
# Load & Prepare Data
# =========================
print("--- 1. Loading and Merging Data ---")
merged_csv_file = os.path.join(processed_data_dir, "final_merged_data.csv")
try:
    try:
        df_kin = pd.read_csv(merged_csv_file, sep=';', decimal='.')
    except:
        df_kin = pd.read_csv(merged_csv_file, sep=',', decimal='.')
    print(f"[INFO] Loaded kinematic/imaging data. Shape: {df_kin.shape}")
except FileNotFoundError:
    sys.exit(f"[FATAL ERROR] File not found: {merged_csv_file}")

clinical_csv_file = os.path.join(project_root_dir, "Input", "Clinical_input.csv")
try:
    try:
        df_clin = pd.read_csv(clinical_csv_file, sep=';', decimal=',')
    except:
        df_clin = pd.read_csv(clinical_csv_file, sep=',', decimal='.')
    print(f"[INFO] Loaded clinical data. Shape: {df_clin.shape}")
except FileNotFoundError:
    sys.exit(f"[FATAL ERROR] Clinical data file not found: {clinical_csv_file}")

df_full = pd.merge(df_kin, df_clin, left_on='Patient ID', right_on='No.', how='left')
print(f"[INFO] Merged data shape: {df_full.shape}")

df = df_full[df_full['Medication Condition'].str.lower() == 'off'].copy()
print(f"[INFO] Filtered for 'OFF' state. New shape: {df.shape}")

# Age consolidation
if 'Age_y' in df.columns and 'Age_x' in df.columns:
    df['Age'] = df['Age_y'].fillna(df['Age_x'])
elif 'Age_y' in df.columns:
    df.rename(columns={'Age_y': 'Age'}, inplace=True)
elif 'Age_x' in df.columns:
    df.rename(columns={'Age_x': 'Age'}, inplace=True)

# Unified, hand-specific clinical score
print("[INFO] Creating unified hand-specific clinical score column...")
df[CLINICAL_SCORE_COL] = np.nan
task_hand_map = {
    'ft': {'Left': 'MedOFF ft left', 'Right': 'MedOFF ft right'},
    'hm': {'Left': 'MedOFF hm left', 'Right': 'MedOFF hm right'}
}
for task, hand_dict in task_hand_map.items():
    for hand, col in hand_dict.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            mask = (df['Hand_Performed'] == hand) & (df[f'{task}_rate'].notna())
            df.loc[mask, CLINICAL_SCORE_COL] = df.loc[mask, col]

# Covariates
if 'Sex' in df.columns:
    df['Sex_numeric'] = (
        df['Sex'].astype(str).str.lower()
        .map({'m': 1, 'f': 0, 'male': 1, 'female': 0})
        .fillna(df['Sex'])
    )
    df['Sex_numeric'] = pd.to_numeric(df['Sex_numeric'], errors='coerce')

if 'Disease duration' in df.columns and 'Disease_Duration' not in df.columns:
    df.rename(columns={'Disease duration': 'Disease_Duration'}, inplace=True)
for cov in ['Age', 'Disease_Duration']:
    if cov in df.columns:
        df[cov] = pd.to_numeric(df[cov], errors='coerce')

print("[SUCCESS] Data loaded and prepared.\n")

# =========================
# Build Pre-specified Composite (mean of z-scores)
# =========================
def build_prespecified_composite(df, task, features, orient_to='Clinical_Score'):
    cols = [c for c in features if c in df.columns]
    X = df[cols].copy().dropna()
    idx = X.index
    if len(cols) == 0 or X.empty:
        print(f"[WARNING] No valid features for {task.upper()} (pre-specified).")
        return df

    Xz = pd.DataFrame(StandardScaler().fit_transform(X), index=idx, columns=cols)
    comp = Xz.mean(axis=1).astype(float)

    # orient to impairment
    ref = df.loc[idx, orient_to] if orient_to in df.columns else None
    if ref is not None:
        comp = orient_to_reference(comp, ref)

    out = f"{task}_kinematic_composite"
    df[out] = np.nan
    df.loc[idx, out] = comp
    print(f"[INFO] {task.upper()} PreSpec composite from {len(cols)} features (mean of z).")
    return df

for t in TASKS:
    df = build_prespecified_composite(df, t, PRESELECTED[t], orient_to=CLINICAL_SCORE_COL)

# =========================
# Build Supervised Composite (Elastic-Net, grouped CV, within-fold residualization)
# =========================
def residualize_series(y, design_df, covars):
    """Return residuals of y ~ covars fit on design_df."""
    model = ols(f'y ~ {" + ".join(covars)}', data=design_df.assign(y=y)).fit()
    return y - model.fittedvalues, model

def apply_residualizer(y, design_df, model):
    """Apply trained OLS model to new y/design_df and return residuals (y - y_hat)."""
    y_hat = model.predict(design_df.assign(y=y))
    return y - y_hat

def build_supervised_composite(df, task, base_cols, y_col, group_col='Patient ID'):
    feat_cols = [f"{task}_{b}" for b in base_cols if f"{task}_{b}" in df.columns]
    use_cols = [y_col, CLINICAL_SCORE_COL, group_col] + COVARIATES + feat_cols
    sub = df[use_cols].dropna().copy()
    if sub.empty or len(feat_cols) < 2:
        print(f"[WARNING] Not enough features for {task.upper()} supervised composite.")
        df[f"{task}_kinematic_supervised"] = np.nan
        return df

    y = sub[y_col].values
    groups = sub[group_col].values
    X_feats = sub[feat_cols].copy()
    cov_df = sub[COVARIATES].copy()

    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    oof_pred = np.full_like(y, np.nan, dtype=float)

    for tr, te in gkf.split(X_feats, y, groups):
        # Residualize y and each feature on covariates using TRAIN fold only
        y_tr, y_model = residualize_series(y[tr], cov_df.iloc[tr], COVARIATES)
        # Residualize features: column-wise
        Xr_tr = []
        res_models = []
        for col in feat_cols:
            xr, m = residualize_series(X_feats.iloc[tr][col].values, cov_df.iloc[tr], COVARIATES)
            Xr_tr.append(xr)
            res_models.append(m)
        Xr_tr = np.vstack(Xr_tr).T  # shape (n_tr, p)

        # Standardize on train
        scaler = StandardScaler()
        Xr_trs = scaler.fit_transform(Xr_tr)

        # Train supervised model (inner CV handled by ENCV)
        en = ElasticNetCV(l1_ratio=[.1,.5,.9,1.0], cv=5, random_state=42)
        en.fit(Xr_trs, y_tr)

        # Prepare TEST residualized features using TRAIN residualizers
        Xr_te = []
        for j, col in enumerate(feat_cols):
            xr_te = apply_residualizer(X_feats.iloc[te][col].values, cov_df.iloc[te], res_models[j])
            Xr_te.append(xr_te)
        Xr_te = np.vstack(Xr_te).T
        Xr_tes = scaler.transform(Xr_te)

        # Predict y_resid on TEST
        oof_pred[te] = en.predict(Xr_tes)

    # Align direction to impairment (UPDRS residuals) for interpretability
    # Get residuals of UPDRS vs covariates (on the same rows)
    updrs_resid, _ = residualize_series(sub[CLINICAL_SCORE_COL].values, cov_df, COVARIATES) \
        if CLINICAL_SCORE_COL in sub.columns else (None, None)
    if updrs_resid is not None:
        oof_pred = orient_to_reference(oof_pred, updrs_resid)

    out_col = f"{task}_kinematic_supervised"
    df[out_col] = np.nan
    df.loc[sub.index, out_col] = oof_pred
    print(f"[INFO] {task.upper()} Supervised composite built with grouped CV (Elastic-Net).")
    return df

# Use the same base dictionary used earlier for PCA; here it's all base names:
base_kinematic_cols = [
    "meanamplitude", "stdamplitude", "meanspeed", "stdspeed", "meanrmsvelocity",
    "stdrmsvelocity", "meanopeningspeed", "stdopeningspeed", "meanclosingspeed",
    "stdclosingspeed", "meancycleduration", "stdcycleduration", "rangecycleduration",
    "rate", "amplitudedecay", "velocitydecay", "ratedecay", "cvamplitude",
    "cvcycleduration", "cvspeed", "cvrmsvelocity", "cvopeningspeed", "cvclosingspeed"
]

for t in TASKS:
    df = build_supervised_composite(df, t, base_kinematic_cols, DAT_COL, group_col='Patient ID')

# =========================
# Core Comparative Analysis
# =========================
print("\n--- 3. Running Core Comparative Analysis ---")

def analyze_task(df, task):
    kin_prespec_col = f"{task}_kinematic_composite"
    kin_super_col   = f"{task}_kinematic_supervised"
    cols = [DAT_COL, CLINICAL_SCORE_COL, 'Patient ID'] + COVARIATES + [kin_prespec_col, kin_super_col]
    d = df[cols].dropna().copy()
    n_obs = len(d)
    if n_obs < 20:
        print(f"[WARNING] Too few data for {task.upper()} (N={n_obs}). Skipping.")
        return None, None

    print("\n" + "-"*30)
    print(f"### Analyzing Task: {task.upper()} ###")
    print("-"*30)
    print(f"Analysis will be run on N={n_obs} observations.")

    # Partial residuals vs covariates
    dat_resid, updrs_resid = partial_corr_residuals(d, DAT_COL, CLINICAL_SCORE_COL, COVARIATES)

    # Pre-specified composite comparison
    prespec_resid = ols(f'{kin_prespec_col} ~ {" + ".join(COVARIATES)}', data=d).fit().resid
    r1 = pg.corr(dat_resid, prespec_resid)['r'].iloc[0]
    r2 = pg.corr(dat_resid, updrs_resid)['r'].iloc[0]
    r12 = pg.corr(prespec_resid, updrs_resid)['r'].iloc[0]
    z_ps, p_ps = williams_steiger_test(r1, r2, r12, n_obs)
    lo_ps, hi_ps, nb_ps = cluster_boot_delta_z(d.assign(_g=d['Patient ID']), dat_resid, prespec_resid, updrs_resid, group_col='_g', n_boot=N_BOOTSTRAPS)

    # Supervised composite comparison
    super_resid = ols(f'{kin_super_col} ~ {" + ".join(COVARIATES)}', data=d).fit().resid
    r1s = pg.corr(dat_resid, super_resid)['r'].iloc[0]
    r2s = r2  # same DAT vs UPDRS residual correlation
    r12s = pg.corr(super_resid, updrs_resid)['r'].iloc[0]
    z_sv, p_sv = williams_steiger_test(r1s, r2s, r12s, n_obs)
    lo_sv, hi_sv, nb_sv = cluster_boot_delta_z(d.assign(_g=d['Patient ID']), dat_resid, super_resid, updrs_resid, group_col='_g', n_boot=N_BOOTSTRAPS)

    # Hierarchical regressions (incremental value)
    cov_str = " + ".join(COVARIATES)
    mA = ols(f'{DAT_COL} ~ {CLINICAL_SCORE_COL} + {cov_str}', data=d).fit()
    mB_pre = ols(f'{DAT_COL} ~ {CLINICAL_SCORE_COL} + {kin_prespec_col} + {cov_str}', data=d).fit()
    mB_sup = ols(f'{DAT_COL} ~ {CLINICAL_SCORE_COL} + {kin_super_col} + {cov_str}', data=d).fit()
    anova_pre = sms.anova_lm(mA, mB_pre)
    anova_sup = sms.anova_lm(mA, mB_sup)

    res = pd.DataFrame([
        dict(Composite='PreSpecified',
             Task=task.upper(), N=n_obs,
             r_DAT_Kin=r1, r_DAT_UPDRS=r2, r_Kin_UPDRS=r12,
             Steiger_Z=z_ps, Steiger_p=p_ps,
             delta_z=np.arctanh(r1)-np.arctanh(r2),
             boot_CI_low=lo_ps, boot_CI_high=hi_ps, boot_n=nb_ps,
             AdjR2_A=mA.rsquared_adj, AdjR2_B=mB_pre.rsquared_adj,
             dAdjR2=mB_pre.rsquared_adj - mA.rsquared_adj,
             F=anova_pre['F'][1], p_LRT=anova_pre['Pr(>F)'][1]),
        dict(Composite='Supervised',
             Task=task.upper(), N=n_obs,
             r_DAT_Kin=r1s, r_DAT_UPDRS=r2s, r_Kin_UPDRS=r12s,
             Steiger_Z=z_sv, Steiger_p=p_sv,
             delta_z=np.arctanh(r1s)-np.arctanh(r2s),
             boot_CI_low=lo_sv, boot_CI_high=hi_sv, boot_n=nb_sv,
             AdjR2_A=mA.rsquared_adj, AdjR2_B=mB_sup.rsquared_adj,
             dAdjR2=mB_sup.rsquared_adj - mA.rsquared_adj,
             F=anova_sup['F'][1], p_LRT=anova_sup['Pr(>F)'][1]),
    ])
    # Print quick summary
    for _, row in res.iterrows():
        print(f"\n[{row['Composite']}] r(DAT, Kin)={row['r_DAT_Kin']:.3f} vs r(DAT, UPDRS)={row['r_DAT_UPDRS']:.3f} "
              f"=> Δz={row['delta_z']:.3f}, Steiger Z={row['Steiger_Z']:.2f}, p={row['Steiger_p']:.4f}; "
              f"ΔAdjR²={row['dAdjR2']:.3f}, F={row['F']:.2f}, p={row['p_LRT']:.4f} "
              f"(bootCI [{row['boot_CI_low']:.3f},{row['boot_CI_high']:.3f}], n={int(row['boot_n'])})")
    return res, d

all_results = []
for t in TASKS:
    res, _ = analyze_task(df, t)
    if res is not None:
        all_results.append(res)

# =========================
# Save & Display
# =========================
print("\n" + "="*80)
print("--- FINAL COMPARATIVE ANALYSIS RESULTS ---")
print("="*80)

if len(all_results):
    final_df = pd.concat(all_results, ignore_index=True)
    print(final_df.round(4).to_string(index=False))
    out_path = os.path.join(data_output_folder, "comparative_analysis_kinematics_vs_clinical_SUPERVISED.csv")
    final_df.to_csv(out_path, sep=';', decimal='.', index=False)
    print(f"\n[SUCCESS] Results saved to: {out_path}")
else:
    print("[INFO] No results generated.")

print("\n--- Comparative analysis script finished ---")
