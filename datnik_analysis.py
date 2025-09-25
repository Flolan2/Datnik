# --- START OF FILE datnik_analysis.py (CORRECTED) ---

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core analysis functions for PLS, ElasticNet, and Bivariate Correlations.
These functions now expect pre-processed X and y data for multivariate models.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score
import numpy as np
import pingouin as pg
from statsmodels.stats.multitest import multipletests

# ==============================================================================
# --- BIVARIATE ANALYSIS (Unchanged) ---
# ==============================================================================
def run_correlation_analysis(df, kinematic_cols, imaging_col, significance_alpha=0.05, fdr_method='fdr_bh'):
    results = []
    for col in kinematic_cols:
        subset = df[[col, imaging_col]].dropna()
        if len(subset) > 3:
            corr_result = pg.corr(subset[col], subset[imaging_col])
            r = corr_result['r'].iloc[0]
            p_val = corr_result['p-val'].iloc[0]
            n = corr_result['n'].iloc[0]
            results.append({'Kinematic Variable': col, 'Correlation (r)': r, 'P-value (uncorrected)': p_val, 'N': n})
    if not results: return pd.DataFrame(), {}
    results_df = pd.DataFrame(results)
    p_values = results_df['P-value (uncorrected)']
    reject, q_values, _, _ = multipletests(p_values, alpha=significance_alpha, method=fdr_method)
    results_df['Q-value (FDR corrected)'] = q_values
    results_df['Significant (FDR)'] = reject
    significant_results = results_df[results_df['Significant (FDR)']].copy()
    return results_df, significant_results

# ==============================================================================
# --- PLS ANALYSIS (MODIFIED TO FIX BOOTSTRAP SIGN FLIPPING LOGIC) ---
# ==============================================================================
def run_pls_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    task_prefix: str,
    max_components: int = 5,
    n_permutations: int = 1000,
    n_bootstraps: int = 1000,
    alpha: float = 0.05
):
    """
    Performs PLS Correlation on pre-processed data.
    """
    if X.empty or y.empty:
        print("  PLS Warning: Input X or y data is empty. Skipping.")
        return None

    scaler_X = StandardScaler(); X_scaled = scaler_X.fit_transform(X)
    scaler_y = StandardScaler(); y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

    lv_results = {}; significant_lvs = []
    for n_comp in range(1, max_components + 1):
        pls = PLSRegression(n_components=n_comp, scale=False); pls.fit(X_scaled, y_scaled)
        x_scores, y_scores = pls.transform(X_scaled, y_scaled.reshape(-1, 1))
        corr_lv = np.corrcoef(x_scores[:, -1], y_scores[:, -1])[0, 1]
        perm_corrs = []
        y_perm = np.copy(y_scaled)
        for _ in range(n_permutations):
            np.random.shuffle(y_perm)
            pls_perm = PLSRegression(n_components=n_comp, scale=False); pls_perm.fit(X_scaled, y_perm)
            x_scores_p, y_scores_p = pls_perm.transform(X_scaled, y_perm.reshape(-1,1))
            perm_corrs.append(np.corrcoef(x_scores_p[:, -1], y_scores_p[:, -1])[0, 1])
        p_value = (np.sum(np.abs(perm_corrs) >= np.abs(corr_lv))) / n_permutations
        is_significant = p_value < alpha
        lv_results[n_comp] = {'correlation': corr_lv, 'p_value': p_value, 'significant': is_significant}
        if is_significant and not significant_lvs:
            significant_lvs.append(n_comp)

    if not significant_lvs:
        return {'task': task_prefix, 'significant_lvs': [], 'lv_results': lv_results, 'kinematic_variables': X.columns.tolist(), 'n_samples_pls': len(X)}

    first_sig_lv_n = significant_lvs[0]
    pls_final = PLSRegression(n_components=first_sig_lv_n, scale=False); pls_final.fit(X_scaled, y_scaled)
    original_loadings = pd.Series(pls_final.x_loadings_[:, first_sig_lv_n - 1], index=X.columns)
    lv_results[first_sig_lv_n]['x_loadings'] = original_loadings
    lv_results[first_sig_lv_n]['y_loadings'] = pls_final.y_loadings_[first_sig_lv_n - 1, 0]

    boot_loadings = []
    for _ in range(n_bootstraps):
        boot_indices = np.random.choice(len(X_scaled), size=len(X_scaled), replace=True)
        X_boot, y_boot = X_scaled[boot_indices], y_scaled[boot_indices]
        pls_boot = PLSRegression(n_components=first_sig_lv_n, scale=False); pls_boot.fit(X_boot, y_boot)
        boot_loadings.append(pls_boot.x_loadings_[:, first_sig_lv_n - 1])
    
    boot_loadings_arr = np.array(boot_loadings)
    
    # ==========================================================================
    # --- THIS IS THE CORRECTED LOGIC FOR SIGN FLIPPING ---
    # ==========================================================================
    # Use dot product to check alignment of each bootstrap vector with the original
    signs = np.sign(np.dot(boot_loadings_arr, original_loadings))
    # Handle rare case where dot product is exactly zero, default to positive sign
    signs[signs == 0] = 1
    # Apply correction by broadcasting the signs
    boot_loadings_arr *= signs[:, np.newaxis]
    # ==========================================================================
    
    std_error = np.std(boot_loadings_arr, axis=0)
    # Avoid division by zero for features with no variance in bootstrap
    std_error[std_error == 0] = 1e-9 
    bootstrap_ratios = original_loadings / std_error
    lv_results[first_sig_lv_n]['bootstrap_ratios'] = pd.Series(bootstrap_ratios, index=X.columns)
    
    return {'task': task_prefix, 'significant_lvs': significant_lvs, 'lv_results': lv_results, 'kinematic_variables': X.columns.tolist(), 'n_samples_pls': len(X)}

# ==============================================================================
# --- ELASTICNET ANALYSIS (Unchanged) ---
# ==============================================================================
def run_elasticnet_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    task_prefix: str,
    l1_ratios,
    cv_folds: int,
    max_iter: int,
    random_state: int
):
    """
    Performs ElasticNet Regression on pre-processed data.
    """
    if X.empty or y.empty:
        print("  ElasticNet Warning: Input X or y data is empty. Skipping.")
        return None

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    enet_cv = ElasticNetCV(
        l1_ratio=l1_ratios, cv=cv_folds, random_state=random_state, max_iter=max_iter, n_jobs=-1
    )
    enet_cv.fit(X_scaled, y)

    coefficients = pd.Series(enet_cv.coef_, index=X.columns)
    y_pred = enet_cv.predict(X_scaled)
    r2_val = r2_score(y, y_pred)
    
    results = {
        'task': task_prefix,
        'imaging_variable': y.name,
        'n_samples_enet': len(X),
        'coefficients': coefficients,
        'performance': {
            'alpha': enet_cv.alpha_,
            'l1_ratio': enet_cv.l1_ratio_,
            'R2': r2_val
        }
    }
    return results