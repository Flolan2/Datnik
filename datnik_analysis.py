# --- START OF UPDATED FILE datnik_analysis.py ---

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from sklearn.cross_decomposition import PLSCanonical
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import time # For timing long operations

# <<< ADDED IMPORTS for PCR >>>
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm # For p-values of regression coefficients
from sklearn.linear_model import ElasticNetCV # Added for ElasticNet


def run_correlation_analysis(
    df: pd.DataFrame,
    base_kinematic_cols: list,
    task_prefix: str,
    imaging_base_name: str = "Contralateral_Striatum",
    alpha: float = 0.05,
    patient_id_col: str = "Patient ID" # <<< NEW ARGUMENT >>>
) -> tuple: # <<< MODIFIED RETURN TYPE HINT >>>
    """
    Performs bivariate Pearson correlations between kinematic variables for a specific
    task and the specified imaging Z-SCORE measure. Applies FDR correction.
    Tracks and returns Patient IDs used for each correlation.

    Args:
        df (pd.DataFrame): Input dataframe.
        base_kinematic_cols (list): List of base kinematic variable names.
        task_prefix (str): 'ft' or 'hm'.
        imaging_base_name (str): Base name of the imaging variable (e.g., "Contralateral_Striatum").
        alpha (float): Significance level for FDR correction.
        patient_id_col (str): Name of the column containing patient identifiers.

    Returns:
        tuple: (significant_results_df, patient_ids_per_correlation)
               significant_results_df (pd.DataFrame): DataFrame of significant correlations.
               patient_ids_per_correlation (dict): Dictionary where keys are
                                                   (kinematic_col, img_col) tuples and
                                                   values are lists of Patient IDs
                                                   used for that specific correlation.
    """
    results_list = []
    img_col = f"{imaging_base_name}_Z"
    patient_ids_per_correlation = {} # Initialize dict to store patient IDs

    print(f"\n--- Running Correlation Analysis for Task: {task_prefix} vs {img_col} ---")
    significant_results_df = pd.DataFrame() # Initialize an empty DataFrame for the return value

    if df.empty:
        print("Warning: Input DataFrame is empty. Cannot run analysis.")
        return significant_results_df, patient_ids_per_correlation

    if img_col not in df.columns:
        print(f"Warning: Target imaging column '{img_col}' not found in DataFrame. Cannot run analysis.")
        return significant_results_df, patient_ids_per_correlation
    
    if patient_id_col not in df.columns:
        print(f"Warning: Patient ID column '{patient_id_col}' not found. Cannot track patient IDs for correlations.")
        # Proceeding, but IDs won't be tracked effectively if column is missing.

    print(f"Target Imaging Column: {img_col}")

    for base_col in base_kinematic_cols:
        kinematic_col = f"{task_prefix}_{base_col}"
        if kinematic_col not in df.columns: continue

        # Select columns needed for this pair, including Patient ID if available
        cols_to_select_for_pair = [kinematic_col, img_col]
        if patient_id_col in df.columns:
            cols_to_select_for_pair.append(patient_id_col)
        
        data_pair_full = df[cols_to_select_for_pair].copy()

        # Convert kinematic and imaging columns to numeric, coercing errors
        data_pair_full[kinematic_col] = pd.to_numeric(data_pair_full[kinematic_col].astype(str).str.replace(',', '.'), errors='coerce')
        data_pair_full[img_col] = pd.to_numeric(data_pair_full[img_col].astype(str).str.replace(',', '.'), errors='coerce')
        
        # Drop rows where EITHER the kinematic_col OR the img_col is NaN (these rows cannot be used for correlation)
        data_pair_cleaned_for_corr = data_pair_full.dropna(subset=[kinematic_col, img_col])
        
        n_samples = len(data_pair_cleaned_for_corr)
        if n_samples < 3: continue # Need at least 3 samples for correlation
        
        try:
            # Perform correlation on the cleaned numeric columns
            corr_coef, p_value = pearsonr(data_pair_cleaned_for_corr[kinematic_col], data_pair_cleaned_for_corr[img_col])
            
            if pd.notna(corr_coef) and pd.notna(p_value):
                results_list.append({
                    "Task": task_prefix, "Base Kinematic": base_col,
                    "Kinematic Variable": kinematic_col, "Imaging Variable": img_col,
                    "Pearson Correlation (r)": corr_coef,
                    "P-value (uncorrected)": p_value, "N": n_samples
                })
                # Store patient IDs for this specific correlation if patient_id_col was present and valid
                if patient_id_col in data_pair_cleaned_for_corr.columns:
                    # Get unique patient IDs from the rows that were actually used in this correlation
                    patient_ids_for_this_corr = data_pair_cleaned_for_corr[patient_id_col].astype(str).unique().tolist()
                    patient_ids_per_correlation[(kinematic_col, img_col)] = sorted(patient_ids_for_this_corr)
        except ValueError:
            # Handle cases where correlation cannot be computed (e.g., constant input)
            continue

    if not results_list:
        print(f"No valid correlations could be calculated for task {task_prefix}.")
        return significant_results_df, patient_ids_per_correlation

    results_df = pd.DataFrame(results_list)

    # Apply FDR correction
    try:
        reject, pvals_corrected, _, _ = multipletests(
            results_df["P-value (uncorrected)"].fillna(1.0), # Fill NaN p-values with 1 before correction
             alpha=alpha, method='fdr_bh'
        )
        results_df['Q-value (FDR corrected)'] = pvals_corrected
        results_df['Significant (FDR)'] = reject
    except Exception as e:
        print(f"Warning: FDR correction failed for task {task_prefix}. Error: {e}")
        results_df['Q-value (FDR corrected)'] = np.nan
        results_df['Significant (FDR)'] = False

    # Filter based on FDR significance
    significant_results_df = results_df[results_df['Significant (FDR)'] == True].copy()

    print(f"Found {len(significant_results_df)} significant correlations (q <= {alpha}) for task {task_prefix} after FDR correction.")
    print("------------------------------------------------------------------")

    if not significant_results_df.empty: # Sort if not empty
      significant_results_df.sort_values(by='Q-value (FDR corrected)', inplace=True)

    return significant_results_df, patient_ids_per_correlation


def run_ridge_analysis(
    df: pd.DataFrame,
    base_kinematic_cols: list,
    task_prefix: str,
    imaging_col: str,
    alphas = (0.1, 1.0, 10.0, 100.0, 1000.0),
    cv_folds: int = 5
) -> dict:
    """
    Performs Ridge Regression analysis... (rest of docstring)
    """
    print(f"\n--- Running Ridge Regression Analysis for Task: {task_prefix} vs {imaging_col} ---")
    kinematic_cols = [f"{task_prefix}_{base}" for base in base_kinematic_cols]
    valid_kinematic_cols = [col for col in kinematic_cols if col in df.columns]

    if not valid_kinematic_cols or imaging_col not in df.columns:
        print("Warning: Missing kinematic or imaging columns for Ridge. Skipping.")
        return None

    ridge_data = df[valid_kinematic_cols + [imaging_col]].copy()
    for col in ridge_data.columns:
        ridge_data[col] = pd.to_numeric(ridge_data[col].astype(str).str.replace(',', '.'), errors='coerce')

    ridge_data.dropna(inplace=True)
    n_samples_ridge = len(ridge_data)
    n_features = len(valid_kinematic_cols)

    if n_samples_ridge < n_features or n_samples_ridge < cv_folds or n_samples_ridge < 10:
         print(f"Warning: Insufficient samples (N={n_samples_ridge}) relative to features ({n_features}) or CV folds ({cv_folds}) for Ridge. Skipping.")
         return None

    X = ridge_data[valid_kinematic_cols].values
    y = ridge_data[imaging_col].values
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    try:
        print(f"Fitting RidgeCV (folds={cv_folds}, testing alphas={alphas})...")
        ridge_cv = RidgeCV(alphas=alphas, cv=cv_folds, scoring='neg_mean_squared_error')
        ridge_cv.fit(X_scaled, y)
        optimal_alpha = ridge_cv.alpha_
        coefficients = ridge_cv.coef_
        intercept = ridge_cv.intercept_
        y_pred = ridge_cv.predict(X_scaled)
        r2_full = r2_score(y, y_pred)
        rmse_full = np.sqrt(mean_squared_error(y, y_pred))
        print(f"RidgeCV completed. Optimal alpha: {optimal_alpha:.4f}")
        print(f"Model fit (on full data used for CV): R^2 = {r2_full:.4f}, RMSE = {rmse_full:.4f}")
        coeffs_series = pd.Series(coefficients, index=valid_kinematic_cols)
    except Exception as e:
        print(f"Error during RidgeCV fitting: {e}")
        return None

    final_results = {
        'task': task_prefix, 'n_samples_ridge': n_samples_ridge,
        'kinematic_variables': valid_kinematic_cols, 'imaging_variable': imaging_col,
        'optimal_alpha': optimal_alpha, 'coefficients': coeffs_series,
        'intercept': intercept, 'r2_full_data': r2_full, 'rmse_full_data': rmse_full
    }
    print(f"--- Ridge Regression Analysis Finished for Task {task_prefix} ---")
    return final_results


def run_pls_analysis(
    df: pd.DataFrame,
    base_kinematic_cols: list,
    task_prefix: str,
    imaging_col: str,
    max_components: int = 5,
    n_permutations: int = 1000,
    n_bootstraps: int = 1000,
    alpha: float = 0.05
) -> dict:
    """
    Performs PLS Correlation analysis... (rest of docstring)
    """
    print(f"\n--- Running PLS Analysis for Task: {task_prefix} vs {imaging_col} (Max LVs: {max_components}) ---")
    kinematic_cols = [f"{task_prefix}_{base}" for base in base_kinematic_cols]
    valid_kinematic_cols = [col for col in kinematic_cols if col in df.columns]

    if not valid_kinematic_cols: print("Warning: No valid kinematic columns found for PLS. Skipping."); return None
    if imaging_col not in df.columns: print(f"Warning: Imaging column '{imaging_col}' not found for PLS. Skipping."); return None

    pls_data = df[valid_kinematic_cols + [imaging_col]].copy()
    for col in pls_data.columns:
        pls_data[col] = pd.to_numeric(pls_data[col].astype(str).str.replace(',', '.'), errors='coerce')
    pls_data.dropna(inplace=True)
    n_samples_pls = len(pls_data)
    n_features_x = len(valid_kinematic_cols); n_features_y = 1
    effective_max_components = min(max_components, n_samples_pls, n_features_x, n_features_y)

    if n_samples_pls < 10 or effective_max_components == 0:
        print(f"Warning: Insufficient samples (N={n_samples_pls}) or features relative to max_components for PLS. Skipping."); return None
    if effective_max_components < max_components:
         print(f"Note: Reduced max components to test from {max_components} to {effective_max_components} due to data dimensions (N={n_samples_pls}, K_x={n_features_x}).")

    X = pls_data[valid_kinematic_cols].values; Y = pls_data[[imaging_col]].values
    scaler_X = StandardScaler(); scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X); Y_scaled = scaler_Y.fit_transform(Y)

    try:
        print(f"Fitting initial PLS model with {effective_max_components} components...")
        pls = PLSCanonical(n_components=effective_max_components, scale=False)
        pls.fit(X_scaled, Y_scaled)
        x_scores_orig, y_scores_orig = pls.transform(X_scaled, Y_scaled)
    except Exception as e: print(f"Error fitting initial PLS model: {e}"); return None

    print(f"Running {n_permutations} permutations to test {effective_max_components} LVs sequentially...")
    perm_start_time = time.time()
    lv_p_values = np.full(effective_max_components, np.nan)
    lv_correlations = np.full(effective_max_components, np.nan)
    significant_lvs_indices = []
    y_shuffled = Y_scaled.copy()

    for lv_idx in range(effective_max_components):
        try: r_actual, _ = pearsonr(x_scores_orig[:, lv_idx], y_scores_orig[:, lv_idx]); lv_correlations[lv_idx] = r_actual if pd.notna(r_actual) else np.nan
        except ValueError: r_actual = 0.0; lv_correlations[lv_idx] = 0.0; print(f"Warning: Could not calculate correlation for LV{lv_idx+1}. Setting r=0.")
        perm_corrs_lv = np.zeros(n_permutations)
        print(f"  Permuting for LV{lv_idx+1} (Actual r={r_actual:.4f})...", end="", flush=True); perm_loop_start = time.time()
        for i in range(n_permutations):
            np.random.shuffle(y_shuffled)
            try:
                pls_perm = PLSCanonical(n_components=effective_max_components, scale=False); pls_perm.fit(X_scaled, y_shuffled)
                x_scores_perm, y_scores_perm = pls_perm.transform(X_scaled, y_shuffled)
                r_perm, _ = pearsonr(x_scores_perm[:, lv_idx], y_scores_perm[:, lv_idx]); perm_corrs_lv[i] = r_perm if pd.notna(r_perm) else 0.0
            except Exception: perm_corrs_lv[i] = np.nan
        perm_loop_end = time.time(); print(f" done ({perm_loop_end - perm_loop_start:.1f}s).")
        valid_perm_corrs_lv = perm_corrs_lv[~np.isnan(perm_corrs_lv)]
        if len(valid_perm_corrs_lv) < n_permutations * 0.8: print(f"Warning: High number of failed permutations for LV{lv_idx+1}.")
        if len(valid_perm_corrs_lv) == 0 or np.isnan(r_actual): p_value_lv = 1.0
        else: p_value_lv = (np.sum(np.abs(valid_perm_corrs_lv) >= np.abs(r_actual)) + 1) / (len(valid_perm_corrs_lv) + 1)
        lv_p_values[lv_idx] = p_value_lv; print(f"  LV{lv_idx+1}: p = {p_value_lv:.4f}")
        if p_value_lv <= alpha: significant_lvs_indices.append(lv_idx)
        else: print(f"  LV{lv_idx+1} is not significant. Stopping permutation tests."); break
    perm_end_time = time.time(); print(f"Permutation testing finished ({perm_end_time - perm_start_time:.1f}s). Found {len(significant_lvs_indices)} significant LVs.")

    all_bootstrap_ratios = {}
    if significant_lvs_indices and n_bootstraps > 0:
        print(f"Running {n_bootstraps} bootstraps for {len(significant_lvs_indices)} significant LVs..."); boot_start_time = time.time()
        x_loadings_boot_all_lvs = {lv_idx: [] for lv_idx in significant_lvs_indices}; n_boot_success = 0
        for i in range(n_bootstraps):
            indices = resample(np.arange(n_samples_pls)); X_boot, Y_boot = X_scaled[indices], Y_scaled[indices]
            try:
                pls_boot = PLSCanonical(n_components=effective_max_components, scale=False); pls_boot.fit(X_boot, Y_boot)
                for lv_idx in significant_lvs_indices:
                     if lv_idx < pls_boot.x_loadings_.shape[1]: x_loadings_boot_all_lvs[lv_idx].append(pls_boot.x_loadings_[:, lv_idx])
                n_boot_success += 1
            except Exception: continue
        print(f"  Finished {n_boot_success}/{n_bootstraps} successful bootstrap iterations.")
        for lv_idx in significant_lvs_indices:
            loadings_boot_lv = x_loadings_boot_all_lvs[lv_idx]
            if len(loadings_boot_lv) > 1:
                loadings_boot_lv = np.array(loadings_boot_lv); original_loadings_lv = pls.x_loadings_[:, lv_idx]
                loadings_std_lv = np.std(loadings_boot_lv, axis=0, ddof=1); non_zero_std_mask_lv = loadings_std_lv > 1e-9
                bsr_lv = np.full_like(original_loadings_lv, np.nan)
                bsr_lv[non_zero_std_mask_lv] = original_loadings_lv[non_zero_std_mask_lv] / loadings_std_lv[non_zero_std_mask_lv]
                all_bootstrap_ratios[lv_idx] = pd.Series(bsr_lv, index=valid_kinematic_cols)
            else: print(f"Warning: Insufficient bootstraps for BSR LV{lv_idx+1}."); all_bootstrap_ratios[lv_idx] = pd.Series(np.nan, index=valid_kinematic_cols)
        boot_end_time = time.time(); print(f"Bootstrapping finished ({boot_end_time - boot_start_time:.1f}s).")
    elif not significant_lvs_indices: print("No significant LVs found, skipping bootstrapping.")
    else: print("n_bootstraps <= 0, skipping bootstrapping.")

    final_results = {
        'task': task_prefix, 'significant_lvs': [idx + 1 for idx in significant_lvs_indices], 'lv_results': {},
        'kinematic_variables': valid_kinematic_cols, 'n_samples_pls': n_samples_pls, 'max_components_tested': effective_max_components
    }
    for lv_idx in range(effective_max_components):
        lv_num = lv_idx + 1; is_significant = lv_idx in significant_lvs_indices
        final_results['lv_results'][lv_num] = {
            'significant': is_significant, 'p_value': lv_p_values[lv_idx], 'correlation': lv_correlations[lv_idx],
            'x_loadings': pd.Series(pls.x_loadings_[:, lv_idx], index=valid_kinematic_cols),
            'y_loadings': pls.y_loadings_[0, lv_idx], 'bootstrap_ratios': all_bootstrap_ratios.get(lv_idx, None),
            'x_scores': x_scores_orig[:, lv_idx], 'y_scores': y_scores_orig[:, lv_idx]
        }
    print(f"--- PLS Analysis Finished for Task {task_prefix} ---")
    return final_results


def run_pcr_analysis(
    df: pd.DataFrame,
    base_kinematic_cols: list,
    task_prefix: str,
    imaging_col: str,
    n_components_pca: int = 10,
    alpha: float = 0.05
) -> dict:
    """
    Performs Principal Component Regression (PCR) analysis... (rest of docstring)
    """
    print(f"\n--- Running PCR Analysis for Task: {task_prefix} vs {imaging_col} ---")
    kinematic_cols = [f"{task_prefix}_{base}" for base in base_kinematic_cols]
    valid_kinematic_cols = [col for col in kinematic_cols if col in df.columns]

    if not valid_kinematic_cols or imaging_col not in df.columns: print("Warning: Missing kinematic or imaging columns for PCR. Skipping."); return None
    pcr_data = df[valid_kinematic_cols + [imaging_col]].copy()
    for col in pcr_data.columns: pcr_data[col] = pd.to_numeric(pcr_data[col].astype(str).str.replace(',', '.'), errors='coerce')
    pcr_data.dropna(inplace=True)
    n_samples_pcr = len(pcr_data); n_features = len(valid_kinematic_cols)
    effective_n_components = min(n_components_pca, n_samples_pcr -1 , n_features)
    if effective_n_components <= 0: print(f"Warning: Insufficient samples/features for PCA. Skipping PCR."); return None
    if effective_n_components < n_components_pca: print(f"Note: Reduced PCA components to {effective_n_components}.")

    X = pcr_data[valid_kinematic_cols].values; y = pcr_data[imaging_col].values
    scaler_X = StandardScaler(); X_scaled = scaler_X.fit_transform(X); y_target = y

    try:
        print(f"Performing PCA with {effective_n_components} components...")
        pca = PCA(n_components=effective_n_components); X_pca_scores = pca.fit_transform(X_scaled)
        explained_variance_ratio = pca.explained_variance_ratio_; pca_loadings = pca.components_
    except Exception as e: print(f"Error during PCA: {e}"); return None

    pca_results = {
        'n_components': effective_n_components, 'explained_variance_ratio': explained_variance_ratio.tolist(),
        'cumulative_explained_variance': np.cumsum(explained_variance_ratio).tolist(),
        'loadings': pd.DataFrame(pca_loadings.T, index=valid_kinematic_cols, columns=[f'PC{i+1}' for i in range(effective_n_components)])
    }
    print(f"PCA completed. Cumulative variance by {effective_n_components} PCs: {pca_results['cumulative_explained_variance'][-1]:.3f}")

    print("Performing regression: Y ~ PC1 + PC2 + ..."); X_pca_scores_with_const = sm.add_constant(X_pca_scores, has_constant='add')
    try:
        model = sm.OLS(y_target, X_pca_scores_with_const); results = model.fit()
        pc_names = [f'PC{i+1}' for i in range(effective_n_components)]; print(results.summary(xname=['const'] + pc_names))
        coefficients_dict = results.params.to_dict(); p_values_dict = results.pvalues.to_dict()
        conf_int_df = results.conf_int(); conf_int_dict = {col: conf_int_df[col].to_dict() for col in conf_int_df.columns}
        regression_summary = {
            'r_squared': results.rsquared, 'adj_r_squared': results.rsquared_adj, 'f_pvalue': results.f_pvalue,
            'coefficients': coefficients_dict, 'p_values': p_values_dict, 'conf_int': conf_int_dict
        }
        significant_pcs = [pc for pc, p in p_values_dict.items() if pc != 'const' and p < alpha]
        regression_summary['significant_pcs'] = significant_pcs
        print(f"\nSignificant PCs predicting {imaging_col} (alpha={alpha}): {significant_pcs if significant_pcs else 'None'}")
    except Exception as e: print(f"Error during OLS regression on PCs: {e}"); regression_summary = None

    final_results = {
        'task': task_prefix, 'n_samples_pcr': n_samples_pcr, 'kinematic_variables': valid_kinematic_cols,
        'imaging_variable': imaging_col, 'pca_results': pca_results, 'regression_results': regression_summary
    }
    print(f"--- PCR Analysis Finished for Task {task_prefix} ---")
    return final_results


def run_elasticnet_analysis(
    df: pd.DataFrame,
    base_kinematic_cols: list,
    task_prefix: str,
    imaging_col: str,
    l1_ratios = np.linspace(0.1, 1.0, 10),
    cv_folds: int = 5,
    max_iter: int = 10000,
    random_state: int = 42
) -> dict:
    """
    Performs ElasticNet Regression analysis... (rest of docstring)
    """
    print(f"\n--- Running ElasticNetCV Analysis for Task: {task_prefix} vs {imaging_col} ---")
    kinematic_cols = [f"{task_prefix}_{base}" for base in base_kinematic_cols]
    valid_kinematic_cols = [col for col in kinematic_cols if col in df.columns]

    if not valid_kinematic_cols or imaging_col not in df.columns: print("Warning: Missing columns for ElasticNet. Skipping."); return None
    enet_data = df[valid_kinematic_cols + [imaging_col]].copy()
    for col in enet_data.columns: enet_data[col] = pd.to_numeric(enet_data[col].astype(str).str.replace(',', '.'), errors='coerce')
    enet_data.dropna(inplace=True)
    n_samples_enet = len(enet_data); n_features = len(valid_kinematic_cols)

    if n_samples_enet < n_features or n_samples_enet < cv_folds or n_samples_enet < 10:
         print(f"Warning: Insufficient samples for ElasticNetCV. Skipping."); return None

    X = enet_data[valid_kinematic_cols].values; y = enet_data[imaging_col].values
    scaler_X = StandardScaler(); X_scaled = scaler_X.fit_transform(X)

    try:
        print(f"Fitting ElasticNetCV (folds={cv_folds}, testing {len(l1_ratios)} l1_ratios)...")
        enet_cv = ElasticNetCV(l1_ratio=l1_ratios, cv=cv_folds, random_state=random_state, max_iter=max_iter, n_jobs=-1)
        enet_cv.fit(X_scaled, y)
        optimal_alpha = enet_cv.alpha_; optimal_l1_ratio = enet_cv.l1_ratio_
        coefficients = enet_cv.coef_; intercept = enet_cv.intercept_
        y_pred = enet_cv.predict(X_scaled)
        r2_full = r2_score(y, y_pred); rmse_full = np.sqrt(mean_squared_error(y, y_pred))
        print(f"ElasticNetCV completed. Optimal alpha: {optimal_alpha:.6f}, Optimal l1_ratio: {optimal_l1_ratio:.2f}")
        print(f"Model fit (full data): R^2 = {r2_full:.4f}, RMSE = {rmse_full:.4f}")
        print(f"Number of features selected: {np.sum(coefficients != 0)} / {n_features}")
        coeffs_series = pd.Series(coefficients, index=valid_kinematic_cols)
    except Exception as e: print(f"Error during ElasticNetCV fitting: {e}"); return None

    final_results = {
        'task': task_prefix, 'n_samples_enet': n_samples_enet, 'kinematic_variables': valid_kinematic_cols,
        'imaging_variable': imaging_col, 'optimal_alpha': optimal_alpha, 'optimal_l1_ratio': optimal_l1_ratio,
        'coefficients': coeffs_series, 'intercept': intercept, 'r2_full_data': r2_full, 'rmse_full_data': rmse_full,
        'n_selected_features': int(np.sum(coefficients != 0))
    }
    print(f"--- ElasticNet Analysis Finished for Task {task_prefix} ---")
    return final_results

# --- END OF UPDATED FILE datnik_analysis.py ---