import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from sklearn.cross_decomposition import PLSCanonical
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import time # For timing long operations


def run_correlation_analysis(
    df: pd.DataFrame,
    base_kinematic_cols: list,
    task_prefix: str,
    imaging_base_name: str = "Contralateral_Striatum",
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Performs bivariate Pearson correlations between kinematic variables for a specific
    task and the specified imaging Z-SCORE measure. Applies FDR correction.

    Args:
        df (pd.DataFrame): Input dataframe.
        base_kinematic_cols (list): List of base kinematic variable names.
        task_prefix (str): 'ft' or 'hm'.
        imaging_base_name (str): Base name of the imaging variable (e.g., "Contralateral_Striatum").
        alpha (float): Significance level for FDR correction.

    Returns:
        pd.DataFrame: DataFrame containing only the significant correlations after FDR,
                      or an empty DataFrame if none are found or analysis fails.
    """
    results_list = []
    img_col = f"{imaging_base_name}_Z"

    print(f"\n--- Running Correlation Analysis for Task: {task_prefix} vs {img_col} ---")

    # --- Initialize an empty DataFrame for the return value ---
    significant_results_df = pd.DataFrame()

    if df.empty:
        print("Warning: Input DataFrame is empty. Cannot run analysis.")
        return significant_results_df # Return empty df

    if img_col not in df.columns:
        print(f"Warning: Target imaging column '{img_col}' not found in DataFrame. Cannot run analysis.")
        return significant_results_df # Return empty df
    print(f"Target Imaging Column: {img_col}")

    for base_col in base_kinematic_cols:
        kinematic_col = f"{task_prefix}_{base_col}"
        if kinematic_col not in df.columns: continue
        data_pair = df[[kinematic_col, img_col]].copy()
        # Convert to numeric, coercing errors
        data_pair[kinematic_col] = pd.to_numeric(data_pair[kinematic_col].astype(str).str.replace(',', '.'), errors='coerce')
        data_pair[img_col] = pd.to_numeric(data_pair[img_col].astype(str).str.replace(',', '.'), errors='coerce')
        data_pair.dropna(inplace=True)
        n_samples = len(data_pair)
        if n_samples < 3: continue # Need at least 3 samples for correlation
        try:
            corr_coef, p_value = pearsonr(data_pair[kinematic_col], data_pair[img_col])
            if pd.notna(corr_coef) and pd.notna(p_value):
                results_list.append({
                    "Task": task_prefix, "Base Kinematic": base_col,
                    "Kinematic Variable": kinematic_col, "Imaging Variable": img_col,
                    "Pearson Correlation (r)": corr_coef,
                    "P-value (uncorrected)": p_value, "N": n_samples
                })
        except ValueError:
            # Handle cases where correlation cannot be computed (e.g., constant input)
            continue

    if not results_list:
        print(f"No valid correlations could be calculated for task {task_prefix}.")
        return significant_results_df # Return empty df

    results_df = pd.DataFrame(results_list)

    # Apply FDR correction
    try: # Add try-except for multipletests robustness
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

    # --- Return the (potentially empty) significant results DataFrame ---
    return significant_results_df


# --- UPDATED FUNCTION for PLS Analysis ---
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
    Performs PLS Correlation analysis between a set of kinematic variables (X)
    and an imaging variable (Y). Determines significance of latent variables (LVs)
    using permutation testing sequentially, and assesses loading stability for
    significant LVs using bootstrapping.

    Args:
        df (pd.DataFrame): DataFrame containing kinematic and imaging data.
        base_kinematic_cols (list): Base names of kinematic variables.
        task_prefix (str): Prefix for the task (e.g., 'ft', 'hm').
        imaging_col (str): Column name for the single imaging variable (Y).
        max_components (int): Maximum number of latent variables to compute and test.
        n_permutations (int): Number of permutations for significance testing.
        n_bootstraps (int): Number of bootstrap samples for stability analysis.
        alpha (float): Significance level for permutation tests.

    Returns:
        dict: A dictionary containing PLS results, structured as:
              {
                  'task': str,
                  'significant_lvs': list[int], # List of 1-based indices of significant LVs
                  'lv_results': {
                      lv_index (int): { # Results for specific LV (1-based index)
                          'significant': bool,
                          'p_value': float,
                          'correlation': float,
                          'x_loadings': pd.Series | None,
                          'y_loadings': float | None, # Only one Y variable
                          'bootstrap_ratios': pd.Series | None,
                          'x_scores': np.array | None, # Scores for this LV
                          'y_scores': np.array | None  # Scores for this LV
                      },
                      ...
                  },
                  'kinematic_variables': list[str],
                  'n_samples_pls': int,
                  'max_components_tested': int
              }
              Returns None if analysis cannot be run (e.g., insufficient data).
    """
    print(f"\n--- Running PLS Analysis for Task: {task_prefix} vs {imaging_col} (Max LVs: {max_components}) ---")

    # --- 1. Prepare Data ---
    kinematic_cols = [f"{task_prefix}_{base}" for base in base_kinematic_cols]
    valid_kinematic_cols = [col for col in kinematic_cols if col in df.columns]

    if not valid_kinematic_cols:
        print("Warning: No valid kinematic columns found for PLS. Skipping.")
        return None
    if imaging_col not in df.columns:
        print(f"Warning: Imaging column '{imaging_col}' not found for PLS. Skipping.")
        return None

    pls_data = df[valid_kinematic_cols + [imaging_col]].copy()
    for col in pls_data.columns:
        # Ensure string conversion before replace, handle potential non-string types
        pls_data[col] = pd.to_numeric(pls_data[col].astype(str).str.replace(',', '.'), errors='coerce')

    pls_data.dropna(inplace=True)
    n_samples_pls = len(pls_data)

    # Determine effective max components based on data shape
    n_features_x = len(valid_kinematic_cols)
    n_features_y = 1 # Only one imaging variable
    # PLS requires N >= K, K >= n_components, P >= n_components, Q >= n_components
    effective_max_components = min(max_components, n_samples_pls, n_features_x, n_features_y)

    if n_samples_pls < 10 or effective_max_components == 0:
        print(f"Warning: Insufficient samples (N={n_samples_pls}) or features relative to max_components for PLS. Skipping.")
        return None
    if effective_max_components < max_components:
         print(f"Note: Reduced max components to test from {max_components} to {effective_max_components} due to data dimensions (N={n_samples_pls}, K_x={n_features_x}).")

    X = pls_data[valid_kinematic_cols].values
    Y = pls_data[[imaging_col]].values # Keep as 2D array for Scaler/PLS

    # Scale data
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y)

    # --- 2. Fit Original PLS Model ---
    try:
        print(f"Fitting initial PLS model with {effective_max_components} components...")
        pls = PLSCanonical(n_components=effective_max_components, scale=False) # Data is already scaled
        pls.fit(X_scaled, Y_scaled)
        # Get scores for ALL computed components
        x_scores_orig, y_scores_orig = pls.transform(X_scaled, Y_scaled) # Shape: (n_samples, n_components)
    except Exception as e:
        print(f"Error fitting initial PLS model: {e}")
        return None

    # --- 3. Permutation Testing Sequentially for LV Significance ---
    print(f"Running {n_permutations} permutations to test {effective_max_components} LVs sequentially...")
    perm_start_time = time.time()
    lv_p_values = np.full(effective_max_components, np.nan)
    lv_correlations = np.full(effective_max_components, np.nan)
    significant_lvs_indices = [] # Store 0-based indices here

    y_shuffled = Y_scaled.copy() # Use a copy for shuffling

    for lv_idx in range(effective_max_components):
        # Calculate actual correlation for this LV
        try:
            # Use original scores for actual correlation
            r_actual, _ = pearsonr(x_scores_orig[:, lv_idx], y_scores_orig[:, lv_idx])
            lv_correlations[lv_idx] = r_actual if pd.notna(r_actual) else np.nan
        except ValueError: # Handle potential constant score arrays
             r_actual = 0.0
             lv_correlations[lv_idx] = 0.0
             print(f"Warning: Could not calculate correlation for LV{lv_idx+1} (constant scores?). Setting r=0.")

        # Run permutations for this specific LV
        perm_corrs_lv = np.zeros(n_permutations)
        print(f"  Permuting for LV{lv_idx+1} (Actual r={r_actual:.4f})...", end="", flush=True)
        perm_loop_start = time.time()
        n_valid_perms = 0
        for i in range(n_permutations):
            np.random.shuffle(y_shuffled) # Shuffle Y each time
            try:
                # Re-fit PLS with shuffled Y
                pls_perm = PLSCanonical(n_components=effective_max_components, scale=False)
                pls_perm.fit(X_scaled, y_shuffled)
                # Get permuted scores
                x_scores_perm, y_scores_perm = pls_perm.transform(X_scaled, y_shuffled)
                # Calculate correlation for the CURRENT LV index using permuted scores
                r_perm, _ = pearsonr(x_scores_perm[:, lv_idx], y_scores_perm[:, lv_idx])
                perm_corrs_lv[i] = r_perm if pd.notna(r_perm) else 0.0 # Store 0 if NaN
                n_valid_perms += 1
            except Exception: # Catch potential errors during permuted fit/transform/corr
                perm_corrs_lv[i] = np.nan # Mark as invalid

        perm_loop_end = time.time()
        print(f" done ({perm_loop_end - perm_loop_start:.1f}s).")

        # Calculate p-value for this LV
        valid_perm_corrs_lv = perm_corrs_lv[~np.isnan(perm_corrs_lv)]
        if len(valid_perm_corrs_lv) < n_permutations * 0.8: # Check if too many permutations failed
             print(f"Warning: High number of failed permutations ({n_permutations - len(valid_perm_corrs_lv)}) for LV{lv_idx+1}. P-value may be unreliable.")

        if len(valid_perm_corrs_lv) == 0 or np.isnan(r_actual):
            p_value_lv = 1.0
        else:
             # Two-tailed test: Count how many permuted |r| >= actual |r|
            p_value_lv = (np.sum(np.abs(valid_perm_corrs_lv) >= np.abs(r_actual)) + 1) / (len(valid_perm_corrs_lv) + 1)

        lv_p_values[lv_idx] = p_value_lv
        print(f"  LV{lv_idx+1}: p = {p_value_lv:.4f}")

        # Check significance and decide whether to continue
        if p_value_lv <= alpha:
            significant_lvs_indices.append(lv_idx) # Store 0-based index
        else:
            print(f"  LV{lv_idx+1} is not significant (p > {alpha}). Stopping permutation tests.")
            break # Stop testing further LVs

    perm_end_time = time.time()
    print(f"Permutation testing finished ({perm_end_time - perm_start_time:.1f}s). Found {len(significant_lvs_indices)} significant LVs.")

    # --- 4. Bootstrapping for Loading Stability (Only for Significant LVs) ---
    all_bootstrap_ratios = {} # Store BSR series per significant LV index (0-based)

    if significant_lvs_indices and n_bootstraps > 0:
        print(f"Running {n_bootstraps} bootstraps for {len(significant_lvs_indices)} significant LVs...")
        boot_start_time = time.time()

        # Store bootstrap loadings temporarily, indexed by lv_idx
        x_loadings_boot_all_lvs = {lv_idx: [] for lv_idx in significant_lvs_indices}

        n_boot_success = 0
        for i in range(n_bootstraps):
            # Resample data with replacement
            indices = resample(np.arange(n_samples_pls))
            X_boot, Y_boot = X_scaled[indices], Y_scaled[indices]

            try:
                pls_boot = PLSCanonical(n_components=effective_max_components, scale=False)
                pls_boot.fit(X_boot, Y_boot)

                # Store loadings for EACH significant LV from this bootstrap run
                for lv_idx in significant_lvs_indices:
                     # Ensure the fitted model actually has this many components' loadings
                     if lv_idx < pls_boot.x_loadings_.shape[1]:
                         x_loadings_boot_all_lvs[lv_idx].append(pls_boot.x_loadings_[:, lv_idx])
                     # else: model might have collapsed dimensionality, skip this LV for this bootstrap

                n_boot_success += 1
                if (i + 1) % (n_bootstraps // 5) == 0: # Print progress periodically
                    print(f"  Bootstrap {i+1}/{n_bootstraps} completed.")

            except Exception:
                continue # Skip this bootstrap iteration if it fails

        print(f"  Finished {n_boot_success}/{n_bootstraps} successful bootstrap iterations.")

        # Calculate BSRs for each significant LV
        for lv_idx in significant_lvs_indices:
            loadings_boot_lv = x_loadings_boot_all_lvs[lv_idx]
            if len(loadings_boot_lv) > 1: # Need at least 2 successful bootstraps for std dev
                loadings_boot_lv = np.array(loadings_boot_lv)
                # Use original loadings for the numerator (more stable estimate of effect)
                original_loadings_lv = pls.x_loadings_[:, lv_idx]
                loadings_std_lv = np.std(loadings_boot_lv, axis=0)

                # Calculate BSR = original_loading / bootstrap_std_error
                non_zero_std_mask_lv = loadings_std_lv > 1e-9 # Avoid division by zero/tiny numbers
                bsr_lv = np.full_like(original_loadings_lv, np.nan) # Initialize with NaN
                bsr_lv[non_zero_std_mask_lv] = original_loadings_lv[non_zero_std_mask_lv] / loadings_std_lv[non_zero_std_mask_lv]

                all_bootstrap_ratios[lv_idx] = pd.Series(bsr_lv, index=valid_kinematic_cols)
            else:
                print(f"Warning: Insufficient successful bootstrap samples ({len(loadings_boot_lv)}) to calculate BSR for LV{lv_idx+1}. Setting BSR to NaN.")
                all_bootstrap_ratios[lv_idx] = pd.Series(np.nan, index=valid_kinematic_cols) # Fill with NaN

        boot_end_time = time.time()
        print(f"Bootstrapping finished ({boot_end_time - boot_start_time:.1f}s).")
    elif not significant_lvs_indices:
         print("No significant LVs found, skipping bootstrapping.")
    else: # n_bootstraps <= 0
         print("n_bootstraps set to 0, skipping bootstrapping.")

    # --- 5. Prepare Output ---
    final_results = {
        'task': task_prefix,
        'significant_lvs': [idx + 1 for idx in significant_lvs_indices], # Return 1-based indices
        'lv_results': {},
        'kinematic_variables': valid_kinematic_cols,
        'n_samples_pls': n_samples_pls,
        'max_components_tested': effective_max_components
    }

    # Populate results for each tested LV
    for lv_idx in range(effective_max_components):
        lv_num = lv_idx + 1 # User-facing LV number (1-based)
        is_significant = lv_idx in significant_lvs_indices

        # Store original loadings/scores always, BSR only if significant & computed
        final_results['lv_results'][lv_num] = {
            'significant': is_significant,
            'p_value': lv_p_values[lv_idx],
            'correlation': lv_correlations[lv_idx],
            'x_loadings': pd.Series(pls.x_loadings_[:, lv_idx], index=valid_kinematic_cols),
            'y_loadings': pls.y_loadings_[0, lv_idx], # Y is (1, n_comp), access row 0, col lv_idx
            'bootstrap_ratios': all_bootstrap_ratios.get(lv_idx, None), # Get BSR if computed for this LV
            'x_scores': x_scores_orig[:, lv_idx], # Original scores for this LV
            'y_scores': y_scores_orig[:, lv_idx]  # Original scores for this LV
        }

    print(f"--- PLS Analysis Finished for Task {task_prefix} ---")
    return final_results