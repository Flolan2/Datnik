#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predicts baseline contralateral striatum Z-scores (neurodegeneration severity)
based on the magnitude of Levodopa response observed in kinematic variables.

Uses paired OFF/ON data, calculates improvement scores, selects predictor
variables based on BSR values from a previous MED OFF PLS analysis, and employs
correlation analysis and ElasticNet regression.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import re # Import regular expressions for splitting variable names

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PARENT_DIR = os.path.dirname(SCRIPT_DIR) # Main project folder (e.g., Datnik)

# --- Input Data ---
INPUT_FOLDER = os.path.join(SCRIPT_DIR, "Input") # Where merged_summary is
MERGED_CSV_FILE = os.path.join(INPUT_FOLDER, "merged_summary_with_medon.csv")

# --- Output Data ---
OUTPUT_FOLDER = os.path.join(SCRIPT_PARENT_DIR, "Output", "Prediction_ResponseToDatscan")

# --- PLS Results for Feature Selection ---
PLS_RESULTS_DIR = os.path.join(SCRIPT_PARENT_DIR, "Output", "Data")
PLS_RESULTS_FILE = os.path.join(PLS_RESULTS_DIR, "pls_significant_results_all_tasks_combined_sorted.csv")
# Threshold for selecting variables based on PLS Bootstrap Ratio (BSR)
# Common thresholds are 2.0, 2.5, or 3.0. Adjust as needed.
PLS_BSR_THRESHOLD = 2.0

# --- Pairing Parameters ---
MAX_TIME_DIFF_DAYS = 90 # Max days between paired OFF and ON visits

# --- Model Parameters ---
TEST_SIZE = 0.25
RANDOM_STATE = 42
CV_FOLDS = 5

# --- Kinematic Variable Definitions ---
HIGHER_IS_BETTER_BASES = [
    "meanamplitude", "meanspeed", "meanrmsvelocity",
    "meanopeningspeed", "meanclosingspeed", "rate"
]
LOWER_IS_BETTER_BASES = [
    "stdamplitude", "stdspeed", "stdrmsvelocity",
    "stdopeningspeed", "stdclosingspeed",
    "meancycleduration", "stdcycleduration", "rangecycleduration",
    "amplitudedecay", "velocitydecay", "ratedecay",
    "cvamplitude", "cvcycleduration", "cvspeed", "cvrmsvelocity",
    "cvopeningspeed", "cvclosingspeed"
]
TARGET_IMAGING_COL = "Contralateral_Striatum_Z"

# --- Create Output Folder ---
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"Output will be saved to: {OUTPUT_FOLDER}")
print(f"Attempting to load PLS results from: {PLS_RESULTS_FILE}")
print(f"Using PLS BSR threshold for feature selection: >= {PLS_BSR_THRESHOLD}")


# --- Helper Functions (pair_off_on_data, calculate_levodopa_response - unchanged) ---
def pair_off_on_data(df, max_diff_days=90, target_base_name="Contralateral_Striatum_Z"):
    """Pairs OFF and ON visits for the same patient within a time limit."""
    print("\n--- Pairing OFF and ON Data ---")
    if 'Medication Condition' not in df.columns:
        print("Error: 'Medication Condition' column missing. Cannot pair data.")
        return pd.DataFrame()
    if 'Patient ID' not in df.columns:
        print("Error: 'Patient ID' column missing. Cannot pair data.")
        return pd.DataFrame()
    if 'Date of Visit' not in df.columns:
        print("Error: 'Date of Visit' column missing. Cannot pair data.")
        return pd.DataFrame()

    # Ensure Med Condition is clean before filtering
    df['Medication Condition'] = df['Medication Condition'].astype(str).str.strip().str.lower()
    df_off = df[df['Medication Condition'] == 'off'].copy()
    df_on = df[df['Medication Condition'] == 'on'].copy()

    off_ids = set(df_off['Patient ID'])
    on_ids = set(df_on['Patient ID'])
    paired_ids = list(off_ids.intersection(on_ids))
    print(f"Found {len(paired_ids)} patients with data in both OFF and ON states.")

    if not paired_ids:
        print("No patients found with both OFF and ON data. Cannot proceed with pairing.")
        return pd.DataFrame()

    df_off_paired = df_off[df_off['Patient ID'].isin(paired_ids)].copy()
    df_on_paired = df_on[df_on['Patient ID'].isin(paired_ids)].copy()

    # Add suffixes BEFORE merge
    df_off_paired = df_off_paired.rename(columns=lambda c: f"{c}_OFF" if c not in ['Patient ID'] else c)
    df_on_paired = df_on_paired.rename(columns=lambda c: f"{c}_ON" if c not in ['Patient ID'] else c)

    # Convert dates before merge, handle potential errors
    df_off_paired['Date of Visit_OFF'] = pd.to_datetime(df_off_paired.get('Date of Visit_OFF'), format='%d.%m.%Y', errors='coerce')
    df_on_paired['Date of Visit_ON'] = pd.to_datetime(df_on_paired.get('Date of Visit_ON'), format='%d.%m.%Y', errors='coerce')

    # Drop rows where date conversion failed
    initial_rows_off = len(df_off_paired)
    initial_rows_on = len(df_on_paired)
    df_off_paired.dropna(subset=['Date of Visit_OFF'], inplace=True)
    df_on_paired.dropna(subset=['Date of Visit_ON'], inplace=True)
    if len(df_off_paired) < initial_rows_off: print(f"Dropped {initial_rows_off - len(df_off_paired)} OFF rows due to invalid dates.")
    if len(df_on_paired) < initial_rows_on: print(f"Dropped {initial_rows_on - len(df_on_paired)} ON rows due to invalid dates.")

    if df_off_paired.empty or df_on_paired.empty:
        print("One or both OFF/ON dataframes are empty after date cleaning. Cannot merge.")
        return pd.DataFrame()

    # Merge to create all possible pairs per patient
    paired_data = pd.merge(df_off_paired, df_on_paired, on='Patient ID', how='inner')
    print(f"Total potential OFF/ON combinations after merge: {len(paired_data)}")
    if paired_data.empty: return pd.DataFrame()

    # Calculate time difference
    paired_data['Visit_Time_Diff_Days'] = (paired_data['Date of Visit_ON'] - paired_data['Date of Visit_OFF']).dt.days

    # Filter pairs
    paired_data_filtered = paired_data[
        (paired_data['Visit_Time_Diff_Days'] >= 0) &
        (paired_data['Visit_Time_Diff_Days'] <= max_diff_days)
    ].copy()
    print(f"Found {len(paired_data_filtered)} pairs within {max_diff_days} days (ON >= OFF).")
    if paired_data_filtered.empty: return pd.DataFrame()

    # Select the best pair per patient
    paired_data_filtered.sort_values(by=['Patient ID', 'Visit_Time_Diff_Days'], inplace=True)
    final_paired_df = paired_data_filtered.drop_duplicates(subset=['Patient ID'], keep='first')
    print(f"Selected {len(final_paired_df)} unique closest OFF/ON pairs per patient.")

    # Ensure the target imaging column from the OFF state is present
    target_col_off = f"{target_base_name}_OFF"
    if target_col_off not in final_paired_df.columns:
         print(f"\n--- CRITICAL ERROR ---")
         print(f"Target outcome column '{target_col_off}' not found in the paired data.")
         # ... (error message details as before) ...
         return pd.DataFrame()

    # Drop rows where the target imaging value is missing
    initial_paired_rows = len(final_paired_df)
    final_paired_df.dropna(subset=[target_col_off], inplace=True)
    print(f"Removed {initial_paired_rows - len(final_paired_df)} rows with missing target imaging values ('{target_col_off}').")
    print(f"Final paired dataset size: {len(final_paired_df)}")
    if final_paired_df.empty: print("No paired data remains after removing missing target values.")
    return final_paired_df

def calculate_levodopa_response(df, higher_better, lower_better):
    """Calculates Levodopa response scores for kinematic variables."""
    print("\n--- Calculating Levodopa Response Scores ---")
    response_cols_dict = {} # Use dict to build columns efficiently
    base_kinematic_cols = set()

    # Dynamically find base kinematic columns present
    for col in df.columns:
        if col.endswith('_OFF') or col.endswith('_ON'):
             base_name_parts = col.split('_')
             if len(base_name_parts) > 2 and base_name_parts[0] in ['ft', 'hm']:
                 task_prefix = base_name_parts[0]
                 suffix = base_name_parts[-1]
                 base_name = '_'.join(base_name_parts[1:-1])
                 if suffix in ['OFF', 'ON'] and base_name:
                    base_kinematic_cols.add((task_prefix, base_name))

    print(f"Found {len(base_kinematic_cols)} unique base kinematic variables to calculate response for.")
    calculated_count = 0
    skipped_count = 0

    temp_df = df.copy() # Work on a copy to avoid fragmentation warnings

    for task, base_col in base_kinematic_cols:
        off_col = f"{task}_{base_col}_OFF"
        on_col = f"{task}_{base_col}_ON"
        response_col = f"Response_{task}_{base_col}"

        if off_col in temp_df.columns and on_col in temp_df.columns:
            # Ensure numeric
            off_vals = pd.to_numeric(temp_df[off_col], errors='coerce')
            on_vals = pd.to_numeric(temp_df[on_col], errors='coerce')

            response_values = np.nan # Default
            if base_col in higher_better:
                response_values = on_vals - off_vals
            elif base_col in lower_better:
                response_values = off_vals - on_vals
            else:
                skipped_count +=1
                continue

            if not np.isnan(response_values).all(): # Only add if not all NaN
                response_cols_dict[response_col] = response_values
                calculated_count += 1
        # else: print(f"Skipping {response_col} - missing OFF/ON col")

    print(f"Calculated {calculated_count} Levodopa response scores.")
    if skipped_count > 0: print(f"Skipped {skipped_count} variables not defined in higher/lower better lists.")

    # Add calculated columns to the original DataFrame efficiently
    response_df = pd.DataFrame(response_cols_dict, index=df.index)
    df_with_response = pd.concat([df, response_df], axis=1)

    return df_with_response, list(response_cols_dict.keys())


# --- Function to Load PLS Results and Select Variables ---
def get_pls_selected_variables(pls_file_path, bsr_threshold):
    """Loads PLS results CSV and returns list of (task, base_name) tuples meeting BSR threshold."""
    print(f"\n--- Loading PLS results for feature selection (Threshold: |BSR| >= {bsr_threshold}) ---")
    important_bases_from_pls = []
    try:
        try:
            pls_results_df = pd.read_csv(pls_file_path, sep=';', decimal='.')
        except (FileNotFoundError, pd.errors.ParserError):
            print(f"Warning: Failed to load PLS results with ';'. Trying ','. File: {pls_file_path}")
            pls_results_df = pd.read_csv(pls_file_path, sep=',', decimal='.')

        # Ensure required columns exist
        if 'Kinematic_Variable' not in pls_results_df.columns or 'Bootstrap_Ratio' not in pls_results_df.columns:
            print("Error: PLS results file missing 'Kinematic_Variable' or 'Bootstrap_Ratio' column.")
            return None # Indicate failure

        # Filter by BSR threshold
        pls_results_df.dropna(subset=['Bootstrap_Ratio'], inplace=True)
        significant_bsr_df = pls_results_df[abs(pls_results_df['Bootstrap_Ratio']) >= bsr_threshold].copy()

        if significant_bsr_df.empty:
            print(f"Warning: No variables found in PLS results meeting the BSR threshold >= {bsr_threshold}.")
            return [] # Return empty list, not None

        # Extract unique kinematic variable names (e.g., 'ft_meanamplitude')
        pls_kinematic_vars = significant_bsr_df['Kinematic_Variable'].unique()

        # Convert to (task, base_name) tuples
        for var_name in pls_kinematic_vars:
            match = re.match(r"^(ft|hm)_(.+)$", var_name) # Match ft_ or hm_ prefix
            if match:
                task_prefix = match.group(1)
                base_name = match.group(2)
                important_bases_from_pls.append((task_prefix, base_name))
            else:
                print(f"Warning: Could not parse task/base name from PLS variable: '{var_name}'")

        print(f"Selected {len(important_bases_from_pls)} base variables based on PLS BSR threshold.")
        return important_bases_from_pls

    except FileNotFoundError:
        print(f"Error: PLS results file not found at {pls_file_path}")
        print("Feature selection cannot proceed based on PLS. Consider alternative selection or check paths.")
        return None # Indicate failure
    except Exception as e:
        print(f"Error loading or processing PLS results file: {e}")
        return None # Indicate failure

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data
    print(f"Loading data from: {MERGED_CSV_FILE}")
    try:
        df_merged = pd.read_csv(MERGED_CSV_FILE, sep=';', decimal='.')
        print("Read merged CSV with ';' separator.")
    except (FileNotFoundError, pd.errors.ParserError):
        print("Failed to parse with ';'. Trying ',' separator...")
        try:
            df_merged = pd.read_csv(MERGED_CSV_FILE, sep=',', decimal='.')
            print("Read merged CSV with ',' separator.")
        except FileNotFoundError:
            print(f"Error: Input file not found at {MERGED_CSV_FILE}")
            sys.exit(1)
        except Exception as e_load:
            print(f"Error loading or parsing data with ',' separator: {e_load}")
            sys.exit(1)
    except Exception as e_load_other:
        print(f"An unexpected error occurred during data loading: {e_load_other}")
        sys.exit(1)

    print(f"Data loaded successfully. Shape: {df_merged.shape}")
    # Check for Medication Condition column early
    if 'Medication Condition' not in df_merged.columns:
        print("Error: 'Medication Condition' column missing. Cannot separate OFF/ON states.")
        sys.exit(1)


    # 2. Load PLS results for feature selection
    important_bases_from_pls = get_pls_selected_variables(PLS_RESULTS_FILE, PLS_BSR_THRESHOLD)

    if important_bases_from_pls is None:
        print("Exiting due to issues loading or processing PLS results for feature selection.")
        sys.exit(1)
    elif not important_bases_from_pls:
        print("Warning: No features selected based on PLS results. Analysis might not be meaningful.")
        # Decide whether to exit or proceed with potentially zero predictors
        # For now, let it proceed, but it will likely fail later if list is empty.


    # 3. Pair OFF/ON Data
    paired_df = pair_off_on_data(df_merged, MAX_TIME_DIFF_DAYS, TARGET_IMAGING_COL)

    if paired_df.empty:
        print("Exiting script as no valid paired data was found or prepared.")
        sys.exit(0)


    # 4. Calculate Levodopa Response (All available initially)
    paired_df, all_calculated_response_cols = calculate_levodopa_response(
        paired_df, HIGHER_IS_BETTER_BASES, LOWER_IS_BETTER_BASES
    )


    # 5. Filter Predictors Based on PLS Selection
    selected_response_cols = []
    if important_bases_from_pls: # Only filter if PLS selection yielded variables
        for task, base in important_bases_from_pls:
            response_col_name = f"Response_{task}_{base}"
            if response_col_name in all_calculated_response_cols: # Check against actually calculated cols
                 selected_response_cols.append(response_col_name)
            # else: print(f"Note: Var {response_col_name} from PLS selection not found in calculated responses.") # Optional verbose
        response_predictor_cols = selected_response_cols
        print(f"\nFiltered predictor list to {len(response_predictor_cols)} response variables based on PLS selection.")
    else:
        print("\nNo variables selected from PLS. Proceeding with zero predictors (will likely fail).")
        response_predictor_cols = [] # Ensure it's an empty list


    # --- Check if predictors remain ---
    if not response_predictor_cols:
        print("Error: No predictor variables remain after PLS-based selection. Cannot build model.")
        sys.exit(1)


    # 6. Define Outcome Variable
    outcome_col = f"{TARGET_IMAGING_COL}_OFF"
    print(f"Outcome variable (to predict): {outcome_col}")
    print(f"Predictor variables selected (from PLS): {len(response_predictor_cols)}")
    if response_predictor_cols: print(f"Selected predictors: {response_predictor_cols}")


    # 7. Prepare Final Data for Modeling
    analysis_df = paired_df[[outcome_col] + response_predictor_cols].copy()
    initial_rows = len(analysis_df)
    analysis_df.dropna(inplace=True) # Drop rows with NaNs in outcome OR selected predictors
    print(f"\nRemoved {initial_rows - len(analysis_df)} rows with missing values in outcome or selected predictors.")
    print(f"Final dataset size for analysis: {len(analysis_df)}")

    if len(analysis_df) < 10:
        print(f"Error: Insufficient data remaining (N={len(analysis_df)} < 10) after handling missing values for modeling. Exiting.")
        sys.exit(1)
    if len(response_predictor_cols) == 0: # Double check predictors
         print("Error: Zero predictor variables available for modeling.")
         sys.exit(1)

    X = analysis_df[response_predictor_cols]
    y = analysis_df[outcome_col]


    # 8. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")


    # 9. Simple Correlation Analysis (Only for selected predictors)
    print("\n--- Simple Correlation Analysis (Selected Response vs DatScan) ---")
    correlation_results = []
    for predictor in response_predictor_cols: # Iterate only over selected predictors
        valid_data_full = analysis_df[[predictor, outcome_col]].dropna()
        if len(valid_data_full) >= 5:
            try:
                corr, p_val = pearsonr(valid_data_full[predictor], valid_data_full[outcome_col])
                if pd.notna(corr) and pd.notna(p_val):
                     correlation_results.append({
                         'Response_Variable': predictor,
                         'Correlation_with_DatScan_OFF': corr,
                         'P_Value': p_val,
                         'N': len(valid_data_full)
                     })
            except ValueError: continue # Skip constant input etc.

    corr_df = pd.DataFrame(correlation_results)
    if not corr_df.empty:
        # FDR Correction over the *selected* predictors only
        try:
            p_values_for_fdr = corr_df['P_Value'].fillna(1.0)
            if len(p_values_for_fdr) > 0:
                reject, pvals_corrected, _, _ = multipletests(p_values_for_fdr, alpha=0.05, method='fdr_bh')
                corr_df['Q_Value'] = pvals_corrected
                corr_df['Significant_FDR'] = reject
            else:
                 corr_df['Q_Value'], corr_df['Significant_FDR'] = np.nan, False
        except Exception as e:
            print(f"Warning: FDR correction failed for correlations: {e}")
            corr_df['Q_Value'], corr_df['Significant_FDR'] = np.nan, False

        corr_df.sort_values(by='P_Value', inplace=True)
        print("Correlations for PLS-selected variables (sorted by uncorrected p-value):")
        print(corr_df)
        significant_corr = corr_df[corr_df['Significant_FDR'] == True]
        print(f"\nFound {len(significant_corr)} PLS-selected variables significantly correlated with {outcome_col} (after FDR):")
        if not significant_corr.empty: print(significant_corr)
        # Save correlation results
        corr_filename = os.path.join(OUTPUT_FOLDER, "correlation_pls_selected_response_vs_datscan.csv")
        try:
            corr_df.to_csv(corr_filename, index=False, sep=';', decimal='.')
            print(f"Correlation results saved to {corr_filename}")
        except Exception as e_save: print(f"Error saving correlation results: {e_save}")
    else:
        print("No valid correlations could be calculated for the selected predictors.")


    # 10. ElasticNet Regression Modeling (using selected predictors)
    print("\n--- ElasticNet Regression Modeling (using PLS-selected predictors) ---")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('elasticnet', ElasticNet(random_state=RANDOM_STATE, max_iter=10000))
    ])
    param_grid = {
        'elasticnet__alpha': np.logspace(-4, 1, 20),
        'elasticnet__l1_ratio': np.arange(0.1, 1.1, 0.1)
    }
    print(f"Running GridSearchCV (CV={CV_FOLDS}) to find best ElasticNet parameters...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=CV_FOLDS,
                               scoring='neg_mean_squared_error', n_jobs=-1, verbose=0) # Less verbose grid search
    try:
        grid_search.fit(X_train, y_train)
        print(f"Best parameters found: {grid_search.best_params_}")
        best_model = grid_search.best_estimator_

        # Evaluate on TEST set
        y_pred_test = best_model.predict(X_test)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        print("\n--- Model Evaluation on Test Set ---")
        print(f"R-squared (R2): {r2_test:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse_test:.4f}")

        # Extract coefficients
        coefficients = best_model.named_steps['elasticnet'].coef_
        coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})
        non_zero_coef = coef_df[np.abs(coef_df['Coefficient']) > 1e-6].sort_values(by='Coefficient', key=abs, ascending=False)
        print(f"\nNon-zero coefficients from ElasticNet model ({len(non_zero_coef)} features selected):")
        print(non_zero_coef)
        # Save coefficients
        coef_filename = os.path.join(OUTPUT_FOLDER, "elasticnet_pls_selected_coefficients.csv")
        try:
            non_zero_coef.to_csv(coef_filename, index=False, sep=';', decimal='.')
            print(f"Non-zero model coefficients saved to {coef_filename}")
        except Exception as e_save: print(f"Error saving coefficients: {e_save}")

    except ValueError as e_fit:
         print(f"\nError during GridSearchCV fitting: {e_fit}")
         best_model = None
         non_zero_coef = pd.DataFrame()
         r2_test, rmse_test = np.nan, np.nan


    # 11. Visualization (if model fitted)
    print("\n--- Generating Plots ---")
    if best_model is not None and not y_test.empty:
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, y_pred_test, alpha=0.6, edgecolors='k', s=50)
        min_val = min(y_test.min(), y_pred_test.min())
        max_val = max(y_test.max(), y_pred_test.max())
        plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', linewidth=2)
        plt.xlabel(f"Actual {outcome_col}", fontsize=12)
        plt.ylabel(f"Predicted {outcome_col}", fontsize=12)
        plt.title(f"ElasticNet (PLS Selected): Predicted vs Actual {outcome_col} (Test Set)\n$R^2 = {r2_test:.3f}$, RMSE = {rmse_test:.3f}", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        pred_actual_filename = os.path.join(OUTPUT_FOLDER, "elasticnet_pls_selected_predicted_vs_actual.png")
        try:
            plt.savefig(pred_actual_filename, dpi=300, bbox_inches='tight')
            print(f"Predicted vs Actual plot saved to {pred_actual_filename}")
        except Exception as e_save: print(f"Error saving predicted vs actual plot: {e_save}")
        plt.close()

        # Plot Feature Importance
        if not non_zero_coef.empty:
            plt.figure(figsize=(10, max(6, len(non_zero_coef) * 0.35)))
            plot_coef = non_zero_coef.iloc[::-1]
            colors = ['#d62728' if c < 0 else '#2ca02c' for c in plot_coef['Coefficient']]
            plt.barh(plot_coef['Feature'], plot_coef['Coefficient'], color=colors, edgecolor='black', linewidth=0.5)
            plt.xlabel("Coefficient Value (from ElasticNet)", fontsize=12)
            plt.ylabel("Levodopa Response Variable (PLS Selected)", fontsize=12)
            plt.title(f"Feature Importance for Predicting {outcome_col}\n(Non-Zero ElasticNet Coefficients)", fontsize=14)
            plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
            plt.grid(axis='x', linestyle='--', alpha=0.6)
            plt.tight_layout()
            coef_plot_filename = os.path.join(OUTPUT_FOLDER, "elasticnet_pls_selected_coefficients_plot.png")
            try:
                plt.savefig(coef_plot_filename, dpi=300, bbox_inches='tight')
                print(f"Coefficients plot saved to {coef_plot_filename}")
            except Exception as e_save: print(f"Error saving coefficients plot: {e_save}")
            plt.close()
        else:
            print("Skipping coefficient plot as no features were selected by ElasticNet.")
    else:
        print("Skipping plots as model fitting failed or test data was empty.")

    print("\n--- Prediction Script Finished ---")