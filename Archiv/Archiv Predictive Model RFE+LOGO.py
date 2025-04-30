import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    cross_val_score, cross_val_predict, LeaveOneGroupOut, 
    GridSearchCV, permutation_test_score
)
from sklearn.metrics import mean_squared_error, r2_score

# Define the list of base kinematic variables (without the task prefix)
BASE_KINEMATIC_COLS = [
    "meanamplitude", "stdamplitude", "meanspeed", "stdspeed", "meanrmsvelocity",
    "stdrmsvelocity", "meanopeningspeed", "stdopeningspeed", "meanclosingspeed",
    "stdclosingspeed", "meancycleduration", "stdcycleduration", "rangecycleduration",
    "rate", "amplitudedecay", "velocitydecay", "ratedecay", "cvamplitude",
    "cvcycleduration", "cvspeed", "cvrmsvelocity", "cvopeningspeed", "cvclosingspeed"
]

def load_data(file_path, task="ft", target="Contralateral_Striatum_Z"):
    """
    Load the merged CSV file and select the kinematic features for the given task along with the target imaging variable.
    Also returns the patient ID groups for use in group-based CV.
    """
    df = pd.read_csv(file_path)
    
    # Build list of kinematic feature names for the given task
    kinematic_cols = [f"{task}_{col}" for col in BASE_KINEMATIC_COLS]
    
    # Check that the necessary columns exist
    missing_cols = [col for col in kinematic_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: The following kinematic columns are missing for task {task} and will be ignored: {missing_cols}")
        kinematic_cols = [col for col in kinematic_cols if col in df.columns]
    
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in the data.")
    
    # Convert columns to numeric
    for col in kinematic_cols + [target]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    # Ensure "Patient ID" is present and drop rows with missing values
    if "Patient ID" not in df.columns:
        raise ValueError("Column 'Patient ID' is required for group-based CV.")
    df_clean = df.dropna(subset=kinematic_cols + [target, "Patient ID"])
    
    X = df_clean[kinematic_cols]
    y = df_clean[target]
    groups = df_clean["Patient ID"]
    
    return X, y, groups

def plot_predicted_vs_actual(X, y, groups, pipeline, output_file="predicted_vs_actual.png"):
    """
    Plots a scatter plot of actual vs. predicted values using LOGO CV predictions,
    and annotates the plot with performance metrics (MSE and R²).
    """
    logo = LeaveOneGroupOut()
    predictions = cross_val_predict(pipeline, X, y, cv=logo, groups=groups)
    
    mse_val = mean_squared_error(y, predictions)
    r2_val = r2_score(y, predictions)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y, predictions, alpha=0.7, edgecolor='k')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel("Actual DatScan Uptake")
    plt.ylabel("Predicted DatScan Uptake")
    plt.title("Predicted vs. Actual Values (LOGO CV)")
    
    annotation_text = f"CV MSE: {mse_val:.2f}\nCV R²: {r2_val:.2f}"
    plt.gca().text(0.05, 0.95, annotation_text, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Predicted vs. Actual plot saved to {output_file}")
    return mse_val, r2_val

def grid_search_n_features(file_path, task="ft", target="Contralateral_Striatum_Z", cv="logo", feature_range=range(3, 16)):
    """
    Perform grid search over a range of n_features to determine the optimal number of features.
    Also saves a scree plot showing CV score vs. number of features.
    """
    X, y, groups = load_data(file_path, task, target)
    estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', RFE(estimator)),
        ('regressor', estimator)
    ])
    
    param_grid = {'feature_selection__n_features_to_select': list(feature_range)}
    
    # Use LeaveOneGroupOut CV
    cv_object = LeaveOneGroupOut() if (isinstance(cv, str) and cv.lower() == "logo") else cv
    
    grid = GridSearchCV(pipeline, param_grid, cv=cv_object, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X, y, groups=groups)
    
    best_n = grid.best_params_['feature_selection__n_features_to_select']
    best_score = grid.best_score_
    print(f"Optimal number of features for task '{task}': {best_n} with CV score: {best_score:.4f}")
    
    grid_results = {params['feature_selection__n_features_to_select']: score 
                    for params, score in zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score'])}
    
    scree_plot_file = f"scree_plot_{task}.png"
    feature_numbers = sorted(grid_results.keys())
    cv_scores = [grid_results[n] for n in feature_numbers]
    plt.figure(figsize=(8,6))
    plt.plot(feature_numbers, cv_scores, marker='o')
    plt.xlabel("Number of Features Selected")
    plt.ylabel("Mean CV Score (neg MSE)")
    plt.title(f"Scree Plot for Task '{task}'")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(scree_plot_file, dpi=300)
    plt.close()
    print(f"Scree plot saved to {scree_plot_file}")
    
    return best_n, best_score, grid_results, scree_plot_file

def run_analysis(file_path, task="ft", target="Contralateral_Striatum_Z", n_features=5, cv="logo"):
    """
    Run the predictive analysis pipeline with a RandomForestRegressor.
    The default is to use a fixed 5-feature model.
    """
    print(f"\n--- Running Predictive Analysis for task '{task}' ---")
    print("Loading data...")
    X, y, groups = load_data(file_path, task, target)
    print(f"Data loaded. Using {X.shape[1]} kinematic features and {len(y)} samples.")
    
    results = {"task": task, "model_type": f"Fixed {n_features}-feature"}
    
    if n_features is None:
        print("Grid search not run automatically. Please call grid_search_n_features() separately if needed.")
        return None, None
    
    estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    rfe = RFE(estimator, n_features_to_select=n_features)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', rfe),
        ('regressor', estimator)
    ])
    
    cv_object = LeaveOneGroupOut() if (isinstance(cv, str) and cv.lower() == "logo") else cv

    print("Performing cross-validation...")
    mse_scores = -cross_val_score(pipeline, X, y, cv=cv_object, groups=groups, scoring='neg_mean_squared_error')
    mean_mse = mse_scores.mean()
    print(f"Mean Cross-Validated MSE for task '{task}': {mean_mse:.4f}")
    
    predictions = cross_val_predict(pipeline, X, y, cv=cv_object, groups=groups)
    r2 = r2_score(y, predictions)
    print(f"Cross-Validated R² for task '{task}': {r2:.4f}")
    
    score, permutation_scores, p_value = permutation_test_score(
        pipeline, X, y, scoring="neg_mean_squared_error", cv=cv_object, n_permutations=100, groups=groups, n_jobs=-1
    )
    print(f"Permutation test p-value for task '{task}': {p_value:.4f}")
    
    print("Fitting pipeline on the full dataset...")
    pipeline.fit(X, y)
    print("Pipeline training complete.")
    
    selected_mask = pipeline.named_steps['feature_selection'].support_
    selected_features = X.columns[selected_mask].tolist()
    print("Selected kinematic features via RFE:")
    print(selected_features)
    
    plot_file = f"predicted_vs_actual_{task}.png"
    mse_val, r2_val = plot_predicted_vs_actual(X, y, groups, pipeline, output_file=plot_file)
    
    results.update({
        "n_features_used": n_features,
        "mean_cv_mse": mean_mse,
        "cv_r2": r2,
        "permutation_test_p": p_value,
        "selected_features": selected_features,
        "predicted_vs_actual_plot": plot_file
    })
    
    return pipeline, results

def run_dual_analysis(file_path, task="ft", target="Contralateral_Striatum_Z", cv="logo", run_grid_search=False):
    """
    Run the analysis with a fixed 5-feature model by default.
    If run_grid_search is True, also run the grid search model.
    """
    print(f"\n=== Analysis for task '{task}' ===")
    
    print("\nRunning fixed 5-feature model (low complexity)...")
    pipeline_fixed, results_fixed = run_analysis(file_path, task, target, n_features=5, cv=cv)
    
    results = {"fixed_model": results_fixed}
    
    if run_grid_search:
        print("\nRunning grid search model for optimal feature selection...")
        best_n, best_score, grid_results, scree_plot_file = grid_search_n_features(file_path, task, target, cv=cv)
        pipeline_grid, results_grid = run_analysis(file_path, task, target, n_features=best_n, cv=cv)
        results_grid["model_type"] = "Grid Search"
        results_grid["grid_search"] = {
            "best_n": best_n,
            "best_score": best_score,
            "grid_results": grid_results,
            "scree_plot": scree_plot_file
        }
        results["grid_search_model"] = results_grid
    
    return results

def save_analysis_results(results, task, output_dir="results"):
    """
    Save the results dictionary to a JSON file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"analysis_results_{task}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Analysis results saved to {output_file}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, 'Input')
    merged_file = os.path.join(input_folder, 'merged_summary.csv')
    
    tasks = ["ft", "hm"]
    all_results = {}
    
    # Set run_grid_search flag to True so that grid search runs
    for task in tasks:
        try:
            results = run_dual_analysis(
                file_path=merged_file, 
                task=task, 
                target="Contralateral_Striatum_Z", 
                cv="logo", 
                run_grid_search=True
            )
            all_results[task] = results
            save_analysis_results(results, task)
        except Exception as e:
            print(f"An error occurred during analysis for task '{task}': {e}")
