#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_analysis.py

This module builds a predictive pipeline that uses kinematic data to predict a DatScan uptake variable.
It uses Recursive Feature Elimination (RFE) with a RandomForestRegressor estimator. The pipeline includes data scaling,
RFE-based feature selection, and regression. Results are evaluated with cross-validation, additional metrics,
and a grid search to determine the optimal number of features.

Configuration defaults:
    - Task: "ft" (Fingertapping) or "hm" (Hand Movements)
    - Kinematic variables: based on a list of base variables (e.g., "meanamplitude", "stdamplitude", etc.)
    - Target variable: "Contralateral_Striatum_Z" (DatScan uptake via Z-scores)
    
Usage:
    This script can be run directly or imported and called from a main.py script.
    To use LOOCV, call run_analysis with cv="loo".
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict, LeaveOneOut, GridSearchCV, permutation_test_score
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
    
    # Drop rows with missing values
    df_clean = df.dropna(subset=kinematic_cols + [target])
    
    X = df_clean[kinematic_cols]
    y = df_clean[target]
    
    return X, y

def plot_predicted_vs_actual(X, y, pipeline, output_file="predicted_vs_actual.png"):
    """
    Plots a scatter plot of actual vs. predicted values using LOOCV predictions,
    and annotates the plot with LOOCV performance metrics (MSE and R²).
    
    Parameters:
        X (pd.DataFrame): Predictor features.
        y (pd.Series): Actual target values.
        pipeline (Pipeline): The fitted predictive pipeline.
        output_file (str): File path to save the plot.
    """
    loo = LeaveOneOut()
    predictions = cross_val_predict(pipeline, X, y, cv=loo)
    
    # Compute metrics
    mse_val = mean_squared_error(y, predictions)
    r2_val = r2_score(y, predictions)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y, predictions, alpha=0.7, edgecolor='k')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel("Actual DatScan Uptake")
    plt.ylabel("Predicted DatScan Uptake")
    plt.title("Predicted vs. Actual Values (LOOCV)")
    
    # Annotate with performance metrics
    annotation_text = f"LOOCV MSE: {mse_val:.2f}\nLOOCV R²: {r2_val:.2f}"
    plt.gca().text(0.05, 0.95, annotation_text, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Predicted vs. Actual plot saved to {output_file}")


def grid_search_n_features(file_path, task="ft", target="Contralateral_Striatum_Z", cv="loo", feature_range=range(3, 16)):
    """
    Perform grid search over a range of n_features to determine the optimal number of features.
    
    Parameters:
        feature_range: Iterable of integers for the number of features to test.
        
    Returns:
        best_n: The optimal number of features.
        best_score: The best cross-validated score (e.g., negative MSE).
    """
    X, y = load_data(file_path, task, target)
    estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # We will use a pipeline with scaling, RFE, and the estimator.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', RFE(estimator)),
        ('regressor', estimator)
    ])
    
    param_grid = {
        'feature_selection__n_features_to_select': list(feature_range)
    }
    
    if isinstance(cv, str) and cv.lower() == "loo":
        cv_object = LeaveOneOut()
    else:
        cv_object = cv
    
    grid = GridSearchCV(pipeline, param_grid, cv=cv_object, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X, y)
    
    best_n = grid.best_params_['feature_selection__n_features_to_select']
    best_score = grid.best_score_
    print(f"Optimal number of features for task '{task}': {best_n} with CV score: {best_score:.4f}")
    return best_n, best_score

def run_analysis(file_path, task="ft", target="Contralateral_Striatum_Z", n_features=None, cv="loo"):
    """
    Run the predictive analysis pipeline with a RandomForestRegressor.
    If n_features is None, perform a grid search to determine the optimal number of features.
    """
    print(f"\n--- Running Predictive Analysis for task '{task}' ---")
    print("Loading data...")
    X, y = load_data(file_path, task, target)
    print(f"Data loaded. Using {X.shape[1]} kinematic features and {len(y)} samples.")
    
    # Determine optimal number of features if not provided
    if n_features is None:
        print("Performing grid search to determine optimal number of features...")
        n_features, _ = grid_search_n_features(file_path, task, target, cv=cv)
    
    # Define the base estimator
    estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Set up RFE with the chosen number of features
    rfe = RFE(estimator, n_features_to_select=n_features)
    
    # Create the pipeline: scaling, feature selection, and regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', rfe),
        ('regressor', estimator)
    ])
    
    # Determine cross-validation object
    if isinstance(cv, str) and cv.lower() == "loo":
        cv_object = LeaveOneOut()
    else:
        cv_object = cv

    # Evaluate model performance (MSE) using cross-validation
    print("Performing cross-validation...")
    mse_scores = -cross_val_score(pipeline, X, y, cv=cv_object, scoring='neg_mean_squared_error')
    mean_mse = mse_scores.mean()
    print(f"Mean Cross-Validated MSE for task '{task}': {mean_mse:.4f}")
    
    # Also compute R^2 using LOOCV predictions
    predictions = cross_val_predict(pipeline, X, y, cv=cv_object)
    r2 = r2_score(y, predictions)
    print(f"Cross-Validated R^2 for task '{task}': {r2:.4f}")
    
    # Perform a permutation test to assess model significance
    score, permutation_scores, p_value = permutation_test_score(
        pipeline, X, y, scoring="neg_mean_squared_error", cv=cv_object, n_permutations=100, n_jobs=-1
    )
    print(f"Permutation test p-value for task '{task}': {p_value:.4f}")
    
    # Fit the pipeline on the full dataset
    print("Fitting pipeline on the full dataset...")
    pipeline.fit(X, y)
    print("Pipeline training complete.")
    
    # Print the selected features
    selected_mask = pipeline.named_steps['feature_selection'].support_
    selected_features = X.columns[selected_mask]
    print("Selected kinematic features via RFE:")
    print(selected_features.tolist())
    
    # Plot predicted vs. actual values using LOOCV predictions
    plot_file = f"predicted_vs_actual_{task}.png"
    plot_predicted_vs_actual(X, y, pipeline, output_file=plot_file)
    
    return pipeline

if __name__ == '__main__':
    # Determine the path to the merged_summary.csv file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, 'Input')
    merged_file = os.path.join(input_folder, 'merged_summary.csv')
    
    # Define the tasks to loop over
    tasks = ["ft", "hm"]
    
    # Run the analysis for each task
    pipelines = {}
    for task in tasks:
        try:
            # If n_features is not provided, grid search is performed to choose the optimal number
            pipeline = run_analysis(file_path=merged_file, task=task, target="Contralateral_Striatum_Z", n_features=None, cv="loo")
            pipelines[task] = pipeline
        except Exception as e:
            print(f"An error occurred during analysis for task '{task}': {e}")
