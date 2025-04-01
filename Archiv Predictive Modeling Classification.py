#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
risk_group_threshold_sweep.py

This script evaluates different z‑score thresholds to define risk groups based on 
the 'Contralateral_Striatum_Z' column. For each candidate threshold, a risk-group 
classification pipeline is run (using a RandomForestClassifier with RFE) and its 
cross-validated accuracy is recorded.

Configuration defaults:
    - Task: "ft" (Fingertapping) or "hm" (Hand Movements)
    - Kinematic variables: defined by a base list of variables.
    - Candidate thresholds: list of z‑score cutoffs to try (e.g., –1.5, –2.0, –2.5).
    
Outputs:
    - A CSV file summarizing cross-validated accuracy for each threshold.
    - Optionally, confusion matrix plots (overwritten for each run).
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# -------------------------
# Global Configuration
# -------------------------

# Base kinematic variable names (without task prefix)
BASE_KINEMATIC_COLS = [
    "meanamplitude", "stdamplitude", "meanspeed", "stdspeed", "meanrmsvelocity",
    "stdrmsvelocity", "meanopeningspeed", "stdopeningspeed", "meanclosingspeed",
    "stdclosingspeed", "meancycleduration", "stdcycleduration", "rangecycleduration",
    "rate", "amplitudedecay", "velocitydecay", "ratedecay", "cvamplitude",
    "cvcycleduration", "cvspeed", "cvrmsvelocity", "cvopeningspeed", "cvclosingspeed"
]

# -------------------------
# Data Loading Function
# -------------------------
def load_data(file_path, task="ft", target="Contralateral_Striatum_Z", risk_threshold=-2):
    """
    Load the merged CSV file, select kinematic features for the given task,
    and create a binary risk target: 1 if z-score < risk_threshold, else 0.
    Returns features (X), binary target (y), and patient groups.
    """
    df = pd.read_csv(file_path)
    
    # Build list of kinematic feature names for the task.
    kinematic_cols = [f"{task}_{col}" for col in BASE_KINEMATIC_COLS]
    missing_cols = [col for col in kinematic_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: The following kinematic columns are missing for task {task} and will be ignored: {missing_cols}")
        kinematic_cols = [col for col in kinematic_cols if col in df.columns]
    
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in the data.")
    
    # Convert kinematic and target columns to numeric.
    for col in kinematic_cols + [target]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    if "Patient ID" not in df.columns:
        raise ValueError("Column 'Patient ID' is required for group-based CV.")
    df_clean = df.dropna(subset=kinematic_cols + [target, "Patient ID"])
    
    # Create binary risk target: 1 if z-score is less than risk_threshold.
    df_clean["risk"] = (df_clean[target] < risk_threshold).astype(int)
    
    X = df_clean[kinematic_cols]
    y = df_clean["risk"]
    groups = df_clean["Patient ID"]
    
    return X, y, groups

# -------------------------
# Analysis Pipeline Function
# -------------------------
def run_analysis(file_path, task="ft", target="Contralateral_Striatum_Z", n_features=5, cv="logo", risk_threshold=-2):
    """
    Run the risk-group classification pipeline with a fixed n_features.
    Returns the fitted pipeline and mean cross-validated accuracy.
    """
    print(f"\n--- Running Risk-Group Analysis for task '{task}' with threshold {risk_threshold} ---")
    X, y, groups = load_data(file_path, task, target, risk_threshold)
    print(f"Data loaded: {X.shape[1]} features, {len(y)} samples")
    
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFE(estimator, n_features_to_select=n_features)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', rfe),
        ('classifier', estimator)
    ])
    
    cv_object = LeaveOneGroupOut()
    cv_scores = cross_val_score(pipeline, X, y, cv=cv_object, groups=groups, scoring='accuracy')
    mean_acc = cv_scores.mean()
    print(f"Mean CV Accuracy: {mean_acc:.4f}")
    
    # Optionally, you can fit the pipeline and plot a confusion matrix.
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    cm = confusion_matrix(y, predictions, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "At Risk"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Threshold {risk_threshold})")
    plt.tight_layout()
    output_cm = f"confusion_matrix_{task}_{risk_threshold}.png"
    plt.savefig(output_cm, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to {output_cm}")
    
    return mean_acc, pipeline

# -------------------------
# Main: Threshold Sweep
# -------------------------
if __name__ == '__main__':
    # Define paths.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, "Input")
    merged_file = os.path.join(input_folder, "merged_summary.csv")
    
    task = "ft"  # change to "hm" if needed
    target = "Contralateral_Striatum_Z"
    
    # Define candidate thresholds to try.
    candidate_thresholds = [-1.5, -2.0, -2.5]
    
    # Store performance results.
    results_list = []
    
    for thr in candidate_thresholds:
        acc, _ = run_analysis(merged_file, task=task, target=target, n_features=5, cv="logo", risk_threshold=thr)
        results_list.append({"threshold": thr, "mean_cv_accuracy": acc})
    
    # Convert results to DataFrame and save.
    results_df = pd.DataFrame(results_list)
    print("\nThreshold Sweep Results:")
    print(results_df)
    
    # Save the threshold performance summary.
    output_summary = os.path.join(script_dir, f"threshold_sweep_results_{task}.csv")
    results_df.to_csv(output_summary, index=False)
    print(f"Threshold sweep summary saved to {output_summary}")
