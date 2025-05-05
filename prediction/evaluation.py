#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 16:07:53 2025

@author: Lange_L
"""

# -*- coding: utf-8 -*-
"""
Functions for evaluating model predictions.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score,
    f1_score, precision_score, recall_score, roc_curve
)
from . import utils # Relative import for numpy conversion

def evaluate_predictions(y_true, y_pred, y_pred_proba=None):
    """
    Calculates standard classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_pred_proba: Predicted probabilities for the positive class (optional, for ROC AUC).

    Returns:
        tuple: (metrics_dict, roc_curve_data)
               metrics_dict: Dictionary containing calculated metrics.
               roc_curve_data: Tuple (fpr, tpr) or None if not calculable.
    """
    metrics = {}
    roc_curve_data = None # Initialize

    # Ensure y_true and y_pred are numpy arrays for consistent handling
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    unique_true_labels = np.unique(y_true_np)

    # --- Check for edge cases ---
    if len(y_true_np) == 0 or len(y_pred_np) == 0:
        print("Warning: Empty y_true or y_pred passed to evaluate_predictions. Returning empty metrics.")
        return {}, None
    if len(unique_true_labels) < 2:
        print(f"Warning: Only one class ({unique_true_labels}) present in y_true. Several metrics (ROC AUC, F1/Precision/Recall macro) are ill-defined.")
        # Calculate accuracy if possible, others will be NaN or default
        try: metrics['accuracy'] = accuracy_score(y_true_np, y_pred_np)
        except ValueError: metrics['accuracy'] = np.nan
        metrics['roc_auc'] = np.nan
        metrics['f1_macro'] = np.nan
        metrics['precision_macro'] = np.nan
        metrics['recall_macro'] = np.nan
        # Confusion matrix might still work depending on labels parameter, but often expects both classes
        try: metrics['confusion_matrix'] = confusion_matrix(y_true_np, y_pred_np).tolist()
        except ValueError: metrics['confusion_matrix'] = []
        return utils.convert_numpy_types(metrics), None # Convert before returning

    # --- Calculate Metrics (when >1 class in y_true) ---
    # ROC AUC and Curve (requires probabilities and >1 class in y_true)
    metrics['roc_auc'] = np.nan
    if y_pred_proba is not None:
        y_pred_proba_np = np.asarray(y_pred_proba)
        try:
            metrics['roc_auc'] = roc_auc_score(y_true_np, y_pred_proba_np)
            fpr, tpr, _ = roc_curve(y_true_np, y_pred_proba_np)
            # Ensure fpr/tpr are valid before storing
            if fpr is not None and tpr is not None and len(fpr) > 1 and len(tpr) > 1:
                 roc_curve_data = (fpr.tolist(), tpr.tolist()) # Store as lists
            else: roc_curve_data = None
        except ValueError as e:
            # This can happen if probabilities are constant or y_true still has issues
            print(f"Warning: Could not calculate ROC AUC score or curve: {e}")
            roc_curve_data = None
    else:
         roc_curve_data = None # No probabilities provided


    # Other standard metrics
    try: metrics['accuracy'] = accuracy_score(y_true_np, y_pred_np)
    except ValueError: metrics['accuracy'] = np.nan

    # Use zero_division=0 to avoid warnings when a class has no predictions (returns 0)
    try: metrics['f1_macro'] = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    except ValueError: metrics['f1_macro'] = np.nan
    try: metrics['precision_macro'] = precision_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    except ValueError: metrics['precision_macro'] = np.nan
    try: metrics['recall_macro'] = recall_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    except ValueError: metrics['recall_macro'] = np.nan

    # Confusion Matrix
    try:
         # Explicitly define labels if needed, especially if only one class was predicted
         unique_all_labels = sorted(list(set(y_true_np) | set(y_pred_np))) # Get all labels present
         if len(unique_all_labels) > 0:
              metrics['confusion_matrix'] = confusion_matrix(y_true_np, y_pred_np, labels=unique_all_labels).tolist()
         else: # Should not happen if input arrays are not empty
              metrics['confusion_matrix'] = []
    except ValueError: metrics['confusion_matrix'] = []


    # Convert numpy types for potential JSON serialization later
    metrics_cleaned = utils.convert_numpy_types(metrics)

    return metrics_cleaned, roc_curve_data