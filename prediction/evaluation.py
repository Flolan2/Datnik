# -*- coding: utf-8 -*-
"""
Functions for evaluating model predictions. Handles binary and multi-class cases.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score,
    f1_score, precision_score, recall_score, roc_curve
)
import logging

# Use relative import
try: from . import utils
except ImportError: import utils; print("Warning: Used direct import for 'utils' in evaluation.py")

logger = logging.getLogger('DatnikExperiment') # Get logger instance

def evaluate_predictions(y_true, y_pred, y_pred_proba=None):
    """
    Calculates standard classification metrics for binary or multi-class problems.

    Args:
        y_true: True labels (n_samples,).
        y_pred: Predicted labels (n_samples,).
        y_pred_proba: Predicted probabilities.
                      For binary: (n_samples,) or (n_samples, 2).
                      For multi-class: (n_samples, n_classes). Optional.

    Returns:
        tuple: (metrics_dict, roc_curve_data)
               metrics_dict: Dictionary containing calculated metrics.
                             Includes 'roc_auc' (appropriate type based on problem).
               roc_curve_data: Tuple (fpr, tpr) for BINARY case only, otherwise None.
    """
    metrics = {}
    roc_curve_data = None # Default to None, only populated for binary

    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    unique_true_labels = np.unique(y_true_np)
    n_classes = len(unique_true_labels)

    # --- Basic Checks ---
    if len(y_true_np) == 0 or len(y_pred_np) == 0:
        logger.warning("Empty y_true or y_pred passed to evaluate_predictions.")
        return {}, None
    if len(y_true_np) != len(y_pred_np):
        logger.error(f"Length mismatch: y_true ({len(y_true_np)}) vs y_pred ({len(y_pred_np)}).")
        # Return empty to avoid potentially misleading results
        return {}, None

    # --- Handle Single Class Case ---
    if n_classes < 2:
        logger.warning(f"Only one class ({unique_true_labels}) present in y_true. Many metrics undefined.")
        try: metrics['accuracy'] = accuracy_score(y_true_np, y_pred_np)
        except Exception: metrics['accuracy'] = np.nan
        metrics['roc_auc'] = np.nan; metrics['f1_macro'] = np.nan; metrics['precision_macro'] = np.nan; metrics['recall_macro'] = np.nan
        try: metrics['confusion_matrix'] = confusion_matrix(y_true_np, y_pred_np).tolist()
        except Exception: metrics['confusion_matrix'] = []
        return utils.convert_numpy_types(metrics), None

    # --- Calculate Standard Metrics (Work for Binary & Multi-class) ---
    try: metrics['accuracy'] = accuracy_score(y_true_np, y_pred_np)
    except Exception as e: logger.warning(f"Could not calculate accuracy: {e}"); metrics['accuracy'] = np.nan
    try: metrics['f1_macro'] = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    except Exception as e: logger.warning(f"Could not calculate f1_macro: {e}"); metrics['f1_macro'] = np.nan
    try: metrics['precision_macro'] = precision_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    except Exception as e: logger.warning(f"Could not calculate precision_macro: {e}"); metrics['precision_macro'] = np.nan
    try: metrics['recall_macro'] = recall_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    except Exception as e: logger.warning(f"Could not calculate recall_macro: {e}"); metrics['recall_macro'] = np.nan
    try:
        unique_all_labels = sorted(list(set(y_true_np) | set(y_pred_np)))
        if len(unique_all_labels) > 0:
             metrics['confusion_matrix'] = confusion_matrix(y_true_np, y_pred_np, labels=unique_all_labels).tolist()
        else: metrics['confusion_matrix'] = []
    except Exception as e: logger.warning(f"Could not calculate confusion_matrix: {e}"); metrics['confusion_matrix'] = []


    # --- Calculate ROC AUC (Handles Binary & Multi-class) ---
    metrics['roc_auc'] = np.nan # Default
    roc_curve_data = None      # Default (only set for binary)

    if y_pred_proba is not None:
        y_pred_proba_np = np.asarray(y_pred_proba)

        # Check probability dimensions match expectation
        prob_shape_ok = False
        if n_classes == 2:
            # Binary: Allow (n,) probabilities for positive class OR (n, 2)
            if y_pred_proba_np.ndim == 1 and len(y_pred_proba_np) == len(y_true_np):
                prob_shape_ok = True
            elif y_pred_proba_np.ndim == 2 and y_pred_proba_np.shape == (len(y_true_np), 2):
                 prob_shape_ok = True
                 y_pred_proba_np = y_pred_proba_np[:, 1] # Use prob of positive class
        elif n_classes > 2:
             # Multi-class: Require (n, n_classes)
             if y_pred_proba_np.ndim == 2 and y_pred_proba_np.shape == (len(y_true_np), n_classes):
                  prob_shape_ok = True

        if prob_shape_ok:
             try:
                 if n_classes == 2:
                     metrics['roc_auc'] = roc_auc_score(y_true_np, y_pred_proba_np)
                     # Calculate ROC curve data ONLY for binary case
                     fpr, tpr, _ = roc_curve(y_true_np, y_pred_proba_np)
                     if fpr is not None and tpr is not None and len(fpr) > 1 and len(tpr) > 1:
                          roc_curve_data = (fpr.tolist(), tpr.tolist()) # Store as lists
                     else: roc_curve_data = None
                 else: # Multi-class
                      # Use One-vs-Rest (OvR) strategy, average macro or weighted
                      # Requires probabilities for ALL classes (shape n_samples x n_classes)
                      metrics['roc_auc'] = roc_auc_score(y_true_np, y_pred_proba_np, multi_class='ovr', average='macro')
                      # Weighted average example:
                      # metrics['roc_auc_weighted'] = roc_auc_score(y_true_np, y_pred_proba_np, multi_class='ovr', average='weighted')
                      # ROC curve data is complex for multi-class, skipping for now
                      roc_curve_data = None

             except ValueError as e_auc:
                 # Handles cases like constant probabilities etc.
                 logger.warning(f"Could not calculate ROC AUC score: {e_auc}")
             except Exception as e_auc_other:
                  logger.exception(f"Unexpected error calculating ROC AUC:") # Log full traceback

        else: # Prob shape mismatch
             logger.warning(f"Shape mismatch for y_pred_proba ({y_pred_proba_np.shape}) and n_classes ({n_classes}). Cannot calculate ROC AUC.")

    else: # y_pred_proba is None
         logger.debug("y_pred_proba not provided, skipping ROC AUC calculation.")


    # Convert numpy types for potential JSON serialization later
    metrics_cleaned = utils.convert_numpy_types(metrics)

    return metrics_cleaned, roc_curve_data