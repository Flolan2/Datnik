# --- START: NEW evaluation.py ---

# -*- coding: utf-8 -*-
"""
Functions for evaluating model predictions. Handles binary classification.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve
)
import logging

try:
    from . import utils
except ImportError:
    import utils
    print("Warning: Used direct import for 'utils' in evaluation.py.")

logger = logging.getLogger('DatnikExperiment')

def evaluate_predictions(y_true, y_pred, y_pred_proba=None):
    """
    Calculates classification metrics for binary problems.
    Crucially, sensitivity and specificity are derived from the optimal
    threshold on the ROC curve (maximizing Youden's J), not from y_pred.
    """
    metrics = {}
    roc_curve_data = None

    try:
        y_true_np = np.asarray(y_true)
    except Exception as e:
        logger.error(f"Error converting y_true to numpy array: {e}")
        return utils.convert_numpy_types(metrics), roc_curve_data

    if len(np.unique(y_true_np)) < 2:
        logger.warning(f"Only one class present in y_true. Metrics are undefined.")
        metrics['roc_auc'] = np.nan
        metrics['sensitivity'] = np.nan
        metrics['specificity'] = np.nan
        metrics['accuracy'] = np.nan
        return utils.convert_numpy_types(metrics), roc_curve_data

    # --- ROC AUC and Optimal Threshold Calculation ---
    metrics['roc_auc'] = np.nan
    metrics['sensitivity'] = np.nan
    metrics['specificity'] = np.nan
    metrics['accuracy_at_optimal_threshold'] = np.nan

    if y_pred_proba is not None:
        try:
            scores_for_roc = np.asarray(y_pred_proba)
            if scores_for_roc.ndim == 2: scores_for_roc = scores_for_roc[:, 1]

            if len(scores_for_roc) == len(y_true_np):
                metrics['roc_auc'] = roc_auc_score(y_true_np, scores_for_roc)
                fpr, tpr, thresholds = roc_curve(y_true_np, scores_for_roc)
                
                if len(fpr) > 1 and len(tpr) > 1:
                    roc_curve_data = (fpr.tolist(), tpr.tolist())
                    
                    # Calculate Youden's J statistic to find the optimal threshold
                    youden_j = tpr - fpr
                    ix = np.argmax(youden_j)
                    optimal_threshold = thresholds[ix]
                    
                    # Store the sensitivity and specificity from this optimal point
                    metrics['sensitivity'] = tpr[ix]
                    metrics['specificity'] = 1 - fpr[ix]
                    
                    # Calculate accuracy at this new, optimal threshold
                    optimal_preds = (scores_for_roc >= optimal_threshold).astype(int)
                    metrics['accuracy_at_optimal_threshold'] = accuracy_score(y_true_np, optimal_preds)
                else:
                    logger.debug("FPR/TPR too short for ROC curve data.")

        except Exception as e_auc:
            logger.warning(f"Could not calculate ROC-based metrics: {e_auc}")
    else:
        logger.debug("y_pred_proba not provided, cannot calculate ROC-based metrics.")

    # Also calculate accuracy at the default 0.5 threshold for comparison if needed
    try:
        y_pred_np = np.asarray(y_pred)
        metrics['accuracy_at_0.5_threshold'] = accuracy_score(y_true_np, y_pred_np)
    except Exception:
        metrics['accuracy_at_0.5_threshold'] = np.nan

    metrics_cleaned = utils.convert_numpy_types(metrics)
    return metrics_cleaned, roc_curve_data

# --- END: NEW evaluation.py ---