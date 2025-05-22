# -*- coding: utf-8 -*-
"""
Functions for evaluating model predictions. Handles binary classification.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score,
    f1_score, precision_score, recall_score, roc_curve
)
import logging

# Attempt relative import for 'utils' first, then direct as fallback
# This structure assumes 'utils.py' is in the same 'prediction' package directory.
try:
    from . import utils
except ImportError:
    # This fallback might be hit if running evaluation.py directly for testing,
    # or if the package structure isn't fully recognized.
    # For the main script 'datnik_prediction_run_experiments.py',
    # the relative import '.utils' should work when 'prediction' is treated as a package.
    import utils # If run standalone or if .utils fails
    print("Warning: Used direct import for 'utils' in evaluation.py. Ensure 'prediction' is a package.")


logger = logging.getLogger('DatnikExperiment') # Get logger instance from the main script

def evaluate_predictions(y_true, y_pred, y_pred_proba=None):
    """
    Calculates standard classification metrics for binary problems.

    Args:
        y_true (array-like): True labels (n_samples,).
        y_pred (array-like): Predicted labels (n_samples,).
        y_pred_proba (array-like, optional): Predicted probabilities for the positive class
                                            or for all classes.
                                            For binary: (n_samples,) [positive class] or (n_samples, 2).
                                            Defaults to None.

    Returns:
        tuple: (metrics_dict, roc_curve_data)
               metrics_dict (dict): Dictionary containing calculated metrics.
                                    Includes 'roc_auc'.
               roc_curve_data (tuple or None): Tuple (fpr_list, tpr_list) for ROC curve plotting
                                               in the BINARY case if probabilities are valid.
                                               None otherwise or if an error occurs.
    """
    metrics = {}
    roc_curve_data = None # Default, only populated for binary with valid probs

    # Ensure inputs are numpy arrays for consistent processing
    try:
        y_true_np = np.asarray(y_true)
        y_pred_np = np.asarray(y_pred)
    except Exception as e:
        logger.error(f"Error converting y_true/y_pred to numpy arrays: {e}")
        return utils.convert_numpy_types(metrics), roc_curve_data # Return empty if conversion fails

    # --- Basic Checks ---
    if len(y_true_np) == 0 or len(y_pred_np) == 0:
        logger.warning("Empty y_true or y_pred passed to evaluate_predictions.")
        return utils.convert_numpy_types(metrics), roc_curve_data
    if len(y_true_np) != len(y_pred_np):
        logger.error(f"Length mismatch: y_true ({len(y_true_np)}) vs y_pred ({len(y_pred_np)}). Metrics will be unreliable.")
        return utils.convert_numpy_types(metrics), roc_curve_data # Return empty to avoid misleading results

    unique_true_labels = np.unique(y_true_np)
    n_classes = len(unique_true_labels)

    # --- Handle Single Class in True Labels ---
    # This is a critical check, especially for ROC AUC
    if n_classes < 2:
        logger.warning(f"Only one class ({unique_true_labels}) present in y_true. ROC AUC and other metrics are undefined or misleading.")
        try:
            metrics['accuracy'] = accuracy_score(y_true_np, y_pred_np)
        except Exception:
            metrics['accuracy'] = np.nan
        metrics['roc_auc'] = np.nan
        metrics['f1_macro'] = np.nan
        metrics['precision_macro'] = np.nan
        metrics['recall_macro'] = np.nan
        try:
            # Ensure confusion matrix labels are consistent if possible
            # For a single class, CM might be simple e.g. [[N]] or error if y_pred has other labels
            all_labels_present = sorted(list(set(y_true_np).union(set(y_pred_np))))
            if not all_labels_present: all_labels_present = [0] # fallback if all empty
            metrics['confusion_matrix'] = confusion_matrix(y_true_np, y_pred_np, labels=all_labels_present).tolist()
        except Exception:
            metrics['confusion_matrix'] = [[len(y_true_np)]] if n_classes == 1 else [] # Simplistic fallback
        return utils.convert_numpy_types(metrics), roc_curve_data

    # --- Calculate Standard Metrics (These generally work for binary) ---
    try:
        metrics['accuracy'] = accuracy_score(y_true_np, y_pred_np)
    except Exception as e:
        logger.warning(f"Could not calculate accuracy: {e}"); metrics['accuracy'] = np.nan

    # For macro-averaged metrics, ensure labels are specified if y_pred might not contain all true labels
    # Using unique labels from y_true for safety, though scikit-learn often handles this.
    # For binary, usually labels=[0,1] is implicit if both are present.
    present_labels = sorted(np.unique(np.concatenate((y_true_np, y_pred_np))).tolist())
    if len(present_labels) < 2 : # If predictions are all one class, and true is different
        logger.warning("Less than 2 classes in combined y_true/y_pred. Macro metrics might be skewed.")
        # Fallback for labels if only one class is ever predicted/true after all
        if not present_labels and len(unique_true_labels) > 0: present_labels = unique_true_labels.tolist()
        elif not present_labels: present_labels = [0,1] # Default if totally empty

    try:
        metrics['f1_macro'] = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0, labels=present_labels if len(present_labels)>=2 else None)
    except Exception as e:
        logger.warning(f"Could not calculate f1_macro: {e}"); metrics['f1_macro'] = np.nan
    try:
        metrics['precision_macro'] = precision_score(y_true_np, y_pred_np, average='macro', zero_division=0, labels=present_labels if len(present_labels)>=2 else None)
    except Exception as e:
        logger.warning(f"Could not calculate precision_macro: {e}"); metrics['precision_macro'] = np.nan
    try:
        metrics['recall_macro'] = recall_score(y_true_np, y_pred_np, average='macro', zero_division=0, labels=present_labels if len(present_labels)>=2 else None)
    except Exception as e:
        logger.warning(f"Could not calculate recall_macro: {e}"); metrics['recall_macro'] = np.nan
    try:
        metrics['confusion_matrix'] = confusion_matrix(y_true_np, y_pred_np, labels=present_labels).tolist()
    except Exception as e:
        logger.warning(f"Could not calculate confusion_matrix: {e}"); metrics['confusion_matrix'] = []


    # --- Calculate ROC AUC and ROC Curve Data (Binary Specific) ---
    metrics['roc_auc'] = np.nan # Default

    if y_pred_proba is not None:
        try:
            y_pred_proba_np = np.asarray(y_pred_proba)

            # Prepare probabilities for positive class (class 1)
            if y_pred_proba_np.ndim == 2 and y_pred_proba_np.shape[1] == 2:
                # Common case: probabilities for [class_0, class_1]
                scores_for_roc = y_pred_proba_np[:, 1]
            elif y_pred_proba_np.ndim == 1:
                # Assumed to be probabilities for the positive class already
                scores_for_roc = y_pred_proba_np
            else:
                logger.warning(f"y_pred_proba has unexpected shape {y_pred_proba_np.shape}. Expected (n_samples,) or (n_samples, 2) for binary. Skipping ROC AUC.")
                scores_for_roc = None

            if scores_for_roc is not None and len(scores_for_roc) == len(y_true_np):
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true_np, scores_for_roc)
                    fpr, tpr, thresholds = roc_curve(y_true_np, scores_for_roc)
                    # Ensure fpr and tpr are substantial enough for plotting
                    if fpr is not None and tpr is not None and len(fpr) > 1 and len(tpr) > 1:
                        roc_curve_data = (fpr.tolist(), tpr.tolist())
                    else:
                        logger.debug("FPR/TPR too short for ROC curve data.")
                        roc_curve_data = None
                except ValueError as ve_auc: # Handles "Only one class present in y_true" if it slips through earlier checks
                    logger.warning(f"ValueError calculating ROC AUC or curve: {ve_auc}. This can happen if y_true within this specific evaluation has only one class.")
                    metrics['roc_auc'] = np.nan
                    roc_curve_data = None
                except Exception as e_auc_other: # Catch any other unexpected error
                    logger.exception(f"Unexpected error calculating ROC AUC or curve:")
                    metrics['roc_auc'] = np.nan
                    roc_curve_data = None
            else:
                if scores_for_roc is None: # Already logged if shape was wrong
                    pass
                else: # Length mismatch
                    logger.warning(f"Length mismatch between scores_for_roc ({len(scores_for_roc)}) and y_true_np ({len(y_true_np)}). Skipping ROC AUC.")

        except Exception as e_proba_proc: # Error processing probabilities
            logger.error(f"Error processing y_pred_proba: {e_proba_proc}")
            metrics['roc_auc'] = np.nan
            roc_curve_data = None
    else:
         logger.debug("y_pred_proba not provided, skipping ROC AUC and curve calculation.")

    # Convert any remaining numpy types (like np.float64) to native Python types
    metrics_cleaned = utils.convert_numpy_types(metrics)

    return metrics_cleaned, roc_curve_data