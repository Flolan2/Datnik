#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 16:06:54 2025

@author: Lange_L
"""

# -*- coding: utf-8 -*-
"""
Utility functions for the prediction workflow.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning

def convert_numpy_types(obj):
    """Recursively converts NumPy types in a dictionary or list to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        if np.isnan(obj): return None # Represent NaN as None in JSON
        elif np.isinf(obj): return None # Represent Inf as None
        else: return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist() # Convert arrays to lists
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.void)):
        return None
    return obj # Return object itself if not a numpy type

def setup_warnings():
    """Sets up warning filters for cleaner output."""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
    warnings.filterwarnings("ignore", category=FutureWarning)
    # Add other filters as needed

def get_task_from_config_name(config_name_str):
    """
    Helper function to derive a task prefix ('ft', 'hm', or other common ones)
    from a configuration name string. Case-insensitive check.
    Returns 'unknown_task' if no known prefix is found.
    """
    config_name_upper = config_name_str.upper()
    if "_FT_" in config_name_upper:
        return "ft"
    elif "_HM_" in config_name_upper:
        return "hm"
    # Add more task prefixes if you have them, e.g.:
    # elif "_TASKX_" in config_name_upper:
    #     return "taskx"
    else:
        # Consider logging a warning here if you have a logger instance available
        # Or if this function is critical and should always find a task
        # For now, returning a default.
        # print(f"Warning: Could not derive task from config_name: '{config_name_str}'. Returning 'unknown_task'.")
        return "unknown_task"


def get_feature_importances(pipeline, feature_names_before_selection):
    """
    Extracts feature importances or coefficients from the final classifier in a pipeline.
    If a known feature selector (RFE, SelectKBest) is the penultimate step,
    it uses the selector's support to get the correct feature names.
    """
    try:
        final_classifier = pipeline.steps[-1][1] # Assumes classifier is the last step
        
        selected_feature_names = list(feature_names_before_selection) # Default to original names

        if len(pipeline.steps) > 1:
            potential_selector = pipeline.steps[-2][1] # Penultimate step
            if isinstance(potential_selector, (RFE, SelectKBest)) and hasattr(potential_selector, 'get_support'):
                # Ensure the selector has been fitted and has support_ attribute
                if hasattr(potential_selector, 'support_'):
                    support_mask = potential_selector.support_
                    selected_feature_names = [name for i, name in enumerate(feature_names_before_selection) if support_mask[i]]
                else:
                    # This case implies the pipeline wasn't fully fit or selector is problematic
                    # print("Warning: Selector step found but 'support_' attribute missing.")
                    return None 
        
        importances_values = None
        if hasattr(final_classifier, 'coef_'):
            importances_values = final_classifier.coef_[0] if final_classifier.coef_.ndim > 1 else final_classifier.coef_
        elif hasattr(final_classifier, 'feature_importances_'):
            importances_values = final_classifier.feature_importances_
        else:
            # print(f"Note: Final estimator {type(final_classifier).__name__} has no coef_ or feature_importances_.")
            return None

        if len(importances_values) == len(selected_feature_names):
            return pd.Series(importances_values, index=selected_feature_names)
        else:
            # This is the error you were seeing.
            # It means selected_feature_names doesn't match the number of importances from the classifier.
            # This typically happens if feature_names_before_selection was not actually what the selector saw,
            # or if the selector logic is flawed.
            # Given your pipeline structure, feature_names_before_selection *should* be correct.
            print(f"CRITICAL WARNING in get_feature_importances: Length of importance values ({len(importances_values)}) "
                  f"does not match length of derived selected feature names ({len(selected_feature_names)}). "
                  f"Original feature_names_before_selection length: {len(feature_names_before_selection)}. "
                  f"Selected names (first 5): {selected_feature_names[:5]}")
            return None

    except Exception as e:
        # The print statement in the main script's except block will handle this warning.
        # print(f"Error in get_feature_importances: {e}")
        return None # Or re-raise e if you want the main try-except to catch it fully