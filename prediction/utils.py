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

def get_feature_importances(pipeline, feature_names):
    """
    Extracts feature importances or coefficients from the final step of a pipeline.
    Handles Logistic Regression coefficients and Tree-based feature importances.
    Returns a pandas Series indexed by feature names. Returns None if not applicable.
    """
    try:
        # Handle imblearn pipeline if used
        if 'imblearn' in str(type(pipeline)) and hasattr(pipeline, 'steps'):
             final_estimator = pipeline.steps[-1][1]
        elif hasattr(pipeline, 'steps'): # Standard sklearn pipeline
             final_estimator = pipeline.steps[-1][1]
        else: # Assume it's the estimator itself
             final_estimator = pipeline

        if hasattr(final_estimator, 'coef_'): # Linear models
            # Handle multi-class coefficients if necessary, assume binary for now
            if final_estimator.coef_.shape[0] == 1:
                importances = final_estimator.coef_[0]
            else: # Simple approach for multi-class: use magnitude (abs mean) or first class coeffs
                 print("Warning: Multi-class coefficients detected, using coefficients for the first class.")
                 importances = final_estimator.coef_[0] # Or np.abs(final_estimator.coef_).mean(axis=0)
            return pd.Series(importances, index=feature_names)

        elif hasattr(final_estimator, 'feature_importances_'): # Tree-based models
            importances = final_estimator.feature_importances_
            return pd.Series(importances, index=feature_names)

        else:
            # print(f"Note: Final estimator {type(final_estimator).__name__} doesn't have 'coef_' or 'feature_importances_'.")
            return None # Cannot extract importances

    except Exception as e:
        print(f"Warning: Could not extract feature importances: {e}")
        return None