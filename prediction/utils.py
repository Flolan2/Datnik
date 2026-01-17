
import numpy as np
import pandas as pd
import warnings
import logging 
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFE, SelectKBest # Keep imports for clarity

logger = logging.getLogger('DatnikExperiment') 

def convert_numpy_types(obj):
    """Recursively converts NumPy types to native Python types (NumPy 2.0 compatible)."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.void) or obj is None:
        return None
        
    return obj
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
    else:
        return "unknown_task"


def get_feature_importances(pipeline, feature_names_before_selection):
    """
    Extracts feature importances or coefficients from the final classifier in a pipeline.
    If a feature selector is the penultimate step, it uses the selector's support
    to get the correct feature names. THIS IS THE CORRECTED VERSION.
    """
    try:
        final_classifier = pipeline.steps[-1][1]
        selected_feature_names = list(feature_names_before_selection)

        if len(pipeline.steps) > 1:
            potential_selector = pipeline.steps[-2][1]
            

            if hasattr(potential_selector, 'get_support') and hasattr(potential_selector, 'support_'):
            # --- MODIFICATION END ---
                support_mask = potential_selector.get_support()
                selected_feature_names = [name for i, name in enumerate(feature_names_before_selection) if support_mask[i]]
                logger.debug(f"Selector found and applied. Filtered {len(feature_names_before_selection)} features down to {len(selected_feature_names)}.")
            else:
                logger.debug("No valid feature selector found or selector not fitted. Using original feature names.")
        
        importances_values = None
        if hasattr(final_classifier, 'coef_'):
            importances_values = final_classifier.coef_[0] if final_classifier.coef_.ndim > 1 else final_classifier.coef_
        elif hasattr(final_classifier, 'feature_importances_'):
            importances_values = final_classifier.feature_importances_
        else:
            logger.warning(f"Final estimator {type(final_classifier).__name__} has no coef_ or feature_importances_.")
            return None

        if len(importances_values) == len(selected_feature_names):
            return pd.Series(importances_values, index=selected_feature_names)
        else:

            logger.warning(
                f"CRITICAL MISMATCH in get_feature_importances: "
                f"Length of importance values ({len(importances_values)}) "
                f"does not match length of derived feature names ({len(selected_feature_names)}). "
                f"This should not happen with the corrected logic. Returning None."
            )
            return None

    except Exception as e:
        logger.error(f"Error in get_feature_importances: {e}", exc_info=True)
        return None