# -*- coding: utf-8 -*-
"""
Functions to build scikit-learn pipelines based on configuration.
"""

import logging # Use logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression # For RFE internal estimator
from sklearn.base import clone

# Use relative import if this file is inside the 'prediction' package
try:
    from . import config
except ImportError:
    # Fallback for potentially running this script directly or package issues
    import config
    print("Warning: Used direct import for 'config' in pipeline_builder.py")

logger = logging.getLogger('DatnikExperiment') # Get logger instance

def build_pipeline_from_config(exp_config, random_state):
    """
    Builds a scikit-learn pipeline and parameter distribution/grid for tuning
    based on the provided experiment configuration dictionary.

    Args:
        exp_config (dict): A dictionary defining the experiment setup.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: (pipeline, search_params)
               pipeline: The constructed Pipeline object.
               search_params: Hyperparameter grid (dict) or distribution (dict) for tuning.
                              Returns an empty dict if no matching model found.
    """
    pipeline_steps = []
    current_pipeline_class = Pipeline
    search_params = {} # Initialize

    # --- 1. Imputation ---
    imputer_strategy = exp_config.get('imputer', 'median')
    if imputer_strategy == 'median': pipeline_steps.append(('imputer', SimpleImputer(strategy='median')))
    elif imputer_strategy == 'mean': pipeline_steps.append(('imputer', SimpleImputer(strategy='mean')))
    elif imputer_strategy == 'knn': pipeline_steps.append(('imputer', KNNImputer(n_neighbors=exp_config.get('imputer_knn_neighbors', 5))))
    elif imputer_strategy is not None: logger.warning(f"Unrecognized imputer strategy '{imputer_strategy}'. No imputer added.")

    # --- 2. Resampling ---
    resampler_strategy = exp_config.get('resampler')
    if resampler_strategy == 'smote' and config.IMBLEARN_AVAILABLE and config.SMOTE is not None:
        smote_k = exp_config.get('resampler_smote_k', 4)
        pipeline_steps.append(('resampler', config.SMOTE(random_state=random_state, k_neighbors=smote_k)))
        current_pipeline_class = config.ImbPipeline
        logger.info("   -> Using SMOTE resampling (imblearn pipeline).")
    elif resampler_strategy is not None and resampler_strategy == 'smote': # Only warn if specifically requested but unavailable
        logger.warning(f"Resampler '{resampler_strategy}' requested, but imblearn not available. Skipping resampling.")
    elif resampler_strategy is not None:
         logger.warning(f"Unrecognized resampler strategy '{resampler_strategy}'. Skipping resampling.")

    # --- 3. Scaling ---
    scaler_strategy = exp_config.get('scaler', 'standard')
    if scaler_strategy == 'standard': pipeline_steps.append(('scaler', StandardScaler()))
    elif scaler_strategy == 'robust': pipeline_steps.append(('scaler', RobustScaler()))
    elif scaler_strategy == 'minmax': pipeline_steps.append(('scaler', MinMaxScaler()))
    elif scaler_strategy is not None: logger.warning(f"Unrecognized scaler strategy '{scaler_strategy}'. No scaler added.")

    # --- 4. Feature Selection ---
    selector_strategy = exp_config.get('feature_selector')
    selector_k = exp_config.get('selector_k') # Get k if specified

    if selector_strategy == 'select_kbest':
        if selector_k and selector_k > 0:
            pipeline_steps.append(('feature_selector', SelectKBest(score_func=f_classif, k=selector_k)))
            logger.info(f"   -> Using SelectKBest feature selection (k={selector_k}).")
        else:
            logger.warning("SelectKBest requested but 'selector_k' not specified or is <= 0. Skipping.")
    elif selector_strategy == 'rfe':
        if selector_k and selector_k > 0:
            # Define a simple, relatively fast estimator for RFE ranking
            rfe_estimator = LogisticRegression(solver='liblinear', random_state=random_state, max_iter=500, class_weight='balanced')
            pipeline_steps.append((
                'feature_selector',
                RFE(estimator=rfe_estimator, n_features_to_select=selector_k, importance_getter='auto') # importance_getter='auto' handles coef_ or feature_importances_
            ))
            logger.info(f"   -> Using RFE feature selection (k={selector_k}).")
        else:
             logger.warning("RFE selected but selector_k not specified or is <= 0. Skipping RFE.")
    elif selector_strategy is not None:
        logger.warning(f"Unrecognized feature selector strategy '{selector_strategy}'. Skipping.")


    # --- 5. Classifier ---
    model_name = exp_config.get('model_name')
    search_type = exp_config.get('search_type', 'random') # Default to random search if not specified
    model_info = config.MODEL_PIPELINE_STEPS.get(model_name)

    if model_info:
        classifier = clone(model_info['estimator'])
        try: classifier.set_params(random_state=random_state)
        except ValueError: pass # Ignore if model doesn't accept random_state

        pipeline_steps.append(('classifier', classifier))

        # Select grid or distribution based on search_type flag from config
        if search_type == 'grid':
            search_params = model_info.get('param_grid') # Get grid if defined
            if search_params is None: # Check if None or empty
                logger.warning(f"GridSearch specified for {model_name} but no 'param_grid' found/defined in config. No tuning possible.")
                search_params = {} # Ensure it's an empty dict
            elif not isinstance(search_params, dict) or not search_params:
                 logger.warning(f"param_grid for {model_name} is not a non-empty dictionary. Tuning may fail or be skipped.")
                 if search_params is None: search_params = {} # Ensure dict type
        else: # Default to random search
             search_params = model_info.get('param_dist') # Get distribution
             if search_params is None:
                 logger.warning(f"RandomSearch specified for {model_name} but no 'param_dist' found/defined in config. No tuning possible.")
                 search_params = {}
             elif not isinstance(search_params, dict) or not search_params:
                 logger.warning(f"param_dist for {model_name} is not a non-empty dictionary. Tuning may fail or be skipped.")
                 if search_params is None: search_params = {}

    else:
        logger.error(f"Model name '{model_name}' not found in config.MODEL_PIPELINE_STEPS.")
        return current_pipeline_class([]), {} # Return empty pipeline and params

    # Build the final pipeline
    pipeline = current_pipeline_class(pipeline_steps)
    logger.debug(f"   Pipeline steps constructed: {[step[0] for step in pipeline.steps]}")

    return pipeline, search_params