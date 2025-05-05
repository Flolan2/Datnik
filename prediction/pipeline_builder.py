#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 16:07:35 2025

@author: Lange_L
"""

# -*- coding: utf-8 -*-
"""
Functions to build scikit-learn pipelines based on configuration.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.base import clone

from . import config # Relative import

def build_pipeline_from_config(exp_config, random_state):
    """
    Builds a scikit-learn pipeline and parameter distribution for tuning
    based on the provided experiment configuration dictionary.

    Args:
        exp_config (dict): A dictionary defining the experiment setup
                           (model_name, scaler, imputer, etc.).
        random_state (int): Random state for reproducibility in relevant components.

    Returns:
        tuple: (pipeline, param_dist)
               pipeline: The constructed scikit-learn Pipeline object.
               param_dist: The hyperparameter distribution for RandomizedSearchCV.
                           Returns an empty dict if no matching model found in config.
    """
    pipeline_steps = []
    current_pipeline_class = Pipeline # Default pipeline class

    # --- 1. Imputation ---
    imputer_strategy = exp_config.get('imputer', 'median') # Default to median
    if imputer_strategy == 'median':
        pipeline_steps.append(('imputer', SimpleImputer(strategy='median')))
    elif imputer_strategy == 'mean':
        pipeline_steps.append(('imputer', SimpleImputer(strategy='mean')))
    elif imputer_strategy == 'knn':
        n_neighbors = exp_config.get('imputer_knn_neighbors', 5) # Allow configuring k
        pipeline_steps.append(('imputer', KNNImputer(n_neighbors=n_neighbors)))
    elif imputer_strategy is not None:
        print(f"Warning: Unrecognized imputer strategy '{imputer_strategy}'. No imputer added.")
    # If imputer_strategy is None, no imputation step is added

    # --- 2. Resampling (requires imblearn) ---
    resampler_strategy = exp_config.get('resampler')
    if resampler_strategy == 'smote':
        if config.IMBLEARN_AVAILABLE and config.SMOTE is not None:
            # Ensure k_neighbors is feasible (needs to be < smallest class size in a fold)
            # A fixed small value is often okay, or could be tuned.
            smote_k = exp_config.get('resampler_smote_k', 4)
            pipeline_steps.append(('resampler', config.SMOTE(random_state=random_state, k_neighbors=smote_k)))
            current_pipeline_class = config.ImbPipeline # Use imblearn pipeline
            print("   -> Using SMOTE resampling (imblearn pipeline).")
        else:
            print(f"Warning: Resampler '{resampler_strategy}' requested, but imblearn not available or SMOTE failed import. Skipping resampling.")
    elif resampler_strategy is not None:
         print(f"Warning: Unrecognized resampler strategy '{resampler_strategy}'. Skipping resampling.")

    # --- 3. Scaling ---
    scaler_strategy = exp_config.get('scaler', 'standard') # Default to standard
    if scaler_strategy == 'standard':
        pipeline_steps.append(('scaler', StandardScaler()))
    elif scaler_strategy == 'robust':
        pipeline_steps.append(('scaler', RobustScaler()))
    elif scaler_strategy == 'minmax':
        pipeline_steps.append(('scaler', MinMaxScaler()))
    elif scaler_strategy is not None:
        print(f"Warning: Unrecognized scaler strategy '{scaler_strategy}'. No scaler added.")
    # If scaler_strategy is None, no scaling step added

    # --- 4. Feature Selection (Optional) ---
    selector_strategy = exp_config.get('feature_selector')
    if selector_strategy == 'select_kbest':
        k = exp_config.get('selector_k', 10) # Default to 10 features if not specified
        if k: # Only add if k is specified and > 0
            pipeline_steps.append(('feature_selector', SelectKBest(score_func=f_classif, k=k)))
            print(f"   -> Using SelectKBest feature selection (k={k}).")
        else:
            print("Warning: 'select_kbest' requested but 'selector_k' not specified or is 0. Skipping.")
    elif selector_strategy == 'rfe':
        # RFE needs an estimator. Often a simple linear model is used.
        # This adds complexity as the estimator itself has parameters.
        # Could use a default Logistic Regression or allow specifying in config.
        # For simplicity, let's use a default LR here.
        print("Warning: RFE feature selection is complex to tune within pipeline. Using default Logistic Regression estimator for RFE.")
        rfe_k = exp_config.get('selector_k', 10)
        from sklearn.linear_model import LogisticRegression # Local import
        rfe_estimator = LogisticRegression(max_iter=500, solver='liblinear') # Basic estimator for RFE step
        pipeline_steps.append(('feature_selector', RFE(estimator=rfe_estimator, n_features_to_select=rfe_k)))
        print(f"   -> Using RFE feature selection (k={rfe_k}).")

    elif selector_strategy is not None:
        print(f"Warning: Unrecognized feature selector strategy '{selector_strategy}'. Skipping.")


    # --- 5. Classifier ---
    model_name = exp_config.get('model_name')
    model_info = config.MODEL_PIPELINE_STEPS.get(model_name)

    if model_info:
        classifier = clone(model_info['estimator']) # Clone to avoid state issues across runs/configs
        param_dist = model_info['param_dist'].copy() # Get associated tuning parameters

        # Set random_state if the estimator supports it
        try:
            classifier.set_params(random_state=random_state)
        except ValueError:
            pass # Estimator doesn't accept random_state

        pipeline_steps.append(('classifier', classifier))
    else:
        print(f"Error: Model name '{model_name}' not found in config.MODEL_PIPELINE_STEPS.")
        # Return an empty pipeline and empty param_dist to signal failure downstream
        return current_pipeline_class([]), {}

    # --- Build Final Pipeline ---
    pipeline = current_pipeline_class(pipeline_steps)
    # print(f"   Pipeline steps: {[step[0] for step in pipeline.steps]}") # Optional: print steps

    return pipeline, param_dist