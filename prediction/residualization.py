
# -*- coding: utf-8 -*-
\"\"\"
Reusable, leak-safe residualization utilities for scikit-learn workflows.

Core ideas
----------
- Residualize *only* on the training data in a CV split.
- Apply the learned betas to transform the validation/test data.
- Keep everything in sklearn-compatible classes so it fits cleanly in Pipelines.

Components
----------
1) ResidualizeFeatures: sklearn Transformer that removes linear effects of given covariate columns
   from the specified feature columns. Fit on train; transform on any split.

2) ResidualizeTargetCV: small helper to residualize y within a CV split. It learns a linear model
   y ~ covariates on the training indices and applies it to train/test y.
\"\"\"

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]

class ResidualizeFeatures(BaseEstimator, TransformerMixin):
    \"\"\"
    Residualize selected feature columns against specified covariate columns.

    Parameters
    ----------
    feature_cols : Sequence[str] or Sequence[int]
        Columns (by name or index) to be residualized.
    covariate_cols : Sequence[str] or Sequence[int]
        Columns (by name or index) used as predictors in the residualization regression.
    copy : bool
        If True, work on a copy to avoid in-place side effects.
    \"\"\"
    def __init__(self,
                 feature_cols: Sequence[Union[str, int]],
                 covariate_cols: Sequence[Union[str, int]],
                 copy: bool = True):
        self.feature_cols = list(feature_cols)
        self.covariate_cols = list(covariate_cols)
        self.copy = copy
        self._is_fit = False
        self._coefs_ = None  # shape: (n_features_to_resid, n_covariates+1) including intercept
        self._feature_cols_idx_ = None
        self._covariate_cols_idx_ = None

    def _locate_columns(self, X: ArrayLike) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        if isinstance(X, pd.DataFrame):
            if all(isinstance(c, int) for c in self.feature_cols):
                feat_idx = np.array(self.feature_cols, dtype=int)
            else:
                feat_idx = np.array([X.columns.get_loc(c) for c in self.feature_cols], dtype=int)

            if all(isinstance(c, int) for c in self.covariate_cols):
                cov_idx = np.array(self.covariate_cols, dtype=int)
            else:
                cov_idx = np.array([X.columns.get_loc(c) for c in self.covariate_cols], dtype=int)

            return feat_idx, cov_idx, X
        else:
            # Assume numpy array input
            feat_idx = np.array(self.feature_cols, dtype=int)
            cov_idx = np.array(self.covariate_cols, dtype=int)
            return feat_idx, cov_idx, None

    def fit(self, X: ArrayLike, y: ArrayLike=None):
        feat_idx, cov_idx, df = self._locate_columns(X)

        X_np = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        Z = X_np[:, cov_idx]  # covariates
        ones = np.ones((Z.shape[0], 1))
        Z1 = np.concatenate([ones, Z], axis=1)  # add intercept

        coefs = []
        for j in feat_idx:
            f = X_np[:, j:j+1]
            beta, *_ = np.linalg.lstsq(Z1, f, rcond=None)  # (n_cov+1, 1)
            coefs.append(beta.ravel())
        self._coefs_ = np.stack(coefs, axis=0)  # (n_feat_to_resid, n_cov+1)
        self._feature_cols_idx_ = feat_idx
        self._covariate_cols_idx_ = cov_idx
        self._is_fit = True
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        if not self._is_fit:
            raise RuntimeError("ResidualizeFeatures must be fit before transform.")
        feat_idx = self._feature_cols_idx_
        cov_idx = self._covariate_cols_idx_

        X_np = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        Z = X_np[:, cov_idx]
        ones = np.ones((Z.shape[0], 1))
        Z1 = np.concatenate([ones, Z], axis=1)  # add intercept

        X_out = X.copy() if (self.copy and isinstance(X, pd.DataFrame)) else (X_np.copy() if self.copy else X_np)
        # Subtract fitted effect for each targeted feature
        for i, j in enumerate(feat_idx):
            pred = Z1 @ self._coefs_[i, :].reshape(-1, 1)
            X_out.iloc[:, j] = X_np[:, j:j+1] - pred if isinstance(X, pd.DataFrame) else (X_np[:, j:j+1] - pred).ravel()
        return X_out


@dataclass
class TargetResidualizationResult:
    y_train_resid: np.ndarray
    y_test_resid: np.ndarray
    beta: np.ndarray  # coefficients incl. intercept used for the transform


def residualize_target_cv(y: ArrayLike,
                          covariates: ArrayLike,
                          train_idx: np.ndarray,
                          test_idx: np.ndarray) -> TargetResidualizationResult:
    \"\"\"
    Residualize y against covariates using parameters learned on training indices,
    then apply the same transform to both train and test.

    Returns
    -------
    TargetResidualizationResult
        Contains residualized y_train and y_test, and the betas used.
    \"\"\"
    y_np = np.asarray(y).reshape(-1, 1)
    Z = np.asarray(covariates)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    Ztr = Z[train_idx, :]
    ytr = y_np[train_idx, :]
    ones_tr = np.ones((Ztr.shape[0], 1))
    Ztr1 = np.concatenate([ones_tr, Ztr], axis=1)

    # Fit on train only
    beta, *_ = np.linalg.lstsq(Ztr1, ytr, rcond=None)  # (n_cov+1, 1)

    # Apply to train
    ytr_pred = Ztr1 @ beta
    ytr_res = ytr - ytr_pred

    # Apply to test
    Zte = Z[test_idx, :]
    ones_te = np.ones((Zte.shape[0], 1))
    Zte1 = np.concatenate([ones_te, Zte], axis=1)
    yte = y_np[test_idx, :]
    yte_pred = Zte1 @ beta
    yte_res = yte - yte_pred

    return TargetResidualizationResult(
        y_train_resid=ytr_res.ravel(),
        y_test_resid=yte_res.ravel(),
        beta=beta.ravel()
    )
