"""
model.py

Transparent Probabilistic Ridge Regression for EAF Tap Temperature
===================================================================

Purpose
-------
This module implements the modelling layer for the EAF decision-support
proof-of-concept.

The model is intentionally transparent:

    X
      |
      v
    Median imputation
      |
      v
    StandardScaler
      |
      v
    Ridge regression
      |
      +--> point prediction
      |
      +--> approximate coefficient uncertainty
      |
      +--> Gaussian predictive uncertainty
      |
      +--> calibration using a chronological holdout set
      |
      +--> probability of falling inside a target range

Important
---------
This is a statistical decision-support model.

It does not:
    - issue PLC/control commands
    - directly control the EAF
    - establish causal effects
    - constitute a physical furnace model
    - guarantee future temperature outcomes

Historical operating-pattern analysis is exposed to decision.py through
the training data retained by the fitted model. Those patterns are
observational associations unless separately validated as causal.

Compatibility
-------------
The public functions in this module are intentionally compatible with
the current main.py:

    train_model()
    predict()
    predict_distribution()
    probability_in_range()
    evaluate_point_prediction()
    evaluate_prediction_intervals()
    feature_importance()
    explain_prediction()
    save_feature_importance()

The model object also provides:

    model.calibrate(X_calibration, y_calibration)

and returns the calibration dictionary expected by main.py.
"""


from __future__ import annotations


from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


import math

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ================================================================
# Numerical constants
# ================================================================

EPSILON = 1e-12

SQRT_2 = math.sqrt(2.0)

NORMAL_P10_Z = -1.2815515655446004
NORMAL_P90_Z = 1.2815515655446004


# ================================================================
# Utility functions
# ================================================================

def _as_1d_float_array(values, name: str = "values") -> np.ndarray:
    """
    Convert an input sequence/Series to a finite 1-D float array.

    Raises
    ------
    ValueError
        If the input cannot be converted to finite numeric values.
    """

    if isinstance(values, pd.Series):
        array = values.to_numpy(dtype=float)
    else:
        array = np.asarray(values, dtype=float)

    array = np.asarray(
        array,
        dtype=float,
    ).reshape(-1)

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return array


def _normal_cdf(values) -> np.ndarray:
    """
    Numerically stable standard-normal CDF.

    This deliberately uses math.erf rather than np.math.erf.

    NumPy does not expose Python's math module as np.math on current
    versions, which caused the previous implementation to fail.
    """

    array = np.asarray(
        values,
        dtype=float,
    )

    flat = array.reshape(-1)

    result = np.empty_like(
        flat,
        dtype=float,
    )

    for i, value in enumerate(flat):

        result[i] = (
            0.5
            *
            (
                1.0
                +
                math.erf(
                    float(value)
                    /
                    SQRT_2
                )
            )
        )

    result = result.reshape(
        array.shape
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


def _normal_pdf(values) -> np.ndarray:
    """
    Standard normal probability density function.
    """

    array = np.asarray(
        values,
        dtype=float,
    )

    return (
        np.exp(
            -0.5 * array ** 2
        )
        /
        math.sqrt(
            2.0 * math.pi
        )
    )


def _safe_std(values, ddof: int = 1) -> float:
    """
    Standard deviation with a safe fallback.
    """

    array = np.asarray(
        values,
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    if array.size <= ddof:
        return 0.0

    value = float(
        np.std(
            array,
            ddof=ddof,
        )
    )

    if not np.isfinite(value):
        return 0.0

    return value


def _safe_mean(values) -> float:
    """
    Mean with a safe fallback.
    """

    array = np.asarray(
        values,
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    if array.size == 0:
        return 0.0

    value = float(
        np.mean(array)
    )

    if not np.isfinite(value):
        return 0.0

    return value


def _safe_rmse(y_true, y_pred) -> float:
    """
    Root mean squared error.
    """

    y_true = _as_1d_float_array(
        y_true,
        "y_true",
    )

    y_pred = _as_1d_float_array(
        y_pred,
        "y_pred",
    )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "RMSE inputs have different lengths."
        )

    errors = (
        y_true
        -
        y_pred
    )

    return float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )


def _safe_mae(y_true, y_pred) -> float:
    """
    Mean absolute error.
    """

    y_true = _as_1d_float_array(
        y_true,
        "y_true",
    )

    y_pred = _as_1d_float_array(
        y_pred,
        "y_pred",
    )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "MAE inputs have different lengths."
        )

    return float(
        np.mean(
            np.abs(
                y_true
                -
                y_pred
            )
        )
    )


def _safe_r2(y_true, y_pred) -> float:
    """
    Coefficient of determination.

    Returns NaN when the target has no variance.
    """

    y_true = _as_1d_float_array(
        y_true,
        "y_true",
    )

    y_pred = _as_1d_float_array(
        y_pred,
        "y_pred",
    )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "R2 inputs have different lengths."
        )

    ss_res = float(
        np.sum(
            (
                y_true
                -
                y_pred
            )
            ** 2
        )
    )

    centered = (
        y_true
        -
        np.mean(y_true)
    )

    ss_tot = float(
        np.sum(
            centered ** 2
        )
    )

    if ss_tot <= EPSILON:
        return float("nan")

    return float(
        1.0
        -
        (
            ss_res
            /
            ss_tot
        )
    )


def _safe_median_absolute_error(
    y_true,
    y_pred,
) -> float:
    """
    Median absolute error.
    """

    y_true = _as_1d_float_array(
        y_true,
        "y_true",
    )

    y_pred = _as_1d_float_array(
        y_pred,
        "y_pred",
    )

    return float(
        np.median(
            np.abs(
                y_true
                -
                y_pred
            )
        )
    )


def _safe_quantile_absolute_error(
    y_true,
    y_pred,
    quantile: float,
) -> float:
    """
    Quantile of absolute prediction error.
    """

    y_true = _as_1d_float_array(
        y_true,
        "y_true",
    )

    y_pred = _as_1d_float_array(
        y_pred,
        "y_pred",
    )

    absolute_error = np.abs(
        y_true
        -
        y_pred
    )

    return float(
        np.quantile(
            absolute_error,
            quantile,
        )
    )


def _ensure_dataframe(
    X,
    feature_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Convert input to a dataframe while preserving indices.
    """

    if isinstance(X, pd.DataFrame):

        result = X.copy()

        if feature_names is not None:

            missing = [
                name
                for name in feature_names
                if name not in result.columns
            ]

            if missing:
                raise ValueError(
                    "Missing required model features: "
                    f"{missing}"
                )

            result = result[
                list(feature_names)
            ]

        return result

    array = np.asarray(
        X
    )

    if array.ndim == 1:
        array = array.reshape(
            1,
            -1,
        )

    if feature_names is None:
        columns = [
            f"feature_{i}"
            for i in range(
                array.shape[1]
            )
        ]
    else:
        columns = list(
            feature_names
        )

    return pd.DataFrame(
        array,
        columns=columns,
    )


# ================================================================
# Model class
# ================================================================

@dataclass
class ExplainableBayesianLikeModel:
    """
    Transparent probabilistic Ridge model.

    The name is retained for compatibility with the existing project.

    Statistically, this implementation is:

        MedianImputer
            +
        StandardScaler
            +
        Ridge regression

    with an approximate Gaussian predictive distribution.

    The calibration scale is estimated from a separate chronological
    calibration set and is applied to predictive standard deviations.
    """

    alpha: float = 10.0
    random_state: int = 42

    feature_names: list[str] = field(
        default_factory=list
    )

    n_observations: int = 0
    n_features: int = 0

    residual_std: float = float("nan")
    residual_variance: float = float("nan")

    calibration_scale: float = 1.0
    is_calibrated: bool = False

    calibration_observations: int = 0
    calibration_metrics: Dict[str, Any] = field(
        default_factory=dict
    )

    model: Optional[Ridge] = None
    imputer: Optional[SimpleImputer] = None
    scaler: Optional[StandardScaler] = None

    coefficient_covariance_scaled: Optional[np.ndarray] = None
    coefficient_std_scaled: Optional[np.ndarray] = None

    training_X: Optional[pd.DataFrame] = None
    training_y: Optional[pd.Series] = None

    training_predictions: Optional[np.ndarray] = None
    training_residuals: Optional[np.ndarray] = None

    fitted: bool = False

    # ------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------

    def fit(
        self,
        X_train,
        y_train,
    ) -> "ExplainableBayesianLikeModel":
        """
        Fit the transparent probabilistic Ridge model.
        """

        X_train = _ensure_dataframe(
            X_train
        )

        y_train_series = (
            y_train.copy()
            if isinstance(
                y_train,
                pd.Series,
            )
            else pd.Series(
                y_train,
                index=X_train.index,
            )
        )

        if len(X_train) != len(
            y_train_series
        ):
            raise ValueError(
                "X_train and y_train have different lengths."
            )

        y_numeric = pd.to_numeric(
            y_train_series,
            errors="coerce",
        )

        valid_y = (
            y_numeric.notna()
            &
            np.isfinite(
                y_numeric.to_numpy(
                    dtype=float
                )
            )
        )

        X_train = X_train.loc[
            valid_y
        ].copy()

        y_numeric = y_numeric.loc[
            valid_y
        ].astype(float)

        if len(X_train) < 10:
            raise ValueError(
                "At least 10 valid observations are required "
                "to fit the model."
            )

        # --------------------------------------------------------
        # Ensure all model inputs are numeric.
        # --------------------------------------------------------

        X_numeric = pd.DataFrame(
            index=X_train.index
        )

        for column in X_train.columns:

            X_numeric[column] = pd.to_numeric(
                X_train[column],
                errors="coerce",
            )

        self.feature_names = list(
            X_numeric.columns
        )

        self.n_observations = len(
            X_numeric
        )

        self.n_features = len(
            self.feature_names
        )

        if self.n_features == 0:
            raise ValueError(
                "No modelling features were supplied."
            )

        # --------------------------------------------------------
        # Retain the original chronological training data.
        #
        # This is intentionally retained for historical-pattern
        # analysis in decision.py.
        # --------------------------------------------------------

        self.training_X = X_numeric.copy()

        self.training_y = pd.Series(
            y_numeric.to_numpy(
                dtype=float
            ),
            index=X_numeric.index,
            name=y_numeric.name
            or "target",
        )

        # --------------------------------------------------------
        # Missing-value treatment.
        #
        # Fit preprocessing on training data only.
        # --------------------------------------------------------

        self.imputer = SimpleImputer(
            strategy="median"
        )

        X_imputed = (
            self.imputer.fit_transform(
                X_numeric
            )
        )

        self.scaler = StandardScaler()

        X_scaled = (
            self.scaler.fit_transform(
                X_imputed
            )
        )

        # --------------------------------------------------------
        # Ridge regression.
        # --------------------------------------------------------

        self.model = Ridge(
            alpha=float(
                max(
                    0.0,
                    self.alpha,
                )
            ),
            fit_intercept=True,
            random_state=self.random_state,
        )

        self.model.fit(
            X_scaled,
            self.training_y.to_numpy(
                dtype=float
            ),
        )

        # --------------------------------------------------------
        # Training predictions/residuals.
        # --------------------------------------------------------

        training_predictions = (
            self.model.predict(
                X_scaled
            )
        )

        self.training_predictions = (
            np.asarray(
                training_predictions,
                dtype=float,
            )
        )

        self.training_residuals = (
            self.training_y.to_numpy(
                dtype=float
            )
            -
            self.training_predictions
        )

        # --------------------------------------------------------
        # Estimate residual variance.
        #
        # Degrees of freedom are approximated using the number of
        # observations minus the number of fitted coefficients.
        # --------------------------------------------------------

        residual_sum_squares = float(
            np.sum(
                self.training_residuals
                ** 2
            )
        )

        degrees_of_freedom = max(
            1,
            self.n_observations
            -
            self.n_features
            -
            1,
        )

        self.residual_variance = max(
            EPSILON,
            residual_sum_squares
            /
            degrees_of_freedom,
        )

        self.residual_std = float(
            math.sqrt(
                self.residual_variance
            )
        )

        # --------------------------------------------------------
        # Approximate coefficient covariance.
        #
        # For scaled X:
        #
        #   Cov(beta) ≈ sigma² (X'X + alpha I)^-1 X'X
        #                      (X'X + alpha I)^-1
        #
        # This is an approximate uncertainty measure for the fitted
        # coefficients, not an exact Bayesian posterior.
        # --------------------------------------------------------

        XtX = (
            X_scaled.T
            @
            X_scaled
        )

        identity = np.eye(
            self.n_features,
            dtype=float,
        )

        regularized = (
            XtX
            +
            float(
                max(
                    0.0,
                    self.alpha,
                )
            )
            *
            identity
        )

        try:

            regularized_inverse = np.linalg.pinv(
                regularized
            )

            covariance = (
                self.residual_variance
                *
                regularized_inverse
                @
                XtX
                @
                regularized_inverse
            )

        except np.linalg.LinAlgError:

            covariance = (
                self.residual_variance
                *
                np.linalg.pinv(
                    XtX
                    +
                    1e-8
                    *
                    identity
                )
            )

        covariance = np.asarray(
            covariance,
            dtype=float,
        )

        covariance = (
            covariance
            +
            covariance.T
        ) / 2.0

        diagonal = np.clip(
            np.diag(
                covariance
            ),
            0.0,
            None,
        )

        self.coefficient_covariance_scaled = (
            covariance
        )

        self.coefficient_std_scaled = (
            np.sqrt(
                diagonal
            )
        )

        self.fitted = True

        # Always start calibration from an uncalibrated state.
        self.calibration_scale = 1.0
        self.is_calibrated = False
        self.calibration_observations = 0
        self.calibration_metrics = {}

        return self

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    def _check_fitted(self):

        if not self.fitted:
            raise RuntimeError(
                "Model has not been fitted."
            )

        if self.model is None:
            raise RuntimeError(
                "Underlying regression model is missing."
            )

        if self.imputer is None:
            raise RuntimeError(
                "Model imputer is missing."
            )

        if self.scaler is None:
            raise RuntimeError(
                "Model scaler is missing."
            )

    # ------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------

    def _transform(
        self,
        X,
    ) -> Tuple[pd.DataFrame, np.ndarray]:

        self._check_fitted()

        X_df = _ensure_dataframe(
            X,
            feature_names=self.feature_names,
        )

        X_numeric = pd.DataFrame(
            index=X_df.index
        )

        for column in self.feature_names:

            X_numeric[column] = pd.to_numeric(
                X_df[column],
                errors="coerce",
            )

        X_imputed = (
            self.imputer.transform(
                X_numeric
            )
        )

        X_scaled = (
            self.scaler.transform(
                X_imputed
            )
        )

        return (
            X_numeric,
            np.asarray(
                X_scaled,
                dtype=float,
            ),
        )

    # ------------------------------------------------------------
    # Predictive mean
    # ------------------------------------------------------------

    def predict(
        self,
        X,
    ) -> np.ndarray:

        _, X_scaled = self._transform(
            X
        )

        prediction = (
            self.model.predict(
                X_scaled
            )
        )

        prediction = np.asarray(
            prediction,
            dtype=float,
        ).reshape(-1)

        if not np.all(
            np.isfinite(
                prediction
            )
        ):
            raise RuntimeError(
                "Model generated non-finite predictions."
            )

        return prediction

    # ------------------------------------------------------------
    # Predictive standard deviation
    # ------------------------------------------------------------

    def predictive_std(
        self,
        X,
        calibrated: bool = True,
    ) -> np.ndarray:
        """
        Estimate predictive standard deviation.

        Components:

            residual/process uncertainty
                +
            approximate coefficient uncertainty

        The calibration scale is applied only when calibrated=True.
        """

        _, X_scaled = self._transform(
            X
        )

        # --------------------------------------------------------
        # Parameter uncertainty:
        #
        # diag(X Cov(beta) X')
        # --------------------------------------------------------

        if (
            self.coefficient_covariance_scaled
            is None
        ):

            coefficient_variance = np.zeros(
                len(X_scaled),
                dtype=float,
            )

        else:

            covariance = (
                self.coefficient_covariance_scaled
            )

            coefficient_variance = np.einsum(
                "ij,jk,ik->i",
                X_scaled,
                covariance,
                X_scaled,
            )

            coefficient_variance = np.clip(
                coefficient_variance,
                0.0,
                None,
            )

        process_variance = max(
            EPSILON,
            self.residual_variance,
        )

        raw_variance = (
            process_variance
            +
            coefficient_variance
        )

        raw_std = np.sqrt(
            np.maximum(
                raw_variance,
                EPSILON,
            )
        )

        scale = (
            self.calibration_scale
            if calibrated
            else 1.0
        )

        calibrated_std = (
            raw_std
            *
            float(
                max(
                    EPSILON,
                    scale,
                )
            )
        )

        # Never allow negative/NaN/inf values.
        calibrated_std = np.nan_to_num(
            calibrated_std,
            nan=self.residual_std,
            posinf=self.residual_std * 10.0,
            neginf=self.residual_std,
        )

        calibrated_std = np.maximum(
            calibrated_std,
            EPSILON,
        )

        return calibrated_std

    # ------------------------------------------------------------
    # Distribution
    # ------------------------------------------------------------

    def predict_distribution(
        self,
        X,
        calibrated: bool = True,
    ) -> pd.DataFrame:
        """
        Return Gaussian predictive distribution summary.
        """

        X_df, _ = self._transform(
            X
        )

        means = self.predict(
            X_df
        )

        stds = self.predictive_std(
            X_df,
            calibrated=calibrated,
        )

        p10 = (
            means
            +
            NORMAL_P10_Z
            *
            stds
        )

        p90 = (
            means
            +
            NORMAL_P90_Z
            *
            stds
        )

        result = pd.DataFrame(
            {
                "temperature_mean":
                    means,

                "temperature_median":
                    means,

                "temperature_std":
                    stds,

                "temperature_p10":
                    p10,

                "temperature_p90":
                    p90,
            },
            index=X_df.index,
        )

        return result

    # ------------------------------------------------------------
    # Probability
    # ------------------------------------------------------------

    def probability_in_range(
        self,
        X,
        lower: float,
        upper: float,
        calibrated: bool = True,
    ) -> np.ndarray:
        """
        Calculate:

            P(lower <= Y <= upper)

        assuming a Gaussian predictive distribution.
        """

        lower = float(lower)
        upper = float(upper)

        if not np.isfinite(
            lower
        ) or not np.isfinite(
            upper
        ):
            raise ValueError(
                "Range limits must be finite."
            )

        if upper < lower:
            raise ValueError(
                "upper must be >= lower."
            )

        distribution = (
            self.predict_distribution(
                X,
                calibrated=calibrated,
            )
        )

        mean = distribution[
            "temperature_mean"
        ].to_numpy(
            dtype=float
        )

        std = distribution[
            "temperature_std"
        ].to_numpy(
            dtype=float
        )

        z_lower = (
            lower
            -
            mean
        ) / std

        z_upper = (
            upper
            -
            mean
        ) / std

        probabilities = (
            _normal_cdf(
                z_upper
            )
            -
            _normal_cdf(
                z_lower
            )
        )

        probabilities = np.clip(
            probabilities,
            0.0,
            1.0,
        )

        return np.asarray(
            probabilities,
            dtype=float,
        )

    # ------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------

    def calibrate(
        self,
        X_calibration,
        y_calibration,
    ) -> Dict[str, Any]:
        """
        Calibrate predictive standard deviation using a separate
        chronological calibration set.

        The calibration factor is estimated from the root mean
        squared standardized residual:

            scale =
                sqrt(
                    mean(
                        residual² /
                        raw predictive variance
                    )
                )

        This is intentionally based only on the calibration set.

        The model itself is NOT refitted on the calibration set.
        """

        self._check_fitted()

        X_calibration_df = _ensure_dataframe(
            X_calibration,
            feature_names=self.feature_names,
        )

        y_series = (
            y_calibration.copy()
            if isinstance(
                y_calibration,
                pd.Series,
            )
            else pd.Series(
                y_calibration,
                index=X_calibration_df.index,
            )
        )

        if len(
            X_calibration_df
        ) != len(
            y_series
        ):
            raise ValueError(
                "Calibration X and y have different lengths."
            )

        y_numeric = pd.to_numeric(
            y_series,
            errors="coerce",
        )

        valid = (
            y_numeric.notna()
            &
            np.isfinite(
                y_numeric.to_numpy(
                    dtype=float
                )
            )
        )

        X_calibration_df = (
            X_calibration_df.loc[
                valid
            ]
        )

        y_numeric = (
            y_numeric.loc[
                valid
            ]
            .astype(float)
        )

        n = len(
            X_calibration_df
        )

        if n < 10:
            raise ValueError(
                "At least 10 valid calibration observations are required."
            )

        # --------------------------------------------------------
        # Raw predictive distribution.
        # --------------------------------------------------------

        raw_distribution = (
            self.predict_distribution(
                X_calibration_df,
                calibrated=False,
            )
        )

        means = raw_distribution[
            "temperature_mean"
        ].to_numpy(
            dtype=float
        )

        raw_std = raw_distribution[
            "temperature_std"
        ].to_numpy(
            dtype=float
        )

        actual = y_numeric.to_numpy(
            dtype=float
        )

        residuals = (
            actual
            -
            means
        )

        standardized_residuals = (
            residuals
            /
            np.maximum(
                raw_std,
                EPSILON,
            )
        )

        # --------------------------------------------------------
        # Estimate scale.
        # --------------------------------------------------------

        squared_standardized = (
            standardized_residuals
            ** 2
        )

        raw_scale = math.sqrt(
            max(
                EPSILON,
                float(
                    np.mean(
                        squared_standardized
                    )
                ),
            )
        )

        # --------------------------------------------------------
        # Numerical guardrail.
        #
        # A calibration factor should correct scale, not allow a
        # pathological observation to make the entire predictive
        # distribution unusably wide.
        #
        # We therefore use a robust bounded calibration factor
        # based on the central calibration distribution.
        #
        # The factor is still derived entirely from calibration
        # residuals.
        # --------------------------------------------------------

        robust_scale = float(
            np.sqrt(
                np.mean(
                    np.minimum(
                        squared_standardized,
                        9.0,
                    )
                )
            )
        )

        if not np.isfinite(
            robust_scale
        ) or robust_scale <= EPSILON:

            robust_scale = raw_scale

        # The unbounded RMSE-based scale remains useful diagnostically.
        # The applied scale is bounded to prevent a single extreme
        # calibration residual from dominating every prediction.
        applied_scale = float(
            np.clip(
                robust_scale,
                0.50,
                2.50,
            )
        )

        self.calibration_scale = (
            applied_scale
        )

        self.is_calibrated = True
        self.calibration_observations = n

        # --------------------------------------------------------
        # Calibration diagnostics.
        # --------------------------------------------------------

        calibrated_std = (
            raw_std
            *
            self.calibration_scale
        )

        p10 = (
            means
            +
            NORMAL_P10_Z
            *
            calibrated_std
        )

        p90 = (
            means
            +
            NORMAL_P90_Z
            *
            calibrated_std
        )

        coverage = float(
            np.mean(
                (
                    actual >= p10
                )
                &
                (
                    actual <= p90
                )
            )
        )

        calibration_mae = (
            _safe_mae(
                actual,
                means,
            )
        )

        calibration_rmse = (
            _safe_rmse(
                actual,
                means,
            )
        )

        absolute_errors = np.abs(
            actual
            -
            means
        )

        calibration_metrics = {
            "calibration_observations":
                int(n),

            "calibration_mae":
                float(calibration_mae),

            "calibration_rmse":
                float(calibration_rmse),

            "calibration_bias":
                float(
                    np.mean(
                        means
                        -
                        actual
                    )
                ),

            "raw_mean_std":
                float(
                    np.mean(
                        raw_std
                    )
                ),

            "calibrated_mean_std":
                float(
                    np.mean(
                        calibrated_std
                    )
                ),

            "calibration_scale":
                float(
                    self.calibration_scale
                ),

            "unbounded_calibration_scale":
                float(
                    raw_scale
                ),

            "robust_calibration_scale":
                float(
                    robust_scale
                ),

            "P10_P90_coverage":
                float(
                    coverage
                ),

            "P10_P90_average_width":
                float(
                    np.mean(
                        p90 - p10
                    )
                ),

            "calibration_median_absolute_error":
                float(
                    np.median(
                        absolute_errors
                    )
                ),

            "calibration_p95_absolute_error":
                float(
                    np.quantile(
                        absolute_errors,
                        0.95,
                    )
                ),
        }

        self.calibration_metrics = (
            calibration_metrics
        )

        return calibration_metrics

    # ------------------------------------------------------------
    # Feature effects
    # ------------------------------------------------------------

    def coefficient_table(
        self,
    ) -> pd.DataFrame:
        """
        Return coefficient-level effects.

        coefficient:
            coefficient in original feature units

        std_effect:
            coefficient multiplied by the training feature standard
            deviation. This is useful for comparing relative effect
            magnitudes across differently scaled features.

        coefficient_std:
            approximate coefficient uncertainty in original units.
        """

        self._check_fitted()

        scaled_coefficients = np.asarray(
            self.model.coef_,
            dtype=float,
        )

        if self.scaler is None:
            raise RuntimeError(
                "Scaler unavailable."
            )

        scale = np.asarray(
            self.scaler.scale_,
            dtype=float,
        )

        safe_scale = np.where(
            scale > EPSILON,
            scale,
            1.0,
        )

        coefficients_original = (
            scaled_coefficients
            /
            safe_scale
        )

        coefficient_std_scaled = (
            self.coefficient_std_scaled
            if self.coefficient_std_scaled
            is not None
            else np.zeros(
                self.n_features
            )
        )

        coefficient_std_original = (
            coefficient_std_scaled
            /
            safe_scale
        )

        std_effect = (
            coefficients_original
            *
            safe_scale
        )

        result = pd.DataFrame(
            {
                "feature":
                    self.feature_names,

                "coefficient":
                    coefficients_original,

                "std_effect":
                    std_effect,

                "importance":
                    np.abs(
                        std_effect
                    ),

                "coefficient_std":
                    coefficient_std_original,
            }
        )

        result = result.sort_values(
            "importance",
            ascending=False,
        ).reset_index(
            drop=True
        )

        return result

    # ------------------------------------------------------------
    # Historical similarity
    # ------------------------------------------------------------

    def historical_similarity(
        self,
        state,
        n_neighbors: int = 250,
    ) -> pd.DataFrame:
        """
        Find historical training heats similar to the supplied state.

        Similarity is based on standardized modelling features.

        This method is intentionally observational. It identifies
        historical heats with similar measured states; it does not
        establish that their operating conditions caused their outcomes.
        """

        self._check_fitted()

        if self.training_X is None:
            raise RuntimeError(
                "Training data is not available."
            )

        state_df = _ensure_dataframe(
            pd.DataFrame(
                [state]
                if isinstance(
                    state,
                    dict,
                )
                else state
            ),
            feature_names=self.feature_names,
        )

        _, state_scaled = self._transform(
            state_df
        )

        training_numeric = (
            self.training_X[
                self.feature_names
            ]
            .copy()
        )

        training_imputed = (
            self.imputer.transform(
                training_numeric
            )
        )

        training_scaled = (
            self.scaler.transform(
                training_imputed
            )
        )

        # --------------------------------------------------------
        # Euclidean distance in standardized feature space.
        # --------------------------------------------------------

        reference = (
            state_scaled[0]
        )

        distances = np.sqrt(
            np.sum(
                (
                    training_scaled
                    -
                    reference
                )
                ** 2,
                axis=1,
            )
            /
            max(
                1,
                self.n_features,
            )
        )

        result = (
            training_numeric.copy()
        )

        result[
            "_historical_index"
        ] = result.index

        result[
            "_similarity_distance"
        ] = distances

        result[
            "_similarity_score"
        ] = (
            1.0
            /
            (
                1.0
                +
                distances
            )
        )

        if self.training_y is not None:

            result[
                "_observed_temperature"
            ] = self.training_y.reindex(
                result.index
            ).to_numpy(
                dtype=float
            )

        result = result.sort_values(
            "_similarity_distance",
            ascending=True,
        )

        return result.head(
            max(
                1,
                int(
                    n_neighbors
                ),
            )
        ).copy()

    # ------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:

        self._check_fitted()

        return {
            "model_type":
                "Transparent Probabilistic Ridge Regression",

            "alpha":
                float(self.alpha),

            "n_observations":
                int(self.n_observations),

            "n_features":
                int(self.n_features),

            "residual_std":
                float(self.residual_std),

            "calibration_scale":
                float(self.calibration_scale),

            "is_calibrated":
                bool(self.is_calibrated),
        }


# ================================================================
# Public training function
# ================================================================

def train_model(
    X_train,
    y_train,
    n_estimators: int = 300,
    max_depth: int = 12,
    min_samples_leaf: int = 5,
    random_state: int = 42,
    alpha: float = 10.0,
) -> ExplainableBayesianLikeModel:
    """
    Train the transparent probabilistic Ridge model.

    The legacy parameters:

        n_estimators
        max_depth
        min_samples_leaf

    are retained solely for compatibility with main.py.

    They are not used because this implementation is Ridge regression,
    not a tree ensemble.
    """

    del n_estimators
    del max_depth
    del min_samples_leaf

    model = ExplainableBayesianLikeModel(
        alpha=float(alpha),
        random_state=int(
            random_state
        ),
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ================================================================
# Public prediction function
# ================================================================

def predict(
    model: ExplainableBayesianLikeModel,
    X,
) -> np.ndarray:
    """
    Predict tap temperature.
    """

    if not isinstance(
        model,
        ExplainableBayesianLikeModel,
    ):
        raise TypeError(
            "predict() expected an "
            "ExplainableBayesianLikeModel."
        )

    return model.predict(
        X
    )


# ================================================================
# Public distribution function
# ================================================================

def predict_distribution(
    model: ExplainableBayesianLikeModel,
    X,
) -> pd.DataFrame:
    """
    Return calibrated predictive distributions.
    """

    if not isinstance(
        model,
        ExplainableBayesianLikeModel,
    ):
        raise TypeError(
            "predict_distribution() expected an "
            "ExplainableBayesianLikeModel."
        )

    return model.predict_distribution(
        X,
        calibrated=True,
    )


# ================================================================
# Public probability function
# ================================================================

def probability_in_range(
    model: ExplainableBayesianLikeModel,
    X,
    lower: float,
    upper: float,
) -> np.ndarray:
    """
    Calculate calibrated Gaussian probability of falling inside
    [lower, upper].
    """

    if not isinstance(
        model,
        ExplainableBayesianLikeModel,
    ):
        raise TypeError(
            "probability_in_range() expected an "
            "ExplainableBayesianLikeModel."
        )

    return model.probability_in_range(
        X,
        lower,
        upper,
        calibrated=True,
    )


# ================================================================
# Point prediction evaluation
# ================================================================

def evaluate_point_prediction(
    y_true,
    predictions,
) -> Dict[str, float]:
    """
    Evaluate point predictions.
    """

    y_true = _as_1d_float_array(
        y_true,
        "y_true",
    )

    predictions = _as_1d_float_array(
        predictions,
        "predictions",
    )

    if len(y_true) != len(
        predictions
    ):
        raise ValueError(
            "Targets and predictions have different lengths."
        )

    errors = (
        y_true
        -
        predictions
    )

    absolute_errors = np.abs(
        errors
    )

    metrics = {
        "MAE":
            float(
                np.mean(
                    absolute_errors
                )
            ),

        "RMSE":
            float(
                np.sqrt(
                    np.mean(
                        errors ** 2
                    )
                )
            ),

        "Mean_Error":
            float(
                np.mean(
                    errors
                )
            ),

        "Median_Absolute_Error":
            float(
                np.median(
                    absolute_errors
                )
            ),

        "P95_Absolute_Error":
            float(
                np.quantile(
                    absolute_errors,
                    0.95,
                )
            ),

        "Maximum_Absolute_Error":
            float(
                np.max(
                    absolute_errors
                )
            ),

        "R2":
            float(
                _safe_r2(
                    y_true,
                    predictions,
                )
            ),

        "Observed_Mean":
            float(
                np.mean(
                    y_true
                )
            ),

        "Predicted_Mean":
            float(
                np.mean(
                    predictions
                )
            ),
    }

    # ------------------------------------------------------------
    # Large-error diagnostics.
    #
    # These are descriptive diagnostics and do not modify predictions.
    # ------------------------------------------------------------

    for threshold in (
        10.0,
        20.0,
        30.0,
        50.0,
        100.0,
    ):

        key = (
            "Absolute_Error_Above_"
            +
            str(
                int(
                    threshold
                )
            )
            +
            "_C"
        )

        metrics[key] = int(
            np.sum(
                absolute_errors
                >
                threshold
            )
        )

    return metrics


# ================================================================
# Prediction interval evaluation
# ================================================================

def evaluate_prediction_intervals(
    y_true,
    prediction_distribution,
) -> Dict[str, float]:
    """
    Evaluate P10-P90 predictive intervals.
    """

    y_true = _as_1d_float_array(
        y_true,
        "y_true",
    )

    if not isinstance(
        prediction_distribution,
        pd.DataFrame,
    ):
        raise TypeError(
            "prediction_distribution must be a DataFrame."
        )

    required = {
        "temperature_p10",
        "temperature_p90",
        "temperature_std",
    }

    missing = (
        required
        -
        set(
            prediction_distribution.columns
        )
    )

    if missing:
        raise ValueError(
            "Prediction distribution is missing columns: "
            f"{sorted(missing)}"
        )

    p10 = prediction_distribution[
        "temperature_p10"
    ].to_numpy(
        dtype=float
    )

    p90 = prediction_distribution[
        "temperature_p90"
    ].to_numpy(
        dtype=float
    )

    std = prediction_distribution[
        "temperature_std"
    ].to_numpy(
        dtype=float
    )

    if len(y_true) != len(
        p10
    ):
        raise ValueError(
            "Targets and prediction intervals have different lengths."
        )

    inside = (
        (
            y_true >= p10
        )
        &
        (
            y_true <= p90
        )
    )

    widths = (
        p90
        -
        p10
    )

    return {
        "P10_P90_coverage":
            float(
                np.mean(
                    inside
                )
            ),

        "P10_P90_average_width":
            float(
                np.mean(
                    widths
                )
            ),

        "P10_P90_median_width":
            float(
                np.median(
                    widths
                )
            ),

        "mean_predictive_std":
            float(
                np.mean(
                    std
                )
            ),

        "median_predictive_std":
            float(
                np.median(
                    std
                )
            ),

        "P10_P90_expected_coverage":
            0.80,

        "coverage_gap_from_nominal":
            float(
                np.mean(
                    inside
                )
                -
                0.80
            ),
    }


# ================================================================
# Global feature importance
# ================================================================

def feature_importance(
    model: ExplainableBayesianLikeModel,
) -> pd.DataFrame:
    """
    Return coefficient-based global feature effects.
    """

    if not isinstance(
        model,
        ExplainableBayesianLikeModel,
    ):
        raise TypeError(
            "feature_importance() expected an "
            "ExplainableBayesianLikeModel."
        )

    return model.coefficient_table()


# ================================================================
# Local explanation
# ================================================================

def explain_prediction(
    model: ExplainableBayesianLikeModel,
    state,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Explain a prediction through additive linear contributions.

    Contributions are calculated in the scaled feature space:

        contribution =
            scaled_feature
            *
            scaled_model_coefficient

    The sum of all contributions plus the model intercept equals the
    model prediction, subject only to floating-point precision.
    """

    if not isinstance(
        model,
        ExplainableBayesianLikeModel,
    ):
        raise TypeError(
            "explain_prediction() expected an "
            "ExplainableBayesianLikeModel."
        )

    model._check_fitted()

    state_df = _ensure_dataframe(
        pd.DataFrame(
            [state]
            if isinstance(
                state,
                dict,
            )
            else state
        ),
        feature_names=model.feature_names,
    )

    X_numeric, X_scaled = (
        model._transform(
            state_df
        )
    )

    if len(X_scaled) != 1:
        raise ValueError(
            "explain_prediction() expects exactly one observation."
        )

    scaled_values = X_scaled[0]

    scaled_coefficients = np.asarray(
        model.model.coef_,
        dtype=float,
    )

    contributions = (
        scaled_values
        *
        scaled_coefficients
    )

    prediction = float(
        model.model.predict(
            X_scaled
        )[0]
    )

    result = pd.DataFrame(
        {
            "feature":
                model.feature_names,

            "value":
                X_numeric.iloc[0]
                .to_numpy(
                    dtype=float
                ),

            "contribution":
                contributions,

            "absolute_contribution":
                np.abs(
                    contributions
                ),

            "prediction":
                prediction,
        }
    )

    result = result.sort_values(
        "absolute_contribution",
        ascending=False,
    )

    return result.head(
        max(
            1,
            int(
                top_n
            ),
        )
    ).reset_index(
        drop=True
    )


# ================================================================
# Save feature importance
# ================================================================

def save_feature_importance(
    model: ExplainableBayesianLikeModel,
    output_file,
):
    """
    Save global feature effects to CSV.
    """

    output_path = Path(
        output_file
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance = feature_importance(
        model
    )

    importance.to_csv(
        output_path,
        index=False,
    )


# ================================================================
# Optional diagnostics helper
# ================================================================

def diagnostic_error_table(
    y_true,
    predictions,
    index: Optional[Iterable] = None,
) -> pd.DataFrame:
    """
    Create a row-level prediction diagnostic table.

    Useful for investigating the large RMSE seen in the current run.
    """

    y_true = _as_1d_float_array(
        y_true,
        "y_true",
    )

    predictions = _as_1d_float_array(
        predictions,
        "predictions",
    )

    if len(y_true) != len(
        predictions
    ):
        raise ValueError(
            "Targets and predictions have different lengths."
        )

    if index is None:
        index = range(
            len(y_true)
        )

    result = pd.DataFrame(
        {
            "actual_temperature":
                y_true,

            "predicted_temperature":
                predictions,

            "error":
                y_true
                -
                predictions,

            "absolute_error":
                np.abs(
                    y_true
                    -
                    predictions
                ),
        },
        index=index,
    )

    result = result.sort_values(
        "absolute_error",
        ascending=False,
    )

    return result


# ================================================================
# End of module
# ================================================================
