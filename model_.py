"""
model.py

Nonlinear EAF Tap-Temperature Regression Model
===============================================

This module provides a transparent nonlinear regression model for
EAF tap-temperature prediction.

Model
-----
ExtraTreesRegressor is used as the nonlinear predictive model.

The model can represent:

    - nonlinear relationships
    - feature interactions
    - thresholds
    - saturation effects
    - interactions between charge, energy, oxygen, carbon, time, etc.

Predictive uncertainty
----------------------
The ExtraTrees ensemble provides a distribution of tree-level
predictions for each observation.

A residual uncertainty component is also estimated from the training
data.

A separate chronological calibration set is used to calibrate the
overall predictive standard deviation.

Probability outputs
-------------------
probability_in_range() calculates the probability that the predicted
temperature lies between the requested lower and upper limits under
the calibrated Gaussian approximation.

Important
---------
This is a decision-support POC.

It does NOT:

    - directly control the EAF
    - issue PLC/control commands
    - establish causal effects
    - provide a physically complete furnace model
    - guarantee future probabilities

Feature importance and local explanations describe model behaviour.
They should not be interpreted as causal effects.
"""


from __future__ import annotations


from pathlib import Path
from typing import Dict, Optional


import math

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.preprocessing import StandardScaler


# ================================================================
# Constants
# ================================================================

EPSILON = 1e-12

DEFAULT_N_ESTIMATORS = 300

DEFAULT_MAX_DEPTH = 12

DEFAULT_MIN_SAMPLES_LEAF = 5

DEFAULT_RANDOM_STATE = 42

DEFAULT_ALPHA = 10.0


# ================================================================
# Utility functions
# ================================================================

def _as_numpy_1d(values) -> np.ndarray:
    """
    Convert values to a finite-friendly one-dimensional float array.
    """

    return np.asarray(
        values,
        dtype=float,
    ).reshape(-1)


def _safe_std(values) -> float:
    """
    Sample standard deviation with safe handling of small arrays.
    """

    values = _as_numpy_1d(values)

    values = values[
        np.isfinite(values)
    ]

    if len(values) < 2:
        return 0.0

    return float(
        np.std(
            values,
            ddof=1,
        )
    )


def _normal_cdf(values) -> np.ndarray:
    """
    Standard-normal CDF without requiring scipy.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    return 0.5 * (
        1.0
        +
        np.vectorize(
            math.erf
        )(
            values
            /
            math.sqrt(2.0)
        )
    )


def _normal_pdf(values) -> np.ndarray:
    """
    Standard-normal PDF.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    return (
        np.exp(
            -0.5
            *
            values ** 2
        )
        /
        math.sqrt(
            2.0 * math.pi
        )
    )


def _ensure_dataframe(
    X,
    feature_names=None,
) -> pd.DataFrame:
    """
    Ensure prediction input is a DataFrame.
    """

    if isinstance(
        X,
        pd.DataFrame,
    ):

        result = X.copy()

    elif isinstance(
        X,
        pd.Series,
    ):

        result = pd.DataFrame(
            [X.to_dict()]
        )

    else:

        result = pd.DataFrame(
            X
        )

    if feature_names is not None:

        for feature in feature_names:

            if feature not in result.columns:
                result[feature] = np.nan

        result = result[
            list(feature_names)
        ]

    return result


# ================================================================
# Nonlinear model class
# ================================================================

class EAFNonlinearModel:
    """
    Wrapper around ExtraTreesRegressor.

    The wrapper preserves the interface required by main.py and
    decision.py.
    """

    def __init__(
        self,
        estimator,
        imputer,
        scaler,
        feature_names,
        training_X,
        training_y,
        residual_std,
    ):

        self.estimator = estimator

        self.imputer = imputer

        self.scaler = scaler

        self.feature_names = list(
            feature_names
        )

        self.training_X = training_X.copy()

        self.training_y = (
            pd.Series(
                training_y
            ).copy()
        )

        self.n_observations = int(
            len(training_X)
        )

        self.n_features = int(
            len(feature_names)
        )

        self.residual_std = float(
            residual_std
        )

        # --------------------------------------------------------
        # Raw ensemble uncertainty.
        # --------------------------------------------------------

        self.raw_ensemble_std = (
            self._estimate_training_ensemble_std()
        )

        # --------------------------------------------------------
        # Calibration state.
        # --------------------------------------------------------

        self.calibration_scale = 1.0

        self.calibration_residual_std = (
            self.residual_std
        )

        self.calibration_metrics = {}

        self.is_calibrated = False

    # ============================================================
    # Internal transformations
    # ============================================================

    def _transform(
        self,
        X,
    ) -> np.ndarray:

        X = _ensure_dataframe(
            X,
            self.feature_names,
        )

        imputed = (
            self.imputer.transform(
                X
            )
        )

        scaled = (
            self.scaler.transform(
                imputed
            )
        )

        return scaled

    # ============================================================
    # Ensemble predictions
    # ============================================================

    def _tree_predictions(
        self,
        X,
    ) -> np.ndarray:
        """
        Return individual ExtraTrees predictions.

        Shape:

            observations × trees
        """

        transformed = self._transform(
            X
        )

        predictions = np.column_stack(
            [
                tree.predict(
                    transformed
                )
                for tree
                in self.estimator.estimators_
            ]
        )

        return np.asarray(
            predictions,
            dtype=float,
        )

    def _estimate_training_ensemble_std(
        self,
    ) -> float:
        """
        Estimate typical ensemble spread on the training data.

        This is a descriptive measure of model disagreement.
        """

        try:

            tree_predictions = (
                self._tree_predictions(
                    self.training_X
                )
            )

            tree_std = np.std(
                tree_predictions,
                axis=1,
                ddof=1,
            )

            finite = tree_std[
                np.isfinite(
                    tree_std
                )
            ]

            if len(finite) == 0:
                return 0.0

            return float(
                np.mean(finite)
            )

        except Exception:

            return 0.0

    # ============================================================
    # Point prediction
    # ============================================================

    def predict(
        self,
        X,
    ) -> np.ndarray:
        """
        Predict mean tap temperature.
        """

        transformed = self._transform(
            X
        )

        predictions = (
            self.estimator.predict(
                transformed
            )
        )

        return _as_numpy_1d(
            predictions
        )

    # ============================================================
    # Predictive uncertainty
    # ============================================================

    def predict_distribution(
        self,
        X,
    ) -> pd.DataFrame:
        """
        Calculate nonlinear ensemble predictive distributions.

        The ensemble mean is the point prediction.

        The predictive standard deviation combines:

            1. tree-to-tree ensemble spread
            2. residual/process uncertainty

        Calibration is applied afterwards if a calibration set
        has been supplied.
        """

        X_df = _ensure_dataframe(
            X,
            self.feature_names,
        )

        tree_predictions = (
            self._tree_predictions(
                X_df
            )
        )

        mean_prediction = np.mean(
            tree_predictions,
            axis=1,
        )

        if (
            tree_predictions.shape[1]
            >= 2
        ):

            ensemble_std = np.std(
                tree_predictions,
                axis=1,
                ddof=1,
            )

        else:

            ensemble_std = np.zeros(
                len(mean_prediction),
                dtype=float,
            )

        ensemble_std = np.asarray(
            ensemble_std,
            dtype=float,
        )

        # --------------------------------------------------------
        # Combine ensemble disagreement and residual uncertainty.
        #
        # The residual component represents uncertainty not captured
        # by variation between trees.
        # --------------------------------------------------------

        residual_component = max(
            float(
                self.residual_std
            ),
            EPSILON,
        )

        raw_std = np.sqrt(
            ensemble_std ** 2
            +
            residual_component ** 2
        )

        calibrated_std = (
            raw_std
            *
            float(
                self.calibration_scale
            )
        )

        calibrated_std = np.maximum(
            calibrated_std,
            EPSILON,
        )

        z10 = -1.2815515655446004
        z90 = 1.2815515655446004

        p10 = (
            mean_prediction
            +
            z10
            *
            calibrated_std
        )

        p90 = (
            mean_prediction
            +
            z90
            *
            calibrated_std
        )

        result = pd.DataFrame(
            {
                "temperature_mean":
                    mean_prediction,

                "temperature_median":
                    mean_prediction,

                "temperature_std":
                    calibrated_std,

                "temperature_p10":
                    p10,

                "temperature_p90":
                    p90,
            },
            index=X_df.index,
        )

        return result

    # ============================================================
    # Probability
    # ============================================================

    def probability_in_range(
        self,
        X,
        lower,
        upper,
    ) -> np.ndarray:
        """
        Calculate model-derived Gaussian probability of temperature
        falling inside [lower, upper].
        """

        if upper <= lower:
            raise ValueError(
                "upper must be greater than lower."
            )

        distribution = (
            self.predict_distribution(
                X
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

        std = np.maximum(
            std,
            EPSILON,
        )

        z_lower = (
            float(lower)
            -
            mean
        ) / std

        z_upper = (
            float(upper)
            -
            mean
        ) / std

        probability = (
            _normal_cdf(
                z_upper
            )
            -
            _normal_cdf(
                z_lower
            )
        )

        probability = np.clip(
            probability,
            0.0,
            1.0,
        )

        return np.asarray(
            probability,
            dtype=float,
        )

    # ============================================================
    # Calibration
    # ============================================================

    def calibrate(
        self,
        X_calibration,
        y_calibration,
    ) -> Dict:
        """
        Calibrate predictive uncertainty using a separate
        chronological calibration set.

        Calibration uses the standardized absolute residual:

            |y - mean| / raw_std

        For a Gaussian distribution, the expected absolute
        standardized residual is approximately:

            sqrt(2 / pi)

        The observed calibration residual scale is therefore
        converted into a multiplicative uncertainty scale.

        A robust percentile-based fallback is included for numerical
        stability.
        """

        y_true = _as_numpy_1d(
            y_calibration
        )

        if len(y_true) == 0:
            raise ValueError(
                "Calibration set is empty."
            )

        distribution_before = (
            self._predict_distribution_raw(
                X_calibration
            )
        )

        mean = distribution_before[
            "temperature_mean"
        ].to_numpy(
            dtype=float
        )

        raw_std = distribution_before[
            "temperature_std"
        ].to_numpy(
            dtype=float
        )

        finite_mask = (
            np.isfinite(y_true)
            &
            np.isfinite(mean)
            &
            np.isfinite(raw_std)
        )

        if not finite_mask.any():
            raise ValueError(
                "Calibration data contain no valid observations."
            )

        y_true = y_true[
            finite_mask
        ]

        mean = mean[
            finite_mask
        ]

        raw_std = raw_std[
            finite_mask
        ]

        raw_std = np.maximum(
            raw_std,
            EPSILON,
        )

        residuals = (
            y_true
            -
            mean
        )

        absolute_residuals = np.abs(
            residuals
        )

        # --------------------------------------------------------
        # Empirical uncertainty scale.
        # --------------------------------------------------------

        standardized_abs_error = (
            absolute_residuals
            /
            raw_std
        )

        expected_abs_z = math.sqrt(
            2.0 / math.pi
        )

        empirical_scale = (
            np.median(
                standardized_abs_error
            )
            /
            (
                expected_abs_z
                +
                EPSILON
            )
        )

        if (
            not np.isfinite(
                empirical_scale
            )
            or empirical_scale <= 0
        ):

            empirical_scale = 1.0

        # --------------------------------------------------------
        # Also estimate the calibration residual standard deviation.
        # --------------------------------------------------------

        residual_std = _safe_std(
            residuals
        )

        mean_raw_std = float(
            np.mean(
                raw_std
            )
        )

        if (
            not np.isfinite(
                mean_raw_std
            )
            or mean_raw_std <= EPSILON
        ):

            residual_scale = 1.0

        else:

            residual_scale = (
                residual_std
                /
                mean_raw_std
            )

            if (
                not np.isfinite(
                    residual_scale
                )
                or residual_scale <= 0
            ):

                residual_scale = 1.0

        # --------------------------------------------------------
        # Combine the two calibration estimates.
        #
        # Median standardised residual is robust to outliers.
        # Residual/std ratio reflects absolute calibration error.
        # --------------------------------------------------------

        scale = float(
            0.5
            *
            empirical_scale
            +
            0.5
            *
            residual_scale
        )

        if (
            not np.isfinite(
                scale
            )
            or scale <= EPSILON
        ):

            scale = 1.0

        # --------------------------------------------------------
        # Prevent pathological uncertainty collapse/explosion.
        #
        # This does not improve accuracy; it simply keeps a POC
        # uncertainty estimate numerically stable.
        # --------------------------------------------------------

        scale = float(
            np.clip(
                scale,
                0.10,
                10.0,
            )
        )

        self.calibration_scale = scale

        self.calibration_residual_std = (
            residual_std
        )

        self.is_calibrated = True

        # --------------------------------------------------------
        # Evaluate calibrated intervals on calibration data.
        # --------------------------------------------------------

        calibrated_std = (
            raw_std
            *
            scale
        )

        p10 = (
            mean
            -
            1.2815515655446004
            *
            calibrated_std
        )

        p90 = (
            mean
            +
            1.2815515655446004
            *
            calibrated_std
        )

        coverage = float(
            np.mean(
                (
                    y_true >= p10
                )
                &
                (
                    y_true <= p90
                )
            )
        )

        average_width = float(
            np.mean(
                p90
                -
                p10
            )
        )

        calibration_mae = float(
            mean_absolute_error(
                y_true,
                mean,
            )
        )

        calibration_rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    mean,
                )
            )
        )

        metrics = {
            "calibration_observations":
                int(
                    len(y_true)
                ),

            "calibration_mae":
                calibration_mae,

            "calibration_rmse":
                calibration_rmse,

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
                scale,

            "P10_P90_coverage":
                coverage,

            "P10_P90_average_width":
                average_width,
        }

        self.calibration_metrics = metrics

        return metrics

    # ============================================================
    # Raw distribution
    # ============================================================

    def _predict_distribution_raw(
        self,
        X,
    ) -> pd.DataFrame:
        """
        Predict uncertainty before calibration.

        This is kept separate from predict_distribution() so the
        calibration procedure does not accidentally calibrate itself.
        """

        X_df = _ensure_dataframe(
            X,
            self.feature_names,
        )

        tree_predictions = (
            self._tree_predictions(
                X_df
            )
        )

        mean_prediction = np.mean(
            tree_predictions,
            axis=1,
        )

        if (
            tree_predictions.shape[1]
            >= 2
        ):

            ensemble_std = np.std(
                tree_predictions,
                axis=1,
                ddof=1,
            )

        else:

            ensemble_std = np.zeros(
                len(mean_prediction),
                dtype=float,
            )

        residual_component = max(
            float(
                self.residual_std
            ),
            EPSILON,
        )

        raw_std = np.sqrt(
            ensemble_std ** 2
            +
            residual_component ** 2
        )

        raw_std = np.maximum(
            raw_std,
            EPSILON,
        )

        p10 = (
            mean_prediction
            -
            1.2815515655446004
            *
            raw_std
        )

        p90 = (
            mean_prediction
            +
            1.2815515655446004
            *
            raw_std
        )

        return pd.DataFrame(
            {
                "temperature_mean":
                    mean_prediction,

                "temperature_median":
                    mean_prediction,

                "temperature_std":
                    raw_std,

                "temperature_p10":
                    p10,

                "temperature_p90":
                    p90,
            },
            index=X_df.index,
        )


# ================================================================
# Model training
# ================================================================

def train_model(
    X_train,
    y_train,
    n_estimators=DEFAULT_N_ESTIMATORS,
    max_depth=DEFAULT_MAX_DEPTH,
    min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
    random_state=DEFAULT_RANDOM_STATE,
    alpha=DEFAULT_ALPHA,
):
    """
    Train the nonlinear ExtraTrees EAF regression model.

    Parameters
    ----------
    X_train:
        Training feature DataFrame.

    y_train:
        Training target.

    n_estimators:
        Number of trees in the nonlinear ensemble.

    max_depth:
        Maximum tree depth.

    min_samples_leaf:
        Minimum number of samples in a leaf.

    random_state:
        Reproducibility seed.

    alpha:
        Retained for compatibility with the previous Ridge-based
        interface.

        ExtraTrees does not use alpha, so this value is intentionally
        ignored.
    """

    # ------------------------------------------------------------
    # Preserve compatibility with main.py.
    # ------------------------------------------------------------

    del alpha

    X_train = _ensure_dataframe(
        X_train
    )

    y_train = pd.Series(
        y_train,
        index=X_train.index,
        dtype=float,
    )

    # ------------------------------------------------------------
    # Validate target.
    # ------------------------------------------------------------

    valid_target = (
        y_train.notna()
        &
        np.isfinite(
            y_train
        )
    )

    X_train = X_train.loc[
        valid_target
    ].copy()

    y_train = y_train.loc[
        valid_target
    ].copy()

    if len(X_train) < 10:
        raise ValueError(
            "Too few valid training observations."
        )

    # ------------------------------------------------------------
    # Ensure all features are numeric.
    # ------------------------------------------------------------

    for column in X_train.columns:

        X_train[column] = pd.to_numeric(
            X_train[column],
            errors="coerce",
        )

    feature_names = list(
        X_train.columns
    )

    # ------------------------------------------------------------
    # Imputation.
    #
    # Median imputation is fitted ONLY on the training data.
    # ------------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X_imputed = (
        imputer.fit_transform(
            X_train
        )
    )

    # ------------------------------------------------------------
    # StandardScaler is retained because decision.py expects the
    # fitted model to expose scaler.transform().
    #
    # ExtraTrees itself does not require scaling.
    # ------------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = (
        scaler.fit_transform(
            X_imputed
        )
    )

    # ------------------------------------------------------------
    # Nonlinear regression model.
    # ------------------------------------------------------------

    estimator = ExtraTreesRegressor(
        n_estimators=int(
            n_estimators
        ),

        max_depth=(
            None
            if max_depth is None
            else int(
                max_depth
            )
        ),

        min_samples_leaf=max(
            1,
            int(
                min_samples_leaf
            ),
        ),

        random_state=int(
            random_state
        ),

        n_jobs=-1,

        # Random feature selection increases diversity between trees.
        max_features=1.0,

        bootstrap=False,

        criterion="squared_error",
    )

    estimator.fit(
        X_scaled,
        y_train.to_numpy(
            dtype=float
        ),
    )

    # ------------------------------------------------------------
    # Training residuals.
    # ------------------------------------------------------------

    training_prediction = (
        estimator.predict(
            X_scaled
        )
    )

    residuals = (
        y_train.to_numpy(
            dtype=float
        )
        -
        training_prediction
    )

    residual_std = _safe_std(
        residuals
    )

    # ------------------------------------------------------------
    # Build wrapper.
    # ------------------------------------------------------------

    model = EAFNonlinearModel(
        estimator=estimator,

        imputer=imputer,

        scaler=scaler,

        feature_names=feature_names,

        training_X=X_train,

        training_y=y_train,

        residual_std=residual_std,
    )

    return model


# ================================================================
# Module-level prediction API
# ================================================================

def predict(
    model,
    X,
):
    """
    Predict tap temperature.

    Kept as a module-level function for compatibility with main.py.
    """

    return model.predict(
        X
    )


def predict_distribution(
    model,
    X,
):
    """
    Return predictive mean, median, std, P10 and P90.
    """

    return model.predict_distribution(
        X
    )


def probability_in_range(
    model,
    X,
    lower,
    upper,
):
    """
    Return model-derived probability of target temperature lying
    inside [lower, upper].
    """

    return model.probability_in_range(
        X,
        lower,
        upper,
    )


# ================================================================
# Point-prediction evaluation
# ================================================================

def evaluate_point_prediction(
    y_true,
    predictions,
) -> Dict:
    """
    Calculate standard regression metrics.
    """

    y_true = _as_numpy_1d(
        y_true
    )

    predictions = _as_numpy_1d(
        predictions
    )

    if len(y_true) != len(predictions):
        raise ValueError(
            "Target and prediction lengths do not match."
        )

    finite = (
        np.isfinite(
            y_true
        )
        &
        np.isfinite(
            predictions
        )
    )

    if not finite.any():
        raise ValueError(
            "No finite observations available for evaluation."
        )

    y_true = y_true[
        finite
    ]

    predictions = predictions[
        finite
    ]

    errors = (
        y_true
        -
        predictions
    )

    mae = float(
        mean_absolute_error(
            y_true,
            predictions,
        )
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_true,
                predictions,
            )
        )
    )

    mean_error = float(
        np.mean(
            errors
        )
    )

    return {
        "MAE":
            mae,

        "RMSE":
            rmse,

        "Mean_Error":
            mean_error,
    }


# ================================================================
# Prediction-interval evaluation
# ================================================================

def evaluate_prediction_intervals(
    y_true,
    prediction_distribution,
) -> Dict:
    """
    Evaluate P10-P90 predictive interval coverage and width.
    """

    y_true = _as_numpy_1d(
        y_true
    )

    if not isinstance(
        prediction_distribution,
        pd.DataFrame,
    ):

        prediction_distribution = (
            pd.DataFrame(
                prediction_distribution
            )
        )

    p10 = _as_numpy_1d(
        prediction_distribution[
            "temperature_p10"
        ]
    )

    p90 = _as_numpy_1d(
        prediction_distribution[
            "temperature_p90"
        ]
    )

    if not (
        len(y_true)
        ==
        len(p10)
        ==
        len(p90)
    ):

        raise ValueError(
            "Target and interval lengths do not match."
        )

    finite = (
        np.isfinite(
            y_true
        )
        &
        np.isfinite(
            p10
        )
        &
        np.isfinite(
            p90
        )
    )

    if not finite.any():
        raise ValueError(
            "No finite observations available for interval evaluation."
        )

    y_true = y_true[
        finite
    ]

    p10 = p10[
        finite
    ]

    p90 = p90[
        finite
    ]

    coverage = float(
        np.mean(
            (
                y_true >= p10
            )
            &
            (
                y_true <= p90
            )
        )
    )

    average_width = float(
        np.mean(
            p90
            -
            p10
        )
    )

    return {
        "P10_P90_coverage":
            coverage,

        "P10_P90_average_width":
            average_width,
    }


# ================================================================
# Global feature importance
# ================================================================

def feature_importance(
    model,
) -> pd.DataFrame:
    """
    Return nonlinear-model feature importance.

    Two measures are reported:

        impurity_importance
        permutation_importance

    The ExtraTrees impurity importance captures how much each feature
    contributes to tree splits.

    Permutation importance measures the increase in prediction error
    when a feature is randomly shuffled.

    Permutation importance is generally more useful for interpreting
    predictive relevance, while impurity importance is useful as a
    complementary model-internal measure.
    """

    estimator = model.estimator

    impurity = np.asarray(
        estimator.feature_importances_,
        dtype=float,
    )

    # ------------------------------------------------------------
    # Permutation importance is calculated using the training data.
    #
    # This is a descriptive model diagnostic rather than a generalised
    # out-of-sample importance estimate.
    # ------------------------------------------------------------

    X_train = model.training_X

    y_train = model.training_y

    transformed = model._transform(
        X_train
    )

    permutation = permutation_importance(
        estimator,
        transformed,
        y_train.to_numpy(
            dtype=float
        ),
        n_repeats=5,
        random_state=42,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    permutation_mean = (
        permutation.importances_mean
    )

    permutation_std = (
        permutation.importances_std
    )

    result = pd.DataFrame(
        {
            "feature":
                model.feature_names,

            "impurity_importance":
                impurity,

            "permutation_importance":
                permutation_mean,

            "permutation_std":
                permutation_std,
        }
    )

    result[
        "absolute_permutation_importance"
    ] = np.abs(
        result[
            "permutation_importance"
        ]
    )

    result = result.sort_values(
        [
            "absolute_permutation_importance",
            "impurity_importance",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    return result


def save_feature_importance(
    model,
    output_file,
):
    """
    Calculate and save feature importance.
    """

    output_file = Path(
        output_file
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance = feature_importance(
        model
    )

    importance.to_csv(
        output_file,
        index=False,
    )

    return importance


# ================================================================
# Local explanation
# ================================================================

def explain_prediction(
    model,
    state,
    top_n=10,
) -> pd.DataFrame:
    """
    Explain an individual prediction using feature perturbations.

    Because ExtraTrees is nonlinear, a Ridge-style coefficient
    explanation would no longer be valid.

    Instead, this function estimates each feature's local contribution
    by comparing:

        prediction at actual state

    against:

        prediction when that feature is replaced by its training
        median while all other features remain unchanged.

    The resulting values are local predictive sensitivity estimates.

    Positive contribution:
        feature value moves the prediction above the baseline.

    Negative contribution:
        feature value moves the prediction below the baseline.

    Important:
        These are model-based local explanations, not causal effects.
    """

    if isinstance(
        state,
        pd.Series,
    ):

        state_dict = state.to_dict()

    elif isinstance(
        state,
        dict,
    ):

        state_dict = dict(
            state
        )

    else:

        raise TypeError(
            "state must be a dictionary or pandas Series."
        )

    state_df = _ensure_dataframe(
        pd.DataFrame(
            [state_dict]
        ),
        model.feature_names,
    )

    # ------------------------------------------------------------
    # Actual prediction.
    # ------------------------------------------------------------

    actual_prediction = float(
        model.predict(
            state_df
        )[0]
    )

    # ------------------------------------------------------------
    # Baseline feature values.
    #
    # Use medians calculated from the historical training data.
    # ------------------------------------------------------------

    baseline = {}

    for feature in model.feature_names:

        values = pd.to_numeric(
            model.training_X[
                feature
            ],
            errors="coerce",
        )

        median = values.median()

        if not np.isfinite(
            median
        ):

            median = 0.0

        baseline[
            feature
        ] = float(
            median
        )

    # ------------------------------------------------------------
    # Calculate one-feature-at-a-time perturbation effects.
    # ------------------------------------------------------------

    rows = []

    for feature in model.feature_names:

        actual_value = state_df.iloc[
            0
        ][
            feature
        ]

        if pd.isna(
            actual_value
        ):

            actual_value = baseline[
                feature
            ]

        perturbed = state_df.copy()

        perturbed.loc[
            0,
            feature
        ] = baseline[
            feature
        ]

        baseline_prediction = float(
            model.predict(
                perturbed
            )[0]
        )

        contribution = (
            actual_prediction
            -
            baseline_prediction
        )

        rows.append(
            {
                "feature":
                    feature,

                "value":
                    float(
                        actual_value
                    )
                    if np.isfinite(
                        float(
                            actual_value
                        )
                    )
                    else np.nan,

                "baseline_value":
                    baseline[
                        feature
                    ],

                "baseline_prediction":
                    baseline_prediction,

                "prediction":
                    actual_prediction,

                "contribution":
                    float(
                        contribution
                    ),

                "absolute_contribution":
                    float(
                        abs(
                            contribution
                        )
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    result = result.sort_values(
        "absolute_contribution",
        ascending=False,
    ).reset_index(
        drop=True
    )

    return result.head(
        int(
            top_n
        )
    ).copy()


# ================================================================
# Optional convenience diagnostics
# ================================================================

def model_summary(
    model,
) -> Dict:
    """
    Return a compact model summary.
    """

    return {
        "model_type":
            "ExtraTreesRegressor",

        "n_estimators":
            int(
                len(
                    model.estimator.estimators_
                )
            ),

        "max_depth":
            model.estimator.max_depth,

        "min_samples_leaf":
            model.estimator.min_samples_leaf,

        "n_observations":
            model.n_observations,

        "n_features":
            model.n_features,

        "residual_std":
            model.residual_std,

        "raw_ensemble_std":
            model.raw_ensemble_std,

        "calibration_scale":
            model.calibration_scale,

        "is_calibrated":
            model.is_calibrated,
    }
