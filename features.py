"""
features.py

Feature engineering for the EAF probabilistic decision-support POC.

Design principles
-----------------
1. Keep the feature set small and interpretable.
2. Avoid identifiers and timestamps as predictors.
3. Avoid target leakage.
4. Preserve physically meaningful EAF variables.
5. Remove features with excessive missingness before modelling.
6. Provide a stable interface to main.py.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


# ================================================================
# Configuration
# ================================================================

TARGET_COLUMN = "temp_tap"

# ------------------------------------------------
# Maximum allowed feature missingness.
#
# Features with MORE than this percentage of missing
# observations are excluded from the modelling dataset.
#
# IMPORTANT:
# This filtering happens before model.py performs
# any training-set imputation.
# ------------------------------------------------

MAX_MISSING_PCT = 80.0


# ------------------------------------------------
# Primary modelling features
# ------------------------------------------------
#
# These are variables that describe the heat/process.
# They deliberately exclude:
#
# - HEATID
# - timestamps
# - final target
# - derived variables that directly encode the target
# - quality_ok
#
# The POC therefore remains relatively easy to explain.
# ------------------------------------------------

BASE_FEATURES = [
    "energy_mwh",
    "power_on_min",
    "n_power_segments",
    "n_tap_changes",
    "mean_mw",

    "basket_amount",
    "basket_n",
    "charge_t",

    "eaf_added_amount",
    "eaf_added_n",

    "o2_total",
    "gas_total",
    "carbon_total",

    "o2_flow_mean",
    "carbon_flow_mean",

    "temp_first",
    "n_temp_meas",

    "eaf_c",
    "eaf_si",
    "eaf_mn",
    "eaf_p",
    "eaf_s",
    "eaf_cu",
    "eaf_cr",
    "eaf_mo",
    "eaf_ni",
    "eaf_as",
    "eaf_sn",
    "eaf_n",

    "ladle_tap_amount",
    "ladle_tap_n",
]


# ================================================================
# Target
# ================================================================

def get_target_column(
    df: pd.DataFrame,
    preferred_target: str = TARGET_COLUMN,
) -> str:
    """
    Return the target column.

    Parameters
    ----------
    df:
        Input dataframe.

    preferred_target:
        Preferred target column.

    Returns
    -------
    str
        Target column name.
    """

    if preferred_target in df.columns:
        return preferred_target

    # Simple fallback.
    candidates = [
        "temp_tap",
        "tap_temperature",
        "tap_temp",
    ]

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Could not identify a tap-temperature target column."
    )


# ================================================================
# Feature engineering
# ================================================================

def prepare_features(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """
    Construct the modelling dataframe.

    The returned dataframe contains:

        selected modelling features
        +
        target column

    Features with more than MAX_MISSING_PCT missing observations
    are excluded before modelling.

    Missing values in retained numerical features are NOT imputed
    here. They are subsequently imputed by model.py using
    training-set statistics.

    Parameters
    ----------
    df:
        Input EAF dataframe.

    target_column:
        Name of the tap-temperature target.

    Returns
    -------
    pd.DataFrame
        Modelling dataframe containing retained features and target.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "does not exist."
        )

    available_features = [
        feature
        for feature in BASE_FEATURES
        if feature in df.columns
    ]

    if not available_features:
        raise ValueError(
            "None of the configured modelling features "
            "were found in the dataset."
        )

    # ------------------------------------------------------------
    # Select configured features and target.
    # ------------------------------------------------------------

    modelling_df = df[
        available_features + [target_column]
    ].copy()

    # ------------------------------------------------------------
    # Convert all modelling variables to numeric.
    # ------------------------------------------------------------

    for column in available_features:
        modelling_df[column] = pd.to_numeric(
            modelling_df[column],
            errors="coerce",
        )

    modelling_df[target_column] = pd.to_numeric(
        modelling_df[target_column],
        errors="coerce",
    )

    # ------------------------------------------------------------
    # Replace infinities with NaN.
    #
    # This must happen BEFORE calculating missingness so that
    # infinite values are treated as invalid/missing observations.
    # ------------------------------------------------------------

    modelling_df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    # ------------------------------------------------------------
    # Calculate feature missingness BEFORE imputation.
    # ------------------------------------------------------------

    missing_pct = (
        modelling_df[available_features]
        .isna()
        .mean()
        * 100.0
    )

    # ------------------------------------------------------------
    # Remove features with MORE than the configured threshold.
    #
    # Deliberately use > rather than >=.
    #
    # Therefore:
    #     80.00% missing -> retained
    #     80.01% missing -> removed
    # ------------------------------------------------------------

    features_to_remove = (
        missing_pct[
            missing_pct > MAX_MISSING_PCT
        ]
        .index
        .tolist()
    )

    retained_features = [
        feature
        for feature in available_features
        if feature not in features_to_remove
    ]

    # ------------------------------------------------------------
    # Report filtering decision.
    # ------------------------------------------------------------

    print()
    print(
        "Feature missingness filter:"
    )

    print(
        f"Maximum allowed missingness: "
        f"{MAX_MISSING_PCT:.1f}%"
    )

    if features_to_remove:

        print(
            "Features removed "
            f"(>{MAX_MISSING_PCT:.1f}% missing):"
        )

        for feature in features_to_remove:

            print(
                f"  - {feature}: "
                f"{missing_pct[feature]:.2f}% missing"
            )

    else:

        print(
            "No features exceeded the "
            f"{MAX_MISSING_PCT:.1f}% missingness threshold."
        )

    print(
        f"Features before filtering: "
        f"{len(available_features)}"
    )

    print(
        f"Features after filtering: "
        f"{len(retained_features)}"
    )

    # ------------------------------------------------------------
    # Build final modelling dataframe.
    # ------------------------------------------------------------

    if not retained_features:
        raise ValueError(
            "All configured modelling features were removed "
            "by the missingness filter."
        )

    modelling_df = modelling_df[
        retained_features + [target_column]
    ].copy()

    return modelling_df


# ================================================================
# Feature list
# ================================================================

def get_feature_columns(
    modelling_df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> List[str]:
    """
    Return modelling feature columns.

    Only numeric columns are returned and the target is excluded.
    """

    features = [
        column
        for column in modelling_df.columns
        if column != target_column
    ]

    numeric_features = []

    for column in features:

        if pd.api.types.is_numeric_dtype(
            modelling_df[column]
        ):
            numeric_features.append(column)

    return numeric_features


# ================================================================
# Optional helper
# ================================================================

def get_base_features() -> List[str]:
    """
    Return a copy of the configured base feature list.
    """

    return BASE_FEATURES.copy()
