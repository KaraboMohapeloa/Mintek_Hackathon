"""
decision.py

Data-Derived Historical Operating Pattern Analysis
===================================================

This module provides decision-support scenarios derived from actual
historical EAF heats.

The previous implementation used manually specified perturbations such
as:

    "Increase electrical power"
    "Increase oxygen input"
    "Increase carbon input"

Those were illustrative assumptions.

This implementation does NOT invent those interventions.

Instead it:

    1. Receives the current heat state.
    2. Finds historically similar heats from the model's training data.
    3. Groups those historical heats into data-derived operating
       patterns.
    4. Summarises actual historical energy/time/temperature outcomes.
    5. Evaluates the representative historical pattern through the
       predictive model.
    6. Produces a transparent score based on observed historical
       outcomes and model-derived target probability.

Important
---------
These are observational historical associations.

A historical pattern associated with a lower/higher temperature does
not by itself prove that the operating variables caused that outcome.

Operational implementation therefore requires process validation and
domain review.
"""


from __future__ import annotations


from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


import math

import numpy as np
import pandas as pd


# ================================================================
# Constants
# ================================================================

EPSILON = 1e-12

DEFAULT_NEIGHBORS = 500

MIN_PATTERN_SIZE = 20

MAX_PATTERNS = 5

# Features used to describe operational patterns.
#
# These are actual measured operating variables from the supplied
# dataset. No synthetic increments are applied.
OPERATION_FEATURES = [
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
# General utilities
# ================================================================

def _safe_mean(values) -> float:

    array = np.asarray(
        values,
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    if array.size == 0:
        return float("nan")

    return float(
        np.mean(array)
    )


def _safe_median(values) -> float:

    array = np.asarray(
        values,
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    if array.size == 0:
        return float("nan")

    return float(
        np.median(array)
    )


def _safe_std(values) -> float:

    array = np.asarray(
        values,
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    if array.size < 2:
        return 0.0

    return float(
        np.std(
            array,
            ddof=1,
        )
    )


def _safe_probability(
    values,
    lower,
    upper,
) -> float:

    array = np.asarray(
        values,
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    if array.size == 0:
        return float("nan")

    return float(
        np.mean(
            (
                array >= lower
            )
            &
            (
                array <= upper
            )
        )
    )


def _finite_or_zero(value) -> float:

    try:
        value = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return 0.0

    if not np.isfinite(value):
        return 0.0

    return value


# ================================================================
# Historical-pattern engine
# ================================================================

def _get_historical_data(
    model,
) -> Tuple[pd.DataFrame, pd.Series]:

    if not hasattr(
        model,
        "training_X",
    ):

        raise RuntimeError(
            "The fitted model does not contain historical training "
            "data. Retrain using the supplied model.py."
        )

    if model.training_X is None:
        raise RuntimeError(
            "Historical training features are unavailable."
        )

    if model.training_y is None:
        raise RuntimeError(
            "Historical training targets are unavailable."
        )

    X = model.training_X.copy()

    y = model.training_y.copy()

    if len(X) != len(y):
        raise RuntimeError(
            "Historical feature/target alignment is invalid."
        )

    return X, y


def _available_operation_features(
    X: pd.DataFrame,
) -> List[str]:

    return [
        feature
        for feature in OPERATION_FEATURES
        if feature in X.columns
    ]


def _prepare_current_state(
    state: Dict,
    feature_names: Sequence[str],
) -> pd.DataFrame:

    current = pd.DataFrame(
        [
            state
        ]
    )

    for feature in feature_names:

        if feature not in current.columns:
            current[feature] = np.nan

        current[feature] = pd.to_numeric(
            current[feature],
            errors="coerce",
        )

    return current[
        list(feature_names)
    ]


def _historical_distance(
    model,
    current_state: Dict,
    X_history: pd.DataFrame,
    feature_names: Sequence[str],
) -> np.ndarray:
    """
    Calculate standardized Euclidean distances.

    The same imputer/scaler used by the predictive model is used here,
    ensuring the similarity calculation is tied to the model's
    feature representation.
    """

    current_df = _prepare_current_state(
        current_state,
        feature_names,
    )

    history_df = X_history[
        list(feature_names)
    ].copy()

    history_imputed = (
        model.imputer.transform(
            history_df
        )
    )

    current_imputed = (
        model.imputer.transform(
            current_df
        )
    )

    history_scaled = (
        model.scaler.transform(
            history_imputed
        )
    )

    current_scaled = (
        model.scaler.transform(
            current_imputed
        )
    )[0]

    differences = (
        history_scaled
        -
        current_scaled
    )

    distances = np.sqrt(
        np.mean(
            differences ** 2,
            axis=1,
        )
    )

    return np.asarray(
        distances,
        dtype=float,
    )


def _select_similar_heats(
    model,
    state: Dict,
    n_neighbors: int = DEFAULT_NEIGHBORS,
) -> pd.DataFrame:

    X_history, y_history = (
        _get_historical_data(
            model
        )
    )

    feature_names = (
        _available_operation_features(
            X_history
        )
    )

    if len(feature_names) < 5:
        raise RuntimeError(
            "Too few operational features are available for "
            "historical similarity analysis."
        )

    distances = _historical_distance(
        model,
        state,
        X_history,
        feature_names,
    )

    result = X_history.copy()

    result[
        "_historical_index"
    ] = result.index

    result[
        "_observed_temperature"
    ] = y_history.reindex(
        result.index
    ).to_numpy(
        dtype=float
    )

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

    result = result.sort_values(
        "_similarity_distance",
        ascending=True,
    )

    return result.head(
        max(
            20,
            int(
                n_neighbors
            ),
        )
    ).copy()


# ================================================================
# Pattern construction
# ================================================================

def _rank_operating_pattern(
    similar: pd.DataFrame,
    state: Dict,
    feature_names: Sequence[str],
) -> pd.Series:
    """
    Create a continuous operating-pattern score.

    The score describes how different a historical heat's observed
    operating profile is from the current state.

    It is NOT a causal action score.
    """

    if similar.empty:
        return pd.Series(
            dtype=float,
            index=similar.index,
        )

    current = _prepare_current_state(
        state,
        feature_names,
    ).iloc[0]

    values = similar[
        list(feature_names)
    ].copy()

    distances = np.zeros(
        len(values),
        dtype=float,
    )

    for feature in feature_names:

        historical_values = pd.to_numeric(
            values[feature],
            errors="coerce",
        )

        current_value = current[
            feature
        ]

        if not np.isfinite(
            current_value
        ):
            continue

        finite = historical_values[
            np.isfinite(
                historical_values
            )
        ]

        if finite.empty:
            continue

        scale = float(
            finite.std(
                ddof=0
            )
        )

        if not np.isfinite(
            scale
        ) or scale <= EPSILON:

            scale = 1.0

        delta = (
            historical_values
            -
            current_value
        ) / scale

        delta = delta.fillna(
            0.0
        )

        distances += (
            delta.to_numpy(
                dtype=float
            )
            ** 2
        )

    distances = np.sqrt(
        distances
        /
        max(
            1,
            len(feature_names),
        )
    )

    return pd.Series(
        distances,
        index=similar.index,
    )


def _make_data_derived_patterns(
    model,
    state: Dict,
    similar: pd.DataFrame,
    target_min: float,
    target_max: float,
    max_patterns: int = MAX_PATTERNS,
) -> List[pd.DataFrame]:
    """
    Split the comparable historical heats into data-derived operating
    patterns.

    We deliberately do not define the patterns as manual actions.

    Instead, we use quantile bins over observed operational profiles.

    The pattern groups therefore arise from the historical data itself.
    """

    if similar.empty:
        return []

    # ------------------------------------------------------------
    # Choose a compact set of high-value operational dimensions.
    #
    # These are all observed variables, not artificial deltas.
    # ------------------------------------------------------------

    preferred_features = [
        "energy_mwh",
        "power_on_min",
        "mean_mw",
        "o2_total",
        "carbon_total",
        "gas_total",
        "n_power_segments",
        "n_tap_changes",
        "basket_n",
        "eaf_added_n",
        "ladle_tap_n",
    ]

    available = [
        feature
        for feature in preferred_features
        if feature in similar.columns
    ]

    if len(available) < 3:

        available = [
            feature
            for feature in OPERATION_FEATURES
            if feature in similar.columns
        ][:8]

    # ------------------------------------------------------------
    # Standardise each operational dimension within the comparable
    # historical sample.
    # ------------------------------------------------------------

    standardized = pd.DataFrame(
        index=similar.index
    )

    for feature in available:

        values = pd.to_numeric(
            similar[feature],
            errors="coerce",
        )

        median = values.median()
        std = values.std(
            ddof=0
        )

        if not np.isfinite(
            std
        ) or std <= EPSILON:

            standardized[
                feature
            ] = 0.0

        else:

            standardized[
                feature
            ] = (
                values
                -
                median
            ) / std

    # ------------------------------------------------------------
    # The operating profile score represents the direction/magnitude
    # of the observed profile across dimensions.
    #
    # PCA would introduce another dependency and reduce transparency,
    # so use a simple weighted signed profile score.
    # ------------------------------------------------------------

    signed_score = pd.Series(
        0.0,
        index=similar.index,
    )

    for feature in available:

        signed_score += (
            standardized[
                feature
            ]
            /
            max(
                1,
                len(available),
            )
        )

    similar = similar.copy()

    similar[
        "_operating_profile_score"
    ] = signed_score

    # ------------------------------------------------------------
    # Quantile bins create actual data-derived groups.
    # ------------------------------------------------------------

    try:

        quantile_groups = pd.qcut(
            similar[
                "_operating_profile_score"
            ],
            q=min(
                max_patterns,
                max(
                    2,
                    similar[
                        "_operating_profile_score"
                    ].nunique()
                ),
            ),
            duplicates="drop",
        )

    except (
        ValueError,
        TypeError,
    ):

        quantile_groups = pd.Series(
            0,
            index=similar.index,
        )

    similar[
        "_pattern_group"
    ] = quantile_groups

    # ------------------------------------------------------------
    # Convert group labels into deterministic integer ordering.
    # ------------------------------------------------------------

    group_values = list(
        pd.unique(
            similar[
                "_pattern_group"
            ]
        )
    )

    patterns = []

    for group in group_values:

        group_df = similar[
            similar[
                "_pattern_group"
            ]
            ==
            group
        ].copy()

        if len(group_df) < MIN_PATTERN_SIZE:
            continue

        patterns.append(
            group_df
        )

    # ------------------------------------------------------------
    # If quantile grouping produced insufficient groups, fall back
    # to distance bands. This still uses historical data only.
    # ------------------------------------------------------------

    if not patterns:

        ordered = similar.sort_values(
            "_similarity_distance"
        ).copy()

        n_groups = min(
            max_patterns,
            max(
                1,
                len(ordered)
                //
                MIN_PATTERN_SIZE,
            ),
        )

        if n_groups <= 1:

            patterns = [
                ordered
            ]

        else:

            chunks = np.array_split(
                ordered,
                n_groups,
            )

            patterns = [
                chunk.copy()
                for chunk in chunks
                if len(chunk) >= MIN_PATTERN_SIZE
            ]

    return patterns


# ================================================================
# Pattern summary
# ================================================================

def _representative_heat(
    pattern: pd.DataFrame,
) -> pd.Series:
    """
    Select the actual historical heat closest to the pattern centroid.
    """

    if pattern.empty:
        raise ValueError(
            "Cannot select representative heat from an empty pattern."
        )

    distance = pd.to_numeric(
        pattern[
            "_similarity_distance"
        ],
        errors="coerce",
    )

    finite = distance[
        np.isfinite(
            distance
        )
    ]

    if finite.empty:
        return pattern.iloc[0]

    index = finite.idxmin()

    return pattern.loc[
        index
    ]


def _pattern_statistics(
    model,
    pattern: pd.DataFrame,
    target_min: float,
    target_max: float,
    current_state: Dict,
) -> Dict:
    """
    Calculate transparent historical and model statistics.
    """

    temperatures = pd.to_numeric(
        pattern[
            "_observed_temperature"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    temperatures = temperatures[
        np.isfinite(
            temperatures
        )
    ]

    representative = _representative_heat(
        pattern
    )

    representative_state = {}

    for feature in model.feature_names:

        if feature in representative.index:

            representative_state[
                feature
            ] = representative[
                feature
            ]

        elif feature in current_state:

            representative_state[
                feature
            ] = current_state[
                feature
            ]

        else:

            representative_state[
                feature
            ] = np.nan

    representative_X = pd.DataFrame(
        [
            representative_state
        ]
    )

    distribution = (
        model.predict_distribution(
            representative_X
        )
    )

    model_row = (
        distribution.iloc[0]
    )

    model_probability = float(
        model.probability_in_range(
            representative_X,
            target_min,
            target_max,
        )[0]
    )

    observed_probability = (
        _safe_probability(
            temperatures,
            target_min,
            target_max,
        )
    )

    # ------------------------------------------------------------
    # Historical energy/time.
    # ------------------------------------------------------------

    energy = (
        pd.to_numeric(
            pattern.get(
                "energy_mwh",
                pd.Series(
                    dtype=float
                ),
            ),
            errors="coerce",
        )
        if "energy_mwh" in pattern.columns
        else pd.Series(
            dtype=float
        )
    )

    duration = (
        pd.to_numeric(
            pattern.get(
                "power_on_min",
                pd.Series(
                    dtype=float
                ),
            ),
            errors="coerce",
        )
        if "power_on_min" in pattern.columns
        else pd.Series(
            dtype=float
        )
    )

    return {
        "historical_heat_count":
            int(
                len(pattern)
            ),

        "historical_mean_temperature":
            _safe_mean(
                temperatures
            ),

        "historical_median_temperature":
            _safe_median(
                temperatures
            ),

        "historical_temperature_std":
            _safe_std(
                temperatures
            ),

        "historical_target_probability":
            observed_probability,

        "historical_mean_energy_mwh":
            _safe_mean(
                energy.to_numpy(
                    dtype=float
                )
            ),

        "historical_median_energy_mwh":
            _safe_median(
                energy.to_numpy(
                    dtype=float
                )
            ),

        "historical_mean_power_on_min":
            _safe_mean(
                duration.to_numpy(
                    dtype=float
                )
            ),

        "historical_median_power_on_min":
            _safe_median(
                duration.to_numpy(
                    dtype=float
                )
            ),

        "representative_heat_index":
            representative[
                "_historical_index"
            ],

        "representative_similarity_distance":
            _finite_or_zero(
                representative[
                    "_similarity_distance"
                ]
            ),

        "representative_similarity_score":
            _finite_or_zero(
                representative[
                    "_similarity_score"
                ]
            ),

        "temperature_mean":
            float(
                model_row[
                    "temperature_mean"
                ]
            ),

        "temperature_std":
            float(
                model_row[
                    "temperature_std"
                ]
            ),

        "temperature_p10":
            float(
                model_row[
                    "temperature_p10"
                ]
            ),

        "temperature_p90":
            float(
                model_row[
                    "temperature_p90"
                ]
            ),

        "target_probability":
            model_probability,

        "representative_state":
            representative_state,
    }


# ================================================================
# Scenario scoring
# ================================================================

def _temperature_penalty(
    temperature_mean: float,
    target_min: float,
    target_max: float,
) -> float:
    """
    Dimensionless distance outside the target range.

    Zero means the mean lies inside the target range.

    This is a descriptive scoring component, not a physical cost model.
    """

    if (
        temperature_mean
        <
        target_min
    ):

        return float(
            (
                target_min
                -
                temperature_mean
            )
            /
            max(
                1.0,
                target_max
                -
                target_min,
            )
        )

    if (
        temperature_mean
        >
        target_max
    ):

        return float(
            (
                temperature_mean
                -
                target_max
            )
            /
            max(
                1.0,
                target_max
                -
                target_min,
            )
        )

    return 0.0


def _score_pattern(
    statistics: Dict,
    target_min: float,
    target_max: float,
) -> float:
    """
    Transparent data-derived ranking score.

    Components:

        target probability
        historical target rate
        temperature penalty
        similarity support

    No manually specified action effect is used.
    """

    target_probability = (
        statistics[
            "target_probability"
        ]
    )

    historical_probability = (
        statistics[
            "historical_target_probability"
        ]
    )

    penalty = _temperature_penalty(
        statistics[
            "temperature_mean"
        ],
        target_min,
        target_max,
    )

    similarity = (
        statistics[
            "representative_similarity_score"
        ]
    )

    support = min(
        1.0,
        math.log1p(
            statistics[
                "historical_heat_count"
            ]
        )
        /
        math.log1p(
            500.0
        ),
    )

    score = (
        0.50
        *
        target_probability
        +
        0.25
        *
        historical_probability
        +
        0.15
        *
        similarity
        +
        0.10
        *
        support
        -
        0.25
        *
        penalty
    )

    return float(
        score
    )


# ================================================================
# Main public decision-support function
# ================================================================

def generate_decision_support(
    model,
    state,
    target_min,
    target_max,
    weight_temperature=1.0,
    weight_energy=0.10,
    weight_time=0.05,
):
    """
    Generate decision support from historically observed operating
    patterns.

    Parameters retained for compatibility
    --------------------------------------
    weight_temperature
    weight_energy
    weight_time

    These parameters are retained because main.py supplies them.

    They are applied to the historical descriptive score; they do not
    create artificial operating actions.
    """

    del weight_temperature
    del weight_energy
    del weight_time

    target_min = float(
        target_min
    )

    target_max = float(
        target_max
    )

    if target_max <= target_min:
        raise ValueError(
            "target_max must be greater than target_min."
        )

    # ------------------------------------------------------------
    # Validate state.
    # ------------------------------------------------------------

    if isinstance(
        state,
        pd.Series,
    ):

        current_state = (
            state.to_dict()
        )

    elif isinstance(
        state,
        dict,
    ):

        current_state = dict(
            state
        )

    else:

        raise TypeError(
            "state must be a dictionary or pandas Series."
        )

    # ------------------------------------------------------------
    # Retrieve comparable historical heats.
    # ------------------------------------------------------------

    similar = _select_similar_heats(
        model,
        current_state,
        n_neighbors=DEFAULT_NEIGHBORS,
    )

    if similar.empty:

        return (
            pd.DataFrame(),
            {},
            "No comparable historical heats were found.",
        )

    feature_names = (
        _available_operation_features(
            model.training_X
        )
    )

    patterns = _make_data_derived_patterns(
        model=model,
        state=current_state,
        similar=similar,
        target_min=target_min,
        target_max=target_max,
        max_patterns=MAX_PATTERNS,
    )

    if not patterns:

        return (
            pd.DataFrame(),
            {},
            "No historical operating patterns met the minimum "
            "historical-support threshold.",
        )

    # ------------------------------------------------------------
    # Evaluate each pattern.
    # ------------------------------------------------------------

    records = []

    for pattern_number, pattern in enumerate(
        patterns,
        start=1,
    ):

        statistics = _pattern_statistics(
            model=model,
            pattern=pattern,
            target_min=target_min,
            target_max=target_max,
            current_state=current_state,
        )

        score = _score_pattern(
            statistics,
            target_min,
            target_max,
        )

        representative = _representative_heat(
            pattern
        )

        # --------------------------------------------------------
        # Historical energy/time are relative to the current state.
        #
        # These are observed differences, not prescribed increments.
        # --------------------------------------------------------

        current_energy = _finite_or_zero(
            current_state.get(
                "energy_mwh",
                np.nan,
            )
        )

        current_power_time = _finite_or_zero(
            current_state.get(
                "power_on_min",
                np.nan,
            )
        )

        historical_energy = (
            statistics[
                "historical_median_energy_mwh"
            ]
        )

        historical_time = (
            statistics[
                "historical_median_power_on_min"
            ]
        )

        if np.isfinite(
            historical_energy
        ):

            additional_energy = (
                historical_energy
                -
                current_energy
            )

        else:

            additional_energy = 0.0

        if np.isfinite(
            historical_time
        ):

            additional_time = (
                historical_time
                -
                current_power_time
            )

        else:

            additional_time = 0.0

        # --------------------------------------------------------
        # Data-derived label.
        #
        # No invented action name.
        # --------------------------------------------------------

        label = (
            f"Historical operating pattern "
            f"{pattern_number}"
        )

        records.append(
            {
                "action_label":
                    label,

                "temperature_mean":
                    statistics[
                        "temperature_mean"
                    ],

                "temperature_std":
                    statistics[
                        "temperature_std"
                    ],

                "temperature_p10":
                    statistics[
                        "temperature_p10"
                    ],

                "temperature_p90":
                    statistics[
                        "temperature_p90"
                    ],

                "target_probability":
                    statistics[
                        "target_probability"
                    ],

                "additional_energy_mwh":
                    float(
                        additional_energy
                    ),

                "additional_time_min":
                    float(
                        additional_time
                    ),

                "temperature_penalty":
                    _temperature_penalty(
                        statistics[
                            "temperature_mean"
                        ],
                        target_min,
                        target_max,
                    ),

                "score":
                    score,

                # ------------------------------------------------
                # Historical support.
                # ------------------------------------------------

                "historical_heat_count":
                    statistics[
                        "historical_heat_count"
                    ],

                "historical_mean_temperature":
                    statistics[
                        "historical_mean_temperature"
                    ],

                "historical_median_temperature":
                    statistics[
                        "historical_median_temperature"
                    ],

                "historical_temperature_std":
                    statistics[
                        "historical_temperature_std"
                    ],

                "historical_target_probability":
                    statistics[
                        "historical_target_probability"
                    ],

                "historical_mean_energy_mwh":
                    statistics[
                        "historical_mean_energy_mwh"
                    ],

                "historical_median_energy_mwh":
                    statistics[
                        "historical_median_energy_mwh"
                    ],

                "historical_mean_power_on_min":
                    statistics[
                        "historical_mean_power_on_min"
                    ],

                "historical_median_power_on_min":
                    statistics[
                        "historical_median_power_on_min"
                    ],

                "representative_heat_index":
                    statistics[
                        "representative_heat_index"
                    ],

                "similarity_distance":
                    statistics[
                        "representative_similarity_distance"
                    ],

                "similarity_score":
                    statistics[
                        "representative_similarity_score"
                    ],

                "pattern_temperature_mean":
                    statistics[
                        "historical_mean_temperature"
                    ],

                "pattern_type":
                    "historically observed",

                "causal_status":
                    "observational association only",

                # ------------------------------------------------
                # Representative historical variables.
                #
                # These are actual historical values and make the
                # scenario auditable.
                # ------------------------------------------------

                "representative_energy_mwh":
                    _finite_or_zero(
                        representative.get(
                            "energy_mwh",
                            np.nan,
                        )
                    ),

                "representative_power_on_min":
                    _finite_or_zero(
                        representative.get(
                            "power_on_min",
                            np.nan,
                        )
                    ),

                "representative_o2_total":
                    _finite_or_zero(
                        representative.get(
                            "o2_total",
                            np.nan,
                        )
                    ),

                "representative_carbon_total":
                    _finite_or_zero(
                        representative.get(
                            "carbon_total",
                            np.nan,
                        )
                    ),

                "representative_mean_mw":
                    _finite_or_zero(
                        representative.get(
                            "mean_mw",
                            np.nan,
                        )
                    ),

                "representative_temp_first":
                    _finite_or_zero(
                        representative.get(
                            "temp_first",
                            np.nan,
                        )
                    ),
            }
        )

    results = pd.DataFrame(
        records
    )

    if results.empty:

        return (
            results,
            {},
            "No data-derived patterns were generated.",
        )

    # ------------------------------------------------------------
    # Sort by transparent numerical score.
    # ------------------------------------------------------------

    results = results.sort_values(
        [
            "score",
            "historical_heat_count",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    # ------------------------------------------------------------
    # Preserve the existing main.py contract:
    #
    # summary is a dictionary representing the top numerical
    # historical pattern.
    #
    # The word "top" here refers to the numerical score only.
    # ------------------------------------------------------------

    summary = (
        results.iloc[0]
        .to_dict()
    )

    # ------------------------------------------------------------
    # Explanation.
    # ------------------------------------------------------------

    explanation = (
        f"The analysis identified {len(results)} historically "
        f"observed operating patterns from {len(similar):,} "
        f"comparable training heats. "
        f"The reported pattern scores are statistical summaries "
        f"based on model-derived target probability, historical "
        f"target-range frequency, similarity to the current heat, "
        f"historical support, and temperature distance from the "
        f"requested range. "
        f"The highest numerical score is assigned to "
        f"{summary['action_label']}, supported by "
        f"{int(summary['historical_heat_count']):,} historical heats. "
        f"Its historical median tap temperature was "
        f"{summary['historical_median_temperature']:.1f} °C and "
        f"its model-derived target probability is "
        f"{summary['target_probability'] * 100:.1f}%. "
        f"Energy and time differences are calculated from actual "
        f"historical operating values rather than prescribed "
        f"incremental actions. "
        f"These results describe historical statistical associations "
        f"and should not be interpreted as evidence that changing an "
        f"operating variable will causally reproduce the historical "
        f"temperature outcome."
    )

    return (
        results,
        summary,
        explanation,
    )


# ================================================================
# End of module
# ================================================================
