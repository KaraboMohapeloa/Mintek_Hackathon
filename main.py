"""
main.py

EAF Process Optimisation Proof-of-Concept
=========================================

White-box Bayesian-like tap-temperature decision-support model.

Pipeline
--------
1. Load EAF production data
2. Inspect the dataset
3. Validate target
4. Prepare modelling features
5. Prepare X/y
6. Chronological train/calibration/test split
7. Train transparent Ridge model
8. Report training diagnostics
9. Evaluate final test prediction
10. Evaluate raw predictive uncertainty
11. Evaluate calibrated predictive uncertainty
12. Evaluate target-range probability
13. Report global feature effects
14. Select current heat
15. Predict current heat
16. Explain current heat
17. Evaluate illustrative scenarios
18. Save outputs
19. Print final summary

Important
---------
This is a decision-support POC.

It does NOT:

    - directly control the EAF
    - issue PLC/control commands
    - replace the furnace operator
    - provide a physically complete EAF model
    - establish causal effects of operating actions

The probability outputs are model-derived probabilities after
uncertainty calibration. They should not be treated as guaranteed
probabilities or as proof of causal furnace behaviour.
"""


from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from data_utils import (
    load_data,
    inspect_data,
)

from features import (
    prepare_features,
    get_feature_columns,
    get_target_column,
)

from model import (
    train_model,
    predict,
    predict_distribution,
    probability_in_range,
    evaluate_point_prediction,
    evaluate_prediction_intervals,
    feature_importance,
    explain_prediction,
    save_feature_importance,
)

from decision import (
    generate_decision_support,
)


# ================================================================
# Configuration
# ================================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

DATA_FILE = (
    DATA_DIR
    / "heat_table.csv"
)


# ================================================================
# Target
# ================================================================

TARGET_COLUMN = "temp_tap"

TARGET_MIN = 1600.0
TARGET_MAX = 1630.0


# ================================================================
# Model
# ================================================================

RANDOM_STATE = 42

ALPHA = 10.0

# Retained for compatibility with the existing interface.
N_ESTIMATORS = 300
MAX_DEPTH = 12
MIN_SAMPLES_LEAF = 5


# ================================================================
# Split
# ================================================================

TRAIN_PROPORTION = 0.70
CALIBRATION_PROPORTION = 0.15
TEST_PROPORTION = 0.15


# ================================================================
# Decision weights
# ================================================================

WEIGHT_TEMPERATURE = 1.0
WEIGHT_ENERGY = 0.10
WEIGHT_TIME = 0.05


# ================================================================
# Explainability
# ================================================================

TOP_N_FEATURES = 15
TOP_N_LOCAL_FEATURES = 10


# ================================================================
# Utilities
# ================================================================

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print()


def print_subsection(title):
    print()
    print("-" * 60)
    print(title)
    print("-" * 60)


def ensure_output_directory():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def print_dataset_summary(df):
    print(
        f"Dataset shape: "
        f"{df.shape[0]:,} rows × "
        f"{df.shape[1]:,} columns"
    )

    print(
        f"Missing values: "
        f"{df.isna().sum().sum():,}"
    )

    print(
        f"Duplicate rows: "
        f"{df.duplicated().sum():,}"
    )


def print_prediction_metrics(metrics):
    for metric, value in metrics.items():

        if isinstance(
            value,
            (float, int, np.floating, np.integer),
        ):
            print(
                f"{metric}: "
                f"{float(value):.4f}"
            )

        else:
            print(
                f"{metric}: "
                f"{value}"
            )


def print_action_results(results):
    if results.empty:
        print(
            "No action scenarios were evaluated."
        )
        return

    display_columns = [
        "action_label",
        "temperature_mean",
        "temperature_std",
        "temperature_p10",
        "temperature_p90",
        "target_probability",
        "additional_energy_mwh",
        "additional_time_min",
        "temperature_penalty",
        "score",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in results.columns
    ]

    output = results[
        available_columns
    ].copy()

    for column in [
        "temperature_mean",
        "temperature_std",
        "temperature_p10",
        "temperature_p90",
    ]:
        if column in output.columns:
            output[column] = (
                output[column]
                .round(1)
            )

    if "target_probability" in output.columns:
        output[
            "target_probability"
        ] = (
            output[
                "target_probability"
            ]
            *
            100
        ).round(1)

    if "additional_energy_mwh" in output.columns:
        output[
            "additional_energy_mwh"
        ] = (
            output[
                "additional_energy_mwh"
            ]
            .round(3)
        )

    if "additional_time_min" in output.columns:
        output[
            "additional_time_min"
        ] = (
            output[
                "additional_time_min"
            ]
            .round(1)
        )

    if "temperature_penalty" in output.columns:
        output[
            "temperature_penalty"
        ] = (
            output[
                "temperature_penalty"
            ]
            .round(4)
        )

    if "score" in output.columns:
        output[
            "score"
        ] = (
            output[
                "score"
            ]
            .round(4)
        )

    print(
        output.to_string(
            index=False
        )
    )


def verify_prediction_alignment(
    y,
    predictions,
    expected_index=None,
):
    """
    Explicitly verify prediction/target alignment.

    This function is intentionally strict because a prediction
    metric is meaningless if predictions and targets are in
    different orders.
    """

    y_series = pd.Series(
        y
    )

    predictions = np.asarray(
        predictions,
        dtype=float,
    ).reshape(-1)

    if len(y_series) != len(predictions):
        raise ValueError(
            "CRITICAL ALIGNMENT ERROR: "
            f"{len(y_series)} targets versus "
            f"{len(predictions)} predictions."
        )

    if not np.all(
        np.isfinite(
            predictions
        )
    ):
        raise ValueError(
            "Predictions contain non-finite values."
        )

    if expected_index is not None:

        if not y_series.index.equals(
            expected_index
        ):
            raise ValueError(
                "CRITICAL INDEX ERROR: "
                "target index does not match expected test index."
            )


def calculate_observed_target_probability(
    y,
    lower,
    upper,
):
    """
    Calculate the observed proportion of targets inside a range.
    """

    y = np.asarray(
        y,
        dtype=float,
    )

    return float(
        np.mean(
            (
                y >= lower
            )
            &
            (
                y <= upper
            )
        )
    )


# ================================================================
# Main
# ================================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )

    ensure_output_directory()

    # ============================================================
    # 1. Load
    # ============================================================

    print_section(
        "1. LOAD EAF DATA"
    )

    print(
        f"Data file:\n{DATA_FILE}"
    )

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"\nCould not find the EAF dataset:\n"
            f"{DATA_FILE}"
        )

    df = load_data(
        DATA_FILE
    )

    print(df.columns.tolist())
    print(df.head().to_string())

    print_dataset_summary(
        df
    )

    # ============================================================
    # 2. Inspect
    # ============================================================

    print_section(
        "2. DATASET INSPECTION"
    )

    inspection = inspect_data(
        df
    )

    print(
        inspection
    )

    # ============================================================
    # 3. Target
    # ============================================================

    print_section(
        "3. TARGET VALIDATION"
    )

    target_column = get_target_column(
        df,
        preferred_target=TARGET_COLUMN,
    )

    print(
        f"Target column: "
        f"{target_column}"
    )

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            f"was not found."
        )

    target = pd.to_numeric(
        df[target_column],
        errors="coerce",
    )

    valid_target = (
        target.notna()
        &
        np.isfinite(target)
    )

    print(
        f"Valid target observations: "
        f"{valid_target.sum():,}"
    )

    print(
        f"Missing target observations: "
        f"{(~valid_target).sum():,}"
    )

    if valid_target.sum() < 500:
        raise ValueError(
            "Too few valid target observations."
        )

    # ============================================================
    # 4. Feature engineering
    # ============================================================

    print_section(
        "4. FEATURE ENGINEERING"
    )

    modelling_df = prepare_features(
        df,
        target_column=target_column,
    )

    feature_columns = get_feature_columns(
        modelling_df,
        target_column=target_column,
    )

    print(
        f"Number of modelling features: "
        f"{len(feature_columns)}"
    )

    print_subsection(
        "Selected features"
    )

    for feature in feature_columns:
        print(
            f"  - {feature}"
        )

    # ============================================================
    # 5. Prepare modelling data
    # ============================================================

    print_section(
        "5. PREPARE MODELLING DATA"
    )

    X = modelling_df[
        feature_columns
    ].copy()

    y = pd.to_numeric(
        modelling_df[
            target_column
        ],
        errors="coerce",
    )

    valid_mask = (
        y.notna()
        &
        np.isfinite(y)
    )

    X = X.loc[
        valid_mask
    ].copy()

    y = y.loc[
        valid_mask
    ].copy()

    # ------------------------------------------------------------
    # Preserve chronological ordering exactly as supplied by the
    # input dataset.
    # ------------------------------------------------------------

    print(
        f"Observations available "
        f"for modelling: {len(X):,}"
    )

    print(
        f"Features: {X.shape[1]}"
    )

    print(
        f"Target mean: "
        f"{y.mean():.2f} °C"
    )

    print(
        f"Target std: "
        f"{y.std():.2f} °C"
    )

    print(
        f"Target minimum: "
        f"{y.min():.2f} °C"
    )

    print(
        f"Target maximum: "
        f"{y.max():.2f} °C"
    )

    # ============================================================
    # 6. Temporal train/calibration/test split
    # ============================================================

    print_section(
        "6. TEMPORAL TRAIN / CALIBRATION / TEST SPLIT"
    )

    total = len(X)

    train_end = int(
        total
        *
        TRAIN_PROPORTION
    )

    calibration_end = (
        train_end
        +
        int(
            total
            *
            CALIBRATION_PROPORTION
        )
    )

    # Guarantee every set is non-empty.
    train_end = max(
        1,
        min(
            train_end,
            total - 2,
        ),
    )

    calibration_end = max(
        train_end + 1,
        min(
            calibration_end,
            total - 1,
        ),
    )

    X_train = X.iloc[
        :train_end
    ].copy()

    X_calibration = X.iloc[
        train_end:calibration_end
    ].copy()

    X_test = X.iloc[
        calibration_end:
    ].copy()

    y_train = y.iloc[
        :train_end
    ].copy()

    y_calibration = y.iloc[
        train_end:calibration_end
    ].copy()

    y_test = y.iloc[
        calibration_end:
    ].copy()

    print(
        f"Training observations: "
        f"{len(X_train):,}"
    )

    print(
        f"Calibration observations: "
        f"{len(X_calibration):,}"
    )

    print(
        f"Testing observations: "
        f"{len(X_test):,}"
    )

    print(
        f"Training proportion: "
        f"{len(X_train) / total:.1%}"
    )

    print(
        f"Calibration proportion: "
        f"{len(X_calibration) / total:.1%}"
    )

    print(
        f"Testing proportion: "
        f"{len(X_test) / total:.1%}"
    )

    # ============================================================
    # 7. Train
    # ============================================================

    print_section(
        "7. WHITE-BOX BAYESIAN-LIKE MODEL"
    )

    print(
        "Training transparent probabilistic regression..."
    )

    print(
        "Model: StandardScaler + Ridge regression"
    )

    print(
        f"Regularisation alpha: "
        f"{ALPHA}"
    )

    model = train_model(
        X_train=X_train,
        y_train=y_train,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        alpha=ALPHA,
    )

    print(
        "Model training complete."
    )

    # ------------------------------------------------------------
    # Training diagnostics
    # ------------------------------------------------------------

    training_predictions = predict(
        model,
        X_train,
    )

    training_metrics = (
        evaluate_point_prediction(
            y_train,
            training_predictions,
        )
    )

    print_subsection(
        "Model summary"
    )

    print(
        "Model type: "
        "White-box Bayesian-like Ridge regression"
    )

    print(
        f"Observations: "
        f"{model.n_observations:,}"
    )

    print(
        f"Features: "
        f"{model.n_features}"
    )

    print(
        f"Residual standard deviation: "
        f"{model.residual_std:.3f} °C"
    )

    print(
        f"Training MAE: "
        f"{training_metrics['MAE']:.4f} °C"
    )

    print(
        f"Training RMSE: "
        f"{training_metrics['RMSE']:.4f} °C"
    )

    # ============================================================
    # 8. Calibration
    # ============================================================

    print_section(
        "8. PREDICTIVE UNCERTAINTY CALIBRATION"
    )

    print(
        "Calibrating predictive uncertainty using the "
        "chronological calibration set..."
    )

    calibration_metrics = model.calibrate(
        X_calibration,
        y_calibration,
    )

    print_subsection(
        "Calibration results"
    )

    print(
        f"Calibration observations: "
        f"{calibration_metrics['calibration_observations']:,}"
    )

    print(
        f"Calibration MAE: "
        f"{calibration_metrics['calibration_mae']:.4f} °C"
    )

    print(
        f"Calibration RMSE: "
        f"{calibration_metrics['calibration_rmse']:.4f} °C"
    )

    print(
        f"Raw mean predictive std: "
        f"{calibration_metrics['raw_mean_std']:.3f} °C"
    )

    print(
        f"Calibrated mean predictive std: "
        f"{calibration_metrics['calibrated_mean_std']:.3f} °C"
    )

    print(
        f"Calibration scale: "
        f"{calibration_metrics['calibration_scale']:.4f}"
    )

    print(
        f"Calibrated P10-P90 coverage on calibration set: "
        f"{calibration_metrics['P10_P90_coverage']:.4f}"
    )

    print(
        f"Calibrated P10-P90 average width: "
        f"{calibration_metrics['P10_P90_average_width']:.4f} °C"
    )

    calibration_file = (
        OUTPUT_DIR
        /
        "calibration_results.csv"
    )

    pd.DataFrame(
        [calibration_metrics]
    ).to_csv(
        calibration_file,
        index=False,
    )

    print()
    print(
        f"Calibration results saved to:\n"
        f"{calibration_file}"
    )

    # ============================================================
    # 9. Final test prediction
    # ============================================================

    print_section(
        "9. FINAL TEST SET PREDICTION"
    )

    test_predictions = predict(
        model,
        X_test,
    )

    verify_prediction_alignment(
        y_test,
        test_predictions,
        expected_index=X_test.index,
    )

    test_metrics = (
        evaluate_point_prediction(
            y_test,
            test_predictions,
        )
    )

    print_prediction_metrics(
        test_metrics
    )

    # ------------------------------------------------------------
    # Explicit sanity check.
    # ------------------------------------------------------------

    errors = (
        y_test.to_numpy(dtype=float)
        -
        np.asarray(
            test_predictions,
            dtype=float,
        )
    )

    direct_rmse = float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )

    if not np.isclose(
        direct_rmse,
        test_metrics["RMSE"],
        rtol=1e-10,
        atol=1e-10,
    ):
        raise RuntimeError(
            "RMSE consistency check failed."
        )

    print()
    print(
        f"RMSE independent verification: "
        f"{direct_rmse:.4f} °C"
    )

    # ============================================================
    # 10. Predictive uncertainty
    # ============================================================

    print_section(
        "10. PREDICTIVE UNCERTAINTY"
    )

    print(
        "Estimating calibrated Gaussian predictive distributions..."
    )

    prediction_distribution = (
        predict_distribution(
            model,
            X_test,
        )
    )

    interval_metrics = (
        evaluate_prediction_intervals(
            y_test,
            prediction_distribution,
        )
    )

    print_subsection(
        "P10-P90 predictive interval performance"
    )

    print_prediction_metrics(
        interval_metrics
    )

    print()
    print(
        "The predictive distribution combines:"
    )

    print(
        "  1. Estimated process/residual uncertainty"
    )

    print(
        "  2. Approximate uncertainty in fitted coefficients"
    )

    print(
        "  3. A scale calibration learned from a separate "
        "chronological calibration set"
    )

    print()
    print(
        "The final test-set interval is evaluated only after "
        "model fitting and uncertainty calibration."
    )

    # ============================================================
    # 11. Target-range probability
    # ============================================================

    print_section(
        "11. TARGET-RANGE PROBABILITY"
    )

    print(
        f"Target temperature range: "
        f"{TARGET_MIN:.0f}–"
        f"{TARGET_MAX:.0f} °C"
    )

    print()
    print(
        "Calculating model-derived probability from the "
        "calibrated Gaussian predictive distribution..."
    )

    test_probabilities = (
        probability_in_range(
            model,
            X_test,
            TARGET_MIN,
            TARGET_MAX,
        )
    )

    verify_prediction_alignment(
        y_test,
        test_probabilities,
        expected_index=X_test.index,
    )

    observed_rate = (
        calculate_observed_target_probability(
            y_test,
            TARGET_MIN,
            TARGET_MAX,
        )
    )

    mean_probability = float(
        np.mean(
            test_probabilities
        )
    )

    print(
        f"Observed final test-set in-range rate: "
        f"{observed_rate * 100:.2f}%"
    )

    print(
        f"Mean model-derived probability: "
        f"{mean_probability * 100:.2f}%"
    )

    print()
    print(
        "These probabilities are generated by the calibrated "
        "Gaussian predictive model. Calibration improves "
        "historical agreement but does not make the probabilities "
        "guaranteed or causal."
    )

    # ============================================================
    # 12. Global feature effects
    # ============================================================

    print_section(
        "12. GLOBAL FEATURE EFFECTS"
    )

    importance = feature_importance(
        model
    )

    print(
        importance
        .head(
            TOP_N_FEATURES
        )
        .to_string(
            index=False
        )
    )

    importance_file = (
        OUTPUT_DIR
        /
        "feature_importance.csv"
    )

    save_feature_importance(
        model,
        importance_file,
    )

    print()
    print(
        f"Feature effects saved to:\n"
        f"{importance_file}"
    )

    # ============================================================
    # 13. Current heat
    # ============================================================

    print_section(
        "13. CURRENT HEAT"
    )

    current_index = X_test.index[-1]

    current_state = (
        X_test
        .loc[current_index]
        .to_dict()
    )

    print(
        f"Selected heat/index: "
        f"{current_index}"
    )

    print(
        f"Number of state variables: "
        f"{len(current_state)}"
    )

    # ============================================================
    # 14. Current prediction
    # ============================================================

    print_section(
        "14. CURRENT HEAT PREDICTION"
    )

    current_X = pd.DataFrame(
        [current_state]
    )

    current_distribution = (
        predict_distribution(
            model,
            current_X,
        )
    )

    current_prediction = (
        current_distribution.iloc[0]
    )

    current_probability = float(
        probability_in_range(
            model,
            current_X,
            TARGET_MIN,
            TARGET_MAX,
        )[0]
    )

    print(
        f"Predicted tap temperature: "
        f"{current_prediction['temperature_mean']:.1f} °C"
    )

    print(
        f"Median prediction: "
        f"{current_prediction['temperature_median']:.1f} °C"
    )

    print(
        f"Predictive standard deviation: "
        f"{current_prediction['temperature_std']:.1f} °C"
    )

    print(
        f"P10: "
        f"{current_prediction['temperature_p10']:.1f} °C"
    )

    print(
        f"P90: "
        f"{current_prediction['temperature_p90']:.1f} °C"
    )

    print(
        f"Probability of "
        f"{TARGET_MIN:.0f}–"
        f"{TARGET_MAX:.0f} °C: "
        f"{current_probability * 100:.1f}%"
    )

    # ============================================================
    # 15. Local explanation
    # ============================================================

    print_section(
        "15. CURRENT HEAT EXPLANATION"
    )

    local_explanation = (
        explain_prediction(
            model,
            current_state,
            top_n=TOP_N_LOCAL_FEATURES,
        )
    )

    print(
        "Largest additive model contributions:"
    )

    print()

    print(
        local_explanation.to_string(
            index=False
        )
    )

    explanation_file = (
        OUTPUT_DIR
        /
        "current_heat_explanation.csv"
    )

    local_explanation.to_csv(
        explanation_file,
        index=False,
    )

    print()
    print(
        f"Local explanation saved to:\n"
        f"{explanation_file}"
    )

    # ============================================================
    # 16. Scenario evaluation
    # ============================================================

    print_section(
        "16. ACTION SCENARIO EVALUATION"
    )

    print(
        "Evaluating candidate POC operating scenarios..."
    )

    results, summary, explanation = (
        generate_decision_support(
            model=model,
            state=current_state,

            target_min=TARGET_MIN,
            target_max=TARGET_MAX,

            weight_temperature=
                WEIGHT_TEMPERATURE,

            weight_energy=
                WEIGHT_ENERGY,

            weight_time=
                WEIGHT_TIME,
        )
    )

    if results is None:
        raise RuntimeError(
            "Decision-support function returned None for results."
        )

    if summary is None:
        raise RuntimeError(
            "Decision-support function returned None for summary."
        )

    if explanation is None:
        raise RuntimeError(
            "Decision-support function returned None for explanation."
        )

    print_action_results(
        results
    )

    # ============================================================
    # 17. Decision-support summary
    # ============================================================

    print_section(
        "17. DECISION-SUPPORT SUMMARY"
    )

    print(
        f"Target temperature range: "
        f"{TARGET_MIN:.0f}–"
        f"{TARGET_MAX:.0f} °C"
    )

    print()

    print(
        f"Highest numerical historical-pattern score: "
        f"{summary['action_label']}"
    )


    print(
        f"Predicted tap temperature: "
        f"{summary['temperature_mean']:.1f} °C"
    )

    print(
        f"Predictive interval: "
        f"{summary['temperature_p10']:.1f}–"
        f"{summary['temperature_p90']:.1f} °C"
    )

    print(
        f"Model-derived target probability: "
        f"{summary['target_probability'] * 100:.1f}%"
    )

    print(
        f"Estimated additional energy: "
        f"{summary['additional_energy_mwh']:.3f} MWh"
    )

    print(
        f"Estimated additional process time: "
        f"{summary['additional_time_min']:.1f} min"
    )

    print(
        f"Temperature penalty: "
        f"{summary['temperature_penalty']:.4f}"
    )

    print(
        f"POC scenario score: "
        f"{summary['score']:.4f}"
    )

    print()
    print(
        "Explanation:"
    )

    print(
        explanation
    )

    # ============================================================
    # 18. Save results
    # ============================================================

    print_section(
        "18. SAVE RESULTS"
    )

    # ------------------------------------------------------------
    # Scenario results
    # ------------------------------------------------------------

    results_file = (
        OUTPUT_DIR
        /
        "action_scenarios.csv"
    )

    results.to_csv(
        results_file,
        index=False,
    )

    print(
        f"Action scenario results saved to:\n"
        f"{results_file}"
    )

    # ------------------------------------------------------------
    # Test predictions
    # ------------------------------------------------------------

    test_output = pd.DataFrame(
        {
            "actual_temperature":
                y_test.to_numpy(
                    dtype=float
                ),

            "predicted_temperature":
                np.asarray(
                    test_predictions,
                    dtype=float,
                ),

            "prediction_p10":
                prediction_distribution[
                    "temperature_p10"
                ].to_numpy(
                    dtype=float
                ),

            "prediction_median":
                prediction_distribution[
                    "temperature_median"
                ].to_numpy(
                    dtype=float
                ),

            "prediction_p90":
                prediction_distribution[
                    "temperature_p90"
                ].to_numpy(
                    dtype=float
                ),

            "prediction_std":
                prediction_distribution[
                    "temperature_std"
                ].to_numpy(
                    dtype=float
                ),

            "target_probability":
                np.asarray(
                    test_probabilities,
                    dtype=float,
                ),

            "prediction_error":
                y_test.to_numpy(
                    dtype=float
                )
                -
                np.asarray(
                    test_predictions,
                    dtype=float,
                ),

            "absolute_error":
                np.abs(
                    y_test.to_numpy(
                        dtype=float
                    )
                    -
                    np.asarray(
                        test_predictions,
                        dtype=float,
                    )
                ),
        },
        index=y_test.index,
    )

    predictions_file = (
        OUTPUT_DIR
        /
        "test_predictions.csv"
    )

    test_output.to_csv(
        predictions_file
    )

    print(
        f"Test predictions saved to:\n"
        f"{predictions_file}"
    )

    # ------------------------------------------------------------
    # Save test metrics
    # ------------------------------------------------------------

    metrics_file = (
        OUTPUT_DIR
        /
        "test_metrics.csv"
    )

    pd.DataFrame(
        [
            test_metrics
        ]
    ).to_csv(
        metrics_file,
        index=False,
    )

    print(
        f"Test metrics saved to:\n"
        f"{metrics_file}"
    )

    # ============================================================
    # 19. Final summary
    # ============================================================

    print_section(
        "POC COMPLETE"
    )

    print(
        "The EAF white-box Bayesian-like "
        "decision-support POC has completed successfully."
    )

    print()

    print(
        "The pipeline demonstrated:"
    )

    print(
        "  1. Historical EAF data processing"
    )

    print(
        "  2. Chronological train/calibration/test validation"
    )

    print(
        "  3. Transparent tap-temperature regression"
    )

    print(
        "  4. Gaussian predictive uncertainty"
    )

    print(
        "  5. Separate uncertainty calibration"
    )

    print(
        "  6. Model-derived target-range probability"
    )

    print(
        "  7. Coefficient-based global explainability"
    )

    print(
        "  8. Heat-level additive explanation"
    )

    print(
        "  9. Illustrative scenario simulation"
    )

    print(
        " 10. Transparent numerical scenario scoring"
    )

    print()

    print(
        "Final test metrics:"
    )

    print(
        f"  MAE:  "
        f"{test_metrics['MAE']:.4f} °C"
    )

    print(
        f"  RMSE: "
        f"{test_metrics['RMSE']:.4f} °C"
    )

    print(
        f"  Mean error: "
        f"{test_metrics['Mean_Error']:.4f} °C"
    )

    print(
        f"  P10-P90 coverage: "
        f"{interval_metrics['P10_P90_coverage'] * 100:.2f}%"
    )

    print(
        f"  P10-P90 average width: "
        f"{interval_metrics['P10_P90_average_width']:.2f} °C"
    )

    print()

    print(
        "Output files are available in:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This is a decision-support POC."
    )

    print(
        "The model does not issue furnace-control commands."
    )

    print(
        "Historical operating patterns are derived from observed "
        "training heats. They represent statistical associations, "
        "not proven causal effects of operating actions, and require "
        "process validation before operational use."
    )


    print(
        "The target-range probabilities are calibrated "
        "model-derived probabilities and are not guarantees."
    )


# ================================================================
# Entry point
# ================================================================

if __name__ == "__main__":
    main()
