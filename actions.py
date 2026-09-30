"""
actions.py

Candidate operating actions and simple scenario simulation for
the EAF decision-support POC.

IMPORTANT
---------
The available POC dataset is primarily heat-level data.

Therefore these action transformations are hypothetical scenario
perturbations. They are NOT a physical EAF simulator and should
not be interpreted as validated control laws.
"""

import copy


# ================================================================
# Candidate actions
# ================================================================

ACTIONS = [
    "maintain",
    "increase_power",
    "increase_o2",
    "increase_carbon",
    "prepare_tap",
]


ACTION_LABELS = {
    "maintain":
        "Maintain current operation",

    "increase_power":
        "Increase electrical power",

    "increase_o2":
        "Increase oxygen input",

    "increase_carbon":
        "Increase carbon input",

    "prepare_tap":
        "Prepare for tap",
}


# ================================================================
# Scenario magnitudes
# ================================================================

# These values are intentionally small.

POWER_INCREASE = 0.10
O2_INCREASE = 0.10
CARBON_INCREASE = 0.10


# ================================================================
# Labels
# ================================================================

def get_action_label(action):
    """
    Return human-readable action name.
    """

    return ACTION_LABELS.get(
        action,
        str(action),
    )


# ================================================================
# Helpers
# ================================================================

def _multiply_if_present(
    state,
    column,
    factor,
):
    """
    Multiply a state variable by a factor if present.
    """

    if column in state:

        value = state[column]

        if value is not None:

            try:
                state[column] = (
                    float(value)
                    * factor
                )

            except (
                TypeError,
                ValueError,
            ):
                pass


def _add_if_present(
    state,
    column,
    amount,
):
    """
    Add an amount to a state variable if present.
    """

    if column in state:

        value = state[column]

        if value is not None:

            try:
                state[column] = (
                    float(value)
                    + amount
                )

            except (
                TypeError,
                ValueError,
            ):
                pass


# ================================================================
# Simulate action
# ================================================================

def simulate_action(
    state,
    action,
):
    """
    Create a hypothetical state representing an action.

    The original state is never modified.

    Parameters
    ----------
    state : dict-like
        Current process state.

    action : str
        Candidate action.

    Returns
    -------
    dict
        Hypothetical state.
    """

    if action not in ACTIONS:

        raise ValueError(
            f"Unknown action: {action}. "
            f"Available actions: {ACTIONS}"
        )

    simulated = copy.deepcopy(
        dict(state)
    )

    # ------------------------------------------------------------
    # Maintain
    # ------------------------------------------------------------

    if action == "maintain":
        return simulated

    # ------------------------------------------------------------
    # Increase power
    # ------------------------------------------------------------

    if action == "increase_power":

        if "mean_mw" in simulated:

            _multiply_if_present(
                simulated,
                "mean_mw",
                1.0 + POWER_INCREASE,
            )

        elif "power_mw" in simulated:

            _multiply_if_present(
                simulated,
                "power_mw",
                1.0 + POWER_INCREASE,
            )

        elif "energy_mwh" in simulated:

            _multiply_if_present(
                simulated,
                "energy_mwh",
                1.0 + POWER_INCREASE,
            )

        return simulated

    # ------------------------------------------------------------
    # Increase oxygen
    # ------------------------------------------------------------

    if action == "increase_o2":

        if "o2_total" in simulated:

            _multiply_if_present(
                simulated,
                "o2_total",
                1.0 + O2_INCREASE,
            )

        if "o2_flow_mean" in simulated:

            _multiply_if_present(
                simulated,
                "o2_flow_mean",
                1.0 + O2_INCREASE,
            )

        return simulated

    # ------------------------------------------------------------
    # Increase carbon
    # ------------------------------------------------------------

    if action == "increase_carbon":

        if "carbon_total" in simulated:

            _multiply_if_present(
                simulated,
                "carbon_total",
                1.0 + CARBON_INCREASE,
            )

        if "carbon_flow_mean" in simulated:

            _multiply_if_present(
                simulated,
                "carbon_flow_mean",
                1.0 + CARBON_INCREASE,
            )

        return simulated

    # ------------------------------------------------------------
    # Prepare for tap
    # ------------------------------------------------------------

    if action == "prepare_tap":

        # No major process variable is artificially increased.
        #
        # The scenario is retained because it represents an
        # operational decision rather than a process-input
        # intervention.

        return simulated

    return simulated


# ================================================================
# Additional energy estimate
# ================================================================

def estimate_additional_energy(
    state,
    action,
):
    """
    Estimate a simple additional-energy cost for a scenario.

    These are POC scenario assumptions and should later be replaced
    with estimates learned from time-series process data.
    """

    if action == "maintain":
        return 0.44

    if action == "increase_power":
        return 0.32

    if action == "increase_o2":
        return 0.35

    if action == "increase_carbon":
        return 0.40

    if action == "prepare_tap":
        return 0.20

    raise ValueError(
        f"Unknown action: {action}"
    )


# ================================================================
# Additional time estimate
# ================================================================

def estimate_additional_time(
    state,
    action,
):
    """
    Estimate a simple additional process-time cost.

    These are POC assumptions and should later be learned from
    historical time-series data.
    """

    if action == "maintain":
        return 8.0

    if action == "increase_power":
        return 5.0

    if action == "increase_o2":
        return 6.0

    if action == "increase_carbon":
        return 7.0

    if action == "prepare_tap":
        return 2.0

    raise ValueError(
        f"Unknown action: {action}"
    )
