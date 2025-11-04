"""
Plots for hypercube experiments.
"""

from fastsverl.utils import load_data, barcharts, scatter

ns_ds = [(2, 2), (3, 2), (4, 2), (5, 2), (6, 2),
         (2, 3), (3, 3), (4, 3), (5, 3), (6, 3),
         (2, 4), (3, 4), (4, 4), (5, 4), (6, 4),
         (2, 5), (3, 5), (4, 5), (5, 5), (6, 5)]

# Policy
# Set `group` (from W&B) to match the experiment.
# Example: group = "hypercube_policy_{timestamp}"
group = None
def policy(): return {
    "tags": {
        "Characteristic": "model/PolicyCharacteristicModel_exact_loss",
        "Shapley value": "model/PolicyShapley_exact_loss",
    },
    "figure_name": "hypercube_policy",
    "xlabel": "Number of states (log scale)",
    "ylabel": "Training updates to target loss (log scale)",
    "fontsize": 12.5,
    "bar_width": 0.16,
    "ylim": (1, 10_000),
    "xlim": (1, 10_000 * 1.1),
    "groups_data": {
    (2, 2): load_data("FastSVERL_Hypercube", f"2_2_{group}"),
    (3, 2): load_data("FastSVERL_Hypercube", f"3_2_{group}"),
    (4, 2): load_data("FastSVERL_Hypercube", f"4_2_{group}"),
    (5, 2): load_data("FastSVERL_Hypercube", f"5_2_{group}"),
    (6, 2): load_data("FastSVERL_Hypercube", f"6_2_{group}"),

    (2, 3): load_data("FastSVERL_Hypercube", f"2_3_{group}"),
    (3, 3): load_data("FastSVERL_Hypercube", f"3_3_{group}"),
    (4, 3): load_data("FastSVERL_Hypercube", f"4_3_{group}"),
    (5, 3): load_data("FastSVERL_Hypercube", f"5_3_{group}"),
    (6, 3): load_data("FastSVERL_Hypercube", f"6_3_{group}"),

    (2, 4): load_data("FastSVERL_Hypercube", f"2_4_{group}"),
    (3, 4): load_data("FastSVERL_Hypercube", f"3_4_{group}"),
    (4, 4): load_data("FastSVERL_Hypercube", f"4_4_{group}"),
    (5, 4): load_data("FastSVERL_Hypercube", f"5_4_{group}"),
    (6, 4): load_data("FastSVERL_Hypercube", f"6_4_{group}"),

    (2, 5): load_data("FastSVERL_Hypercube", f"2_5_{group}"),
    (3, 5): load_data("FastSVERL_Hypercube", f"3_5_{group}"),
    (4, 5): load_data("FastSVERL_Hypercube", f"4_5_{group}"),
    (5, 5): load_data("FastSVERL_Hypercube", f"5_5_{group}"),
    (6, 5): load_data("FastSVERL_Hypercube", f"6_5_{group}"),
    },
}

# Value
# Set `group` (from W&B) to match the experiment.
# Example: group = "hypercube_value_{timestamp}"
group = None
def value(): return {
    "tags": {
        "Characteristic": "model/ValueCharacteristicModel_exact_loss",
        "Shapley value": "model/ValueShapley_exact_loss",
    },
    "figure_name": "hypercube_value",
    "xlabel": "Number of states (log scale)",
    "ylabel": "Training updates to target loss (log scale)",
    "fontsize": 12.5,
    "bar_width": 0.16,
    "ylim": (1, 300_000),
    "xlim": (1, 10_000 * 1.1),
    "groups_data": {
    (2, 2): load_data("FastSVERL_Hypercube", f"2_2_{group}"),
    (3, 2): load_data("FastSVERL_Hypercube", f"3_2_{group}"),
    (4, 2): load_data("FastSVERL_Hypercube", f"4_2_{group}"),
    (5, 2): load_data("FastSVERL_Hypercube", f"5_2_{group}"),
    (6, 2): load_data("FastSVERL_Hypercube", f"6_2_{group}"),

    (2, 3): load_data("FastSVERL_Hypercube", f"2_3_{group}"),
    (3, 3): load_data("FastSVERL_Hypercube", f"3_3_{group}"),
    (4, 3): load_data("FastSVERL_Hypercube", f"4_3_{group}"),
    (5, 3): load_data("FastSVERL_Hypercube", f"5_3_{group}"),
    (6, 3): load_data("FastSVERL_Hypercube", f"6_3_{group}"),

    (2, 4): load_data("FastSVERL_Hypercube", f"2_4_{group}"),
    (3, 4): load_data("FastSVERL_Hypercube", f"3_4_{group}"),
    (4, 4): load_data("FastSVERL_Hypercube", f"4_4_{group}"),
    (5, 4): load_data("FastSVERL_Hypercube", f"5_4_{group}"),
    (6, 4): load_data("FastSVERL_Hypercube", f"6_4_{group}"),

    (2, 5): load_data("FastSVERL_Hypercube", f"2_5_{group}"),
    (3, 5): load_data("FastSVERL_Hypercube", f"3_5_{group}"),
    (4, 5): load_data("FastSVERL_Hypercube", f"4_5_{group}"),
    (5, 5): load_data("FastSVERL_Hypercube", f"5_5_{group}"),
    (6, 5): load_data("FastSVERL_Hypercube", f"6_5_{group}"),
    },
}

# Loop over each
for plot_spec in [
    policy,
    value,
]:

    # Plot
    barcharts(**plot_spec())

