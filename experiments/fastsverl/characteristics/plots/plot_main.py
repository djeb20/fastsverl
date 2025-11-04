"""
Plotting script for FastSVERL characteristics' learning curves.
"""

from fastsverl.utils import load_data, plot

# Mastermind-222 Policy
# Set `group_name` (from W&B) to match the experiment.
# Example: group_name = "222_characteristics_dqn_policy_mastermind_{timestamp}"
group_name = None

def mastermind_222_policy(): return {
    "tag": "model/PolicyCharacteristicModel_exact_loss",
    "figure_name": "main_mastermind_222_policy_char",
    "ylabel": "MSE",
    "n_batches": (0.8 * 10_000) // 128,
    "groups_data": [
        {
            'data': load_data("FastSVERL_Mastermind", group_name),
        }
    ],
}

# Mastermind-222 Performance
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_w_exact_onpolicy = None  # e.g. "222_w_exact_onpolicy_characteristics_dqn_performance_mastermind_{timestamp}"
group_wo_exact_onpolicy = None  # e.g. "222_wo_exact_onpolicy_characteristics_dqn_performance_mastermind_{timestamp}"
group_w_exact_offpolicy = None  # e.g. "222_w_exact_offpolicy_characteristics_dqn_performance_mastermind_{timestamp}"
group_wo_exact_offpolicy = None  # e.g. "222_wo_exact_offpolicy_characteristics_dqn_performance_mastermind_{timestamp}"

def mastermind_222_performance(): return {
    "tag": "model/PerformanceCharacteristic_exact_loss",
    "xlim": (-5000, 220000),
    "figure_name": "main_mastermind_222_perf_char",
    "groups_data": [
        {
            'data': load_data("FastSVERL_Mastermind", group_w_exact_onpolicy),
            'label': "On-policy, approx behaviour",
            'colour': "C0",
            'linestyle': "--",
        },
        {
            'data': load_data("FastSVERL_Mastermind", group_wo_exact_onpolicy),
            'label': "On-policy, exact behaviour",
            'colour': "C1",
            'linestyle': "--",
        },
        {
            'data': load_data("FastSVERL_Mastermind", group_w_exact_offpolicy),
            'label': "Off-policy, approx behaviour",
            'colour': "C0",
            'linestyle': "-",
        },
        {
            'data': load_data("FastSVERL_Mastermind", group_wo_exact_offpolicy),
            'label': "Off-policy, exact behaviour",
            'colour': "C1",
            'linestyle': "-",
        },
    ],
}

# Loop over each
for plot_spec_func in [
    mastermind_222_policy,
    mastermind_222_performance,
]:

    # Plot
    plot_spec = plot_spec_func()

    # Common plot settings
    plot_spec["xlabel"] = "Training updates"
    plot_spec["legend_loc"] = "upper right"
    plot_spec["fontsize"] = 16

    plot(**plot_spec)