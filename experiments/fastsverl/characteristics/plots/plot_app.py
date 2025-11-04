"""
Plotting script for appendix characteristic plots.
"""

from fastsverl.utils import load_data, plot

# GWB Policy
# Set `group_name` (from W&B) to match the experiment.
# Example: group_name = "characteristics_dqn_policy_gwb_{timestamp}"
group_name = None

def gwb_policy(): return {
    "tag": "model/PolicyCharacteristicModel_exact_loss",
    "figure_name": "app_gwb_policy_char",
    "title": "Behaviour characteristic",
    "n_batches": (0.8 * 10_000) // 128,
    "groups_data": [
        {
            'data': load_data("FastSVERL_GWB", group_name),
        }
    ],
    }

# GWB Value
# Set `group_name` (from W&B) to match the experiment.
# Example: group_name = "characteristics_dqn_value_gwb_{timestamp}"
group_name = None

def gwb_value(): return {
    "tag": "model/ValueCharacteristicModel_exact_loss",
    "figure_name": "app_gwb_value_char",
    "title": "Prediction characteristic",
    "n_batches": (0.8 * 10_000) // 128,
    "groups_data": [
        {
            'data': load_data("FastSVERL_GWB", group_name),
        }
    ],
}

# GWB Performance
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_w_exact_onpolicy = None  # e.g. "w_exact_onpolicy_characteristics_dqn_performance_gwb_{timestamp}"
group_wo_exact_onpolicy = None  # e.g. "wo_exact_onpolicy_characteristics_dqn_performance_gwb_{timestamp}"
group_w_exact_offpolicy = None  # e.g. "w_exact_offpolicy_characteristics_dqn_performance_gwb_{timestamp}"
group_wo_exact_offpolicy = None  # e.g. "wo_exact_offpolicy_characteristics_dqn_performance_gwb_{timestamp}"

def gwb_performance(): return {
    "tag": "model/PerformanceCharacteristic_exact_loss",
    "figure_name": "app_gwb_perf_char",
    "title": "Outcome characteristic",
    "groups_data": [
        {
            'data': load_data("FastSVERL_GWB", group_w_exact_onpolicy),
            'label': "On-policy, approx behaviour",
            'colour': "C0",
            'linestyle': "--",
        },
        {
            'data': load_data("FastSVERL_GWB", group_wo_exact_onpolicy),
            'label': "On-policy, exact behaviour",
            'colour': "C1",
            'linestyle': "--",
        },
        {
            'data': load_data("FastSVERL_GWB", group_w_exact_offpolicy),
            'label': "Off-policy, approx behaviour",
            'colour': "C0",
            'linestyle': "-",
        },
        {
            'data': load_data("FastSVERL_GWB", group_wo_exact_offpolicy),
            'label': "Off-policy, exact behaviour",
            'colour': "C1",
            'linestyle': "-",
        },
    ],
}

# Mastermind-222 Policy
# Set `group_name` (from W&B) to match the experiment.
# Example: group_name = "222_characteristics_dqn_policy_mastermind_{timestamp}"
group_name = None

def mastermind_222_policy(): return {
    "tag": "model/PolicyCharacteristicModel_exact_loss",
    "figure_name": "app_mastermind_222_policy_char",
    "title": "Behaviour characteristic",
    "ylabel": "MSE",
    "n_batches": (0.8 * 10_000) // 128,
    "groups_data": [
        {
            'data': load_data("FastSVERL_Mastermind", group_name),
        }
    ],
}

# Mastermind-222 Value
# Set `group_name` (from W&B) to match the experiment.
# Example: group_name = "222_characteristics_dqn_value_mastermind_{timestamp}"
group_name = None

def mastermind_222_value(): return {
    "tag": "model/ValueCharacteristicModel_exact_loss",
    "figure_name": "app_mastermind_222_value_char",
    "title": "Prediction characteristic",
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
    "figure_name": "app_mastermind_222_perf_char",
    "title": "Outcome characteristic",
    "ylabel": "MSE",
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

# Mastermind-333 Policy
# Set `group_name` (from W&B) to match the experiment.
# Example: group_name = "333_characteristics_dqn_policy_mastermind_{timestamp}"
group_name = None

def mastermind_333_policy(): return {
    "tag": "model/PolicyCharacteristicModel_exact_loss",
    "figure_name": "app_mastermind_333_policy_char",
    "title": "Behaviour characteristic",
    "n_batches": (0.8 * 10_000) // 128,
    "groups_data": [
        {
            'data': load_data("FastSVERL_Mastermind", group_name),
        }
    ],
}

# Mastermind-333 Value
# Set `group_name` (from W&B) to match the experiment.
# Example: group_name = "333_characteristics_dqn_value_mastermind_{timestamp}"
group_name = None

def mastermind_333_value(): return {
    "tag": "model/ValueCharacteristicModel_exact_loss",
    "figure_name": "app_mastermind_333_value_char",
    "title": "Prediction characteristic",
    "n_batches": (0.8 * 10_000) // 128,
    "xticks": (0, 40_000, 80_000, 120_000),
    "groups_data": [
        {
            'data': load_data("FastSVERL_Mastermind", group_name),
        }
    ],
}

# Mastermind Large Scale Behaviour
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_443 = None  # e.g. "443_characteristics_dqn_policy_mastermind_{timestamp}"
group_453 = None  # e.g. "453_characteristics_dqn_policy_mastermind_{timestamp}"
group_463 = None  # e.g. "463_characteristics_dqn_policy_mastermind_{timestamp}"

def mastermind_scale_policy(): 
    return {
    "tag": "model/PolicyCharacteristicModel_epoch_loss",
    "figure_name": "app_large_policy_char",
    "title": "Characteristic",
    "ylabel": "Training Loss",
    "groups_data": [
        {
            "data": load_data("FastSVERL_Mastermind", group_443),
            "label": "Mastermind-443",
            "colour": "C0",
        },
        {
            "data": load_data("FastSVERL_Mastermind", group_453),
            "label": "Mastermind-453",
            "colour": "C1",
        },
        {
            "data": load_data("FastSVERL_Mastermind", group_463),
            "label": "Mastermind-463",
            "colour": "C2",
        },
        
    ],
}

# Mastermind Large Scale Prediction
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_443 = None  # e.g. "443_characteristics_dqn_value_mastermind_{timestamp}"
group_453 = None  # e.g. "453_characteristics_dqn_value_mastermind_{timestamp}"
group_463 = None  # e.g. "463_characteristics_dqn_value_mastermind_{timestamp}"

def mastermind_scale_value(): 
    return {
    "tag": "model/ValueCharacteristicModel_epoch_loss",
    "figure_name": "app_large_value_char",
    "title": "Characteristic",
    "xticks": (0, 5_000, 10_000, 15_000, 20_000),
    "groups_data": [
        {
            "data": load_data("FastSVERL_Mastermind", group_443),
            "label": "Mastermind-443",
            "colour": "C0",
        },
        {
            "data": load_data("FastSVERL_Mastermind", group_453),
            "label": "Mastermind-453",
            "colour": "C1",
        },
        {
            "data": load_data("FastSVERL_Mastermind", group_463),
            "label": "Mastermind-463",
            "colour": "C2",
        },
        
    ],
}

# Loop over each
for plot_spec_func in [
    gwb_policy,
    gwb_value,
    gwb_performance,
    mastermind_222_policy,
    mastermind_222_value,
    mastermind_222_performance,
    mastermind_333_policy,
    mastermind_333_value,
    mastermind_scale_policy,
    mastermind_scale_value,
]:

    # Plot
    plot_spec = plot_spec_func()

    # Common plot settings
    plot_spec["xlabel"] = "Training updates"
    plot_spec["legend_loc"] = "upper right"
    plot_spec["fontsize"] = 16

    plot(**plot_spec)
