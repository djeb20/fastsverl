"""
Plot FastSVERL Shapley results for appendix.
"""

from fastsverl.utils import calculate_final_stats, load_data, plot

# GWB Policy
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_wo_exact = None  # e.g. "wo_exact_shapleys_dqn_policy_gwb_{timestamp}"
group_w_exact = None  # e.g. "w_exact_shapleys_dqn_policy_gwb_{timestamp}"

def gwb_policy(): return {
    "tag": "model/PolicyShapley_exact_loss",
    "title": "Behaviour Shapley",
    "figure_name": "app_gwb_policy_shapley",
    "groups_data": [
        {
            "data": load_data("FastSVERL_GWB", group_wo_exact),
            "label": "Exact characteristic",
        },
        {
            "data": load_data("FastSVERL_GWB", group_w_exact),
            "label": "Approx characteristic",
        },
    ],
}

# GWB Value
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_wo_exact = None  # e.g. "wo_exact_shapleys_dqn_value_gwb_{timestamp}"
group_w_exact = None  # e.g. "w_exact_shapleys_dqn_value_gwb_{timestamp}"

def gwb_value(): return {
    "tag": "model/ValueShapley_exact_loss",
    "figure_name": "app_gwb_value_shapley",
    "title": "Prediction Shapley",
    "groups_data": [
        {
            "data": load_data("FastSVERL_GWB", group_wo_exact),
            "label": "Exact characteristic",
        },
        {
            "data": load_data("FastSVERL_GWB", group_w_exact),
            "label": "Approx characteristic",
        },
    ],
}

# GWB Performance
# Set `group` (from W&B) to match the experiment.
# Example: group = "shapleys_dqn_performance_gwb_{timestamp}"
def gwb_performance(): 
    group = None
    return {
    "tag": "model/PerformanceShapley_exact_loss",
    "figure_name": "app_gwb_perf_shapley",
    "title": "Outcome Shapley",
    "ylim": (-.01, 0.55),
    "groups_data": [
        {
            "data": load_data("FastSVERL_GWB", f"w_policy_{group}"),
            "label": "Approx outcome, approx behaviour",
            "color": "C2",
        },
        {
            "data": load_data("FastSVERL_GWB", f"wo_policy_{group}"),
            "label": "Approx outcome, exact behaviour",
            "color": "C0",
        },
        {
            "data": load_data("FastSVERL_GWB", f"wo_perf_{group}"),
            "label": "Exact outcome, exact behaviour",
            "color": "C1",
        },
    ],
}

# Mastermind-222 Policy
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_wo_exact = None  # e.g. "222_wo_exact_shapleys_dqn_policy_mastermind_{timestamp}"
group_w_exact = None  # e.g. "222_w_exact_shapleys_dqn_policy_mastermind_{timestamp}"

def mastermind_222_policy(): return {
    "tag": "model/PolicyShapley_exact_loss",
    "figure_name": "app_mastermind_222_policy_shapley",
    "title": "Behaviour Shapley",
    "ylabel": "MSE",
    "groups_data": [
        {
            "data": load_data("FastSVERL_Mastermind", group_wo_exact),
            "label": "Exact characteristic",
        },
        {
            "data": load_data("FastSVERL_Mastermind", group_w_exact),
            "label": "Approx characteristic",
        },
    ],
}

# Mastermind-222 Value
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_wo_exact = None  # e.g. "222_wo_exact_shapleys_dqn_value_mastermind_{timestamp}"
group_w_exact = None  # e.g. "222_w_exact_shapleys_dqn_value_mastermind_{timestamp}"

def mastermind_222_value(): return {
    "tag": "model/ValueShapley_exact_loss",
    "figure_name": "app_mastermind_222_value_shapley",
    "title": "Prediction Shapley",
    "ylabel": "MSE",
    "groups_data": [
        {
            "data": load_data("FastSVERL_Mastermind", group_wo_exact),
            "label": "Exact characteristic",
        },
        {
            "data": load_data("FastSVERL_Mastermind", group_w_exact),
            "label": "Approx characteristic",
        },
    ],
}

# Mastermind-222 Performance
# Set `group` (from W&B) to match the experiment.
# Example: group = "shapleys_dqn_performance_mastermind_{timestamp}"

def mastermind_222_performance(): 
    group = None
    return {
    "tag": "model/PerformanceShapley_exact_loss",
    "xlim": (-100, 7000),
    "figure_name": "app_mastermind_222_perf_shapley",
    "title": "Outcome Shapley",
    "ylabel": "MSE",
    "groups_data": [
        {
            "data": load_data("FastSVERL_Mastermind", f"222_w_policy_{group}"),
            "label": "Approx outcome, approx behaviour",
            "color": "C2",
        },
        {
            "data": load_data("FastSVERL_Mastermind", f"222_wo_policy_{group}"),
            "label": "Approx outcome, exact behaviour",
            "color": "C0",
        },
        {
            "data": load_data("FastSVERL_Mastermind", f"222_wo_perf_{group}"),
            "label": "Exact outcome, exact behaviour",
            "color": "C1",
        },
        
    ],
}

# Mastermind-333 Policy
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_wo_exact = None  # e.g. "333_wo_exact_shapleys_dqn_policy_mastermind_{timestamp}"
group_w_exact = None  # e.g. "333_w_exact_shapleys_dqn_policy_mastermind_{timestamp}"

def mastermind_333_policy(): return {
    "tag": "model/PolicyShapley_exact_loss",
    "figure_name": "app_mastermind_333_policy_shapley",
    "title": "Behaviour Shapley",
    "groups_data": [
        {
            "data": load_data("FastSVERL_Mastermind", group_wo_exact),
            "label": "Exact characteristic",
        },
        {
            "data": load_data("FastSVERL_Mastermind", group_w_exact),
            "label": "Approx characteristic",
        },
    ],
}

# Mastermind-333 Value
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_wo_exact = None  # e.g. "333_wo_exact_shapleys_dqn_value_mastermind_{timestamp}"
group_w_exact = None  # e.g. "333_w_exact_shapleys_dqn_value_mastermind_{timestamp}"

def mastermind_333_value(): return {
    "tag": "model/ValueShapley_exact_loss",
    "figure_name": "app_mastermind_333_value_shapley",
    "title": "Prediction Shapley",
    "xticks": (0, 10_000, 20_000, 30_000),
    "groups_data": [
        {
            "data": load_data("FastSVERL_Mastermind", group_wo_exact),
            "label": "Exact characteristic",
        },
        {
            "data": load_data("FastSVERL_Mastermind", group_w_exact),
            "label": "Approx characteristic",
        },
    ],
}

# Mastermind-222 Large Scale Behaviour
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_443 = None  # e.g. "443_shapleys_dqn_policy_mastermind_{timestamp}"
group_453 = None  # e.g. "453_shapleys_dqn_policy_mastermind_{timestamp}"
group_463 = None  # e.g. "463_shapleys_dqn_policy_mastermind_{timestamp}"

def mastermind_scale_policy(): 
    return {
    "tag": "model/PolicyShapley_epoch_loss",
    "figure_name": "app_large_policy_shapley",
    "title": "Shapley",
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

# Mastermind-222 Large Scale Prediction
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_443 = None  # e.g. "443_shapleys_dqn_value_mastermind_{timestamp}"
group_453 = None  # e.g. "453_shapleys_dqn_value_mastermind_{timestamp}"
group_463 = None  # e.g. "463_shapleys_dqn_value_mastermind_{timestamp}"

def mastermind_scale_value(): 
    return {
    "tag": "model/ValueShapley_epoch_loss",
    "figure_name": "app_large_value_shapley",
    "title": "Shapley",
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

# Mastermind-222 Large Scale On-Policy Performance
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_443 = None  # e.g. "443_onpolicy_shapleys_dqn_performance_mastermind_{timestamp}"
group_453 = None  # e.g. "453_onpolicy_shapleys_dqn_performance_mastermind_{timestamp}"
group_463 = None  # e.g. "463_onpolicy_shapleys_dqn_performance_mastermind_{timestamp}"

def mastermind_scale_perf_onpolicy(): 
    return {
    "tag": "model/PerformanceShapley_epoch_loss",
    "figure_name": "app_large_perf_onpolicy_shapley",
    "title": "Shapley",
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

# Mastermind-222 Large Scale Off-Policy Performance
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_443 = None  # e.g. "443_offpolicy_shapleys_dqn_performance_mastermind_{timestamp}"
group_453 = None  # e.g. "453_offpolicy_shapleys_dqn_performance_mastermind_{timestamp}"
group_463 = None  # e.g. "463_offpolicy_shapleys_dqn_performance_mastermind_{timestamp}"

def mastermind_scale_perf_offpolicy(): 
    return {
    "tag": "model/PerformanceShapley_epoch_loss",
    "figure_name": "app_large_perf_offpolicy_shapley",
    "title": "Shapley",
    "xticks": (0, 200_000, 400_000, 600_000),
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
    mastermind_scale_perf_onpolicy,
    mastermind_scale_perf_offpolicy,
]:
    
    # Plot
    plot_spec = plot_spec_func()

    # Common plot settings
    plot_spec["xlabel"] = "Training updates"
    plot_spec["legend_loc"] = "upper right"
    plot_spec["fontsize"] = 16
    plot_spec["n_batches"] = (0.8 * 10_000) // 128,

    plot(**plot_spec)

############################ For table generation ############################

# Mastermind Large Scale Prediction
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_443_char = None  # e.g. "443_characteristics_dqn_value_mastermind_{timestamp}"
group_453_char = None  # e.g. "453_characteristics_dqn_value_mastermind_{timestamp}"
group_463_char = None  # e.g. "463_characteristics_dqn_value_mastermind_{timestamp}"
group_443_shap = None  # e.g. "443_shapleys_dqn_value_mastermind_{timestamp}"
group_453_shap = None  # e.g. "453_shapleys_dqn_value_mastermind_{timestamp}"
group_463_shap = None  # e.g. "463_shapleys_dqn_value_mastermind_{timestamp}"

def mastermind_scale_value_table(): 
    return {
    "figure_name": "app_large_value_shapley",
    "groups_data": [
        {
            "tag": "model/ValueCharacteristicModel_epoch_loss",
            "data": load_data("FastSVERL_Mastermind", group_443_char, "../../characteristics/runs/"),
            "label": "Mastermind-443 Characteristic",
        },
        {
            "tag": "model/ValueShapley_epoch_loss",
            "data": load_data("FastSVERL_Mastermind", group_443_shap),
            "label": "Mastermind-443 Shapley",
        },
        {
            "tag": "model/ValueCharacteristicModel_epoch_loss",
            "data": load_data("FastSVERL_Mastermind", group_453_char, "../../characteristics/runs/"),
            "label": "Mastermind-453 Characteristic",
        },
        {
            "tag": "model/ValueShapley_epoch_loss",
            "data": load_data("FastSVERL_Mastermind", group_453_shap),
            "label": "Mastermind-453 Shapley",
        },
        {
            "tag": "model/ValueCharacteristicModel_epoch_loss",
            "data": load_data("FastSVERL_Mastermind", group_463_char, "../../characteristics/runs/"),
            "label": "Mastermind-463 Characteristic",
        },
        {
            "tag": "model/ValueShapley_epoch_loss",
            "data": load_data("FastSVERL_Mastermind", group_463_shap),
            "label": "Mastermind-463 Shapley",
        },
        
    ],
}

# Mastermind Large Scale On-Policy Performance
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_443 = None  # e.g. "443_onpolicy_shapleys_dqn_performance_mastermind_{timestamp}"
group_453 = None  # e.g. "453_onpolicy_shapleys_dqn_performance_mastermind_{timestamp}"
group_463 = None  # e.g. "463_onpolicy_shapleys_dqn_performance_mastermind_{timestamp}"

def mastermind_scale_perf_onpolicy_table():
    return {
    "figure_name": "app_large_perf_onpolicy_shapley",
    "groups_data": [
        {
            "tag": "model/PerformanceShapley_epoch_loss",
            "data": load_data("FastSVERL_Mastermind", group_443),
            "label": "Mastermind-443 Shapley",
        },
        {
            "tag": "model/PerformanceShapley_epoch_loss",
            "data": load_data("FastSVERL_Mastermind", group_453),
            "label": "Mastermind-453 Shapley",
        },
        {
            "tag": "model/PerformanceShapley_epoch_loss",
            "data": load_data("FastSVERL_Mastermind", group_463),
            "label": "Mastermind-463 Shapley",
        },
        
    ],
}

# Mastermind Large Scale Off-Policy Performance
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_443 = None  # e.g. "443_offpolicy_shapleys_dqn_performance_mastermind_{timestamp}"
group_453 = None  # e.g. "453_offpolicy_shapleys_dqn_performance_mastermind_{timestamp}"
group_463 = None  # e.g. "463_offpolicy_shapleys_dqn_performance_mastermind_{timestamp}"

def mastermind_scale_perf_offpolicy_table():
    return {
    "figure_name": "app_large_perf_offpolicy_shapley",
    "groups_data": [
        {
            "tag": "model/PerformanceShapley_epoch_loss",
            "data": load_data("FastSVERL_Mastermind", group_443),
            "label": "Mastermind-443 Shapley",
        },
        {
            "tag": "model/PerformanceShapley_epoch_loss",
            "data": load_data("FastSVERL_Mastermind", group_453),
            "label": "Mastermind-453 Shapley",
        },
        {
            "tag": "model/PerformanceShapley_epoch_loss",
            "data": load_data("FastSVERL_Mastermind", group_463),
            "label": "Mastermind-463 Shapley",
        },
        
    ],
}

# Loop over each
for table_spec_func in [
    mastermind_scale_value_table,
    mastermind_scale_perf_onpolicy_table,
    mastermind_scale_perf_offpolicy_table,
]:

    # Plot
    table_spec = table_spec_func()
    table_spec["n_batches"] = (0.8 * 10_000) // 128,
    calculate_final_stats(**table_spec)