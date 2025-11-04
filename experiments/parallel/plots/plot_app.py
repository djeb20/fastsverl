"""
Plotting script for parallel appendix experiments.
"""

from fastsverl.utils import load_data, plot

# GWB Expected Return
# Set group name (from W&B) to match the experiment.
group_name = None # e.g. "1_wo_exact_parallel_training_dqn_policy_gwb_{timestamp}"

def gwb_expected_return(): return {
    "tag": "agent/episodic_return",
    "figure_name": "app_gwb_parallel_return",
    "title": "Expected return",
    "groups_data": [
        {
            "data": load_data("FastSVERL-OffPolicy_UpdateRate_GWB", group_name),
        },
    ],
}

# GWB DQN Loss
# Set group name (from W&B) to match the experiment.
group_name = None  # e.g. "1_wo_exact_update_rate_dqn_policy_gwb_{timestamp}"

def gwb_dqn_loss(): return {
    "tag": "agent/DQN_loss",
    "figure_name": "app_gwb_parallel_dqn_loss",
    "title": "DQN loss",
    "groups_data": [
        {
            "data": load_data("FastSVERL-OffPolicy_UpdateRate_GWB", group_name),
        },
    ],
}

# GWB Policy Update Rate; Approx Characteristic
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_name_1 = None  # e.g. "1_w_exact_update_rate_dqn_policy_gwb_{timestamp}"
group_name_2 = None  # e.g. "2_w_exact_update_rate_dqn_policy_gwb_{timestamp}"
group_name_10 = None  # e.g. "10_w_exact_update_rate_dqn_policy_gwb_{timestamp}"
group_name_50 = None  # e.g. "50_w_exact_update_rate_dqn_policy_gwb_{timestamp}"

def gwb_policy_update_rate_approx(): return {
    "tag": "agent/PolicyShapley_exact_loss",
    "figure_name": "app_gwb_parallel_policy_approx",
    "title": "Behaviour Shapley",
    "ylim": (-.0002, 0.0043),
    "groups_data": [
        {
            "data": load_data("FastSVERL-OffPolicy_UpdateRate_GWB", group_name_1),
            "label": "1:1",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_UpdateRate_GWB", group_name_2),
            "label": "2:1",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_UpdateRate_GWB", group_name_10),
            "label": "10:1",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_UpdateRate_GWB", group_name_50),
            "label": "50:1",
        },
    ],
}

# GWB Value Update Rate; Approx Characteristic
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_name_1 = None  # e.g. "1_w_exact_update_rate_dqn_value_gwb_{timestamp}"
group_name_2 = None  # e.g. "2_w_exact_update_rate_dqn_value_gwb_{timestamp}"
group_name_10 = None  # e.g. "10_w_exact_update_rate_dqn_value_gwb_{timestamp}"
group_name_50 = None  # e.g. "50_w_exact_update_rate_dqn_value_gwb_{timestamp}"

def gwb_value_update_rate_approx(): return {
    "tag": "agent/ValueShapley_exact_loss",
    "figure_name": "app_gwb_parallel_value_approx",
    "title": "Prediction Shapley",
    "groups_data": [
        {
            "data": load_data("FastSVERL-OffPolicy_UpdateRate_GWB", group_name_1),
            "label": "1:1",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_UpdateRate_GWB", group_name_2),
            "label": "2:1",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_UpdateRate_GWB", group_name_10),
            "label": "10:1",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_UpdateRate_GWB", group_name_50),
            "label": "50:1",
        },
    ],
}

# GWB Performance Update Rate; Approx Performance Characteristic + Approx Policy Characteristic
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_name_1 = None  # e.g. "1_w_policy_update_rate_dqn_performance_gwb_{timestamp}"
group_name_2 = None  # e.g. "2_w_policy_update_rate_dqn_performance_gwb_{timestamp}"
group_name_10 = None  # e.g. "10_w_policy_update_rate_dqn_performance_gwb_{timestamp}"
group_name_50 = None  # e.g. "50_w_policy_update_rate_dqn_performance_gwb_{timestamp}"

def gwb_performance_update_rate_approx_perf_approx_policy(): return {
    "tag": "agent/PerformanceShapley_exact_loss",
    "figure_name": "app_gwb_parallel_perf_approx_behaviour",
    "title": "Outcome Shapley",
    "groups_data": [
        {
            "data": load_data("FastSVERL-OffPolicy_UpdateRate_GWB", group_name_1),
            "label": "1:1",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_UpdateRate_GWB", group_name_2),
            "label": "2:1",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_UpdateRate_GWB", group_name_10),
            "label": "10:1",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_UpdateRate_GWB", group_name_50),
            "label": "50:1",
        },
    ],
}

# Mastermind-222 Expected Return
# Set group name (from W&B) to match the experiment.
group_name = None # e.g. "wo_exact_parallel_training_dqn_policy_mastermind_{timestamp}"

def mastermind_222_expected_return(): return {
    "tag": "agent/episodic_return",
    "ylabel": "Return",
    "figure_name": "app_mastermind_222_parallel_return",
    "title": "Expected return",
    "groups_data": [
        {
            "data": load_data("FastSVERL-OffPolicy_Parallel_Mastermind", group_name),
        },
    ],
}

# Mastermind-222 DQN Loss
# Set group name (from W&B) to match the experiment.
group_name = None  # e.g. "wo_exact_update_rate_dqn_policy_mastermind_{timestamp}"

def mastermind_222_dqn_loss(): return {
    "tag": "agent/DQN_loss",
    "ylabel": "Loss",
    "figure_name": "app_mastermind_222_parallel_dqn_loss",
    "title": "DQN loss",
    "groups_data": [
        {
            "data": load_data("FastSVERL-OffPolicy_Parallel_Mastermind", group_name),
        },
    ],
}

# Mastermind-222 Policy Update Rate; Approx Characteristic
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_1_w_exact = None  # e.g. "1_w_exact_update_rate_dqn_policy_mastermind_{timestamp}"
group_2_w_exact = None  # e.g. "2_w_exact_update_rate_dqn_policy_mastermind_{timestamp}"
group_10_w_exact = None  # e.g. "10_w_exact_update_rate_dqn_policy_mastermind_{timestamp}"
group_50_w_exact = None  # e.g. "50_w_exact_update_rate_dqn_policy_mastermind_{timestamp}"

def mastermind_222_policy_update_rate_approx(): return {
    "tag": "agent/PolicyShapley_exact_loss",
    "figure_name": "app_mastermind_222_parallel_policy_approx",
    "title": "Behaviour Shapley",
    "ylim": (-.001, 0.03),
    "ylabel": "MSE",
    "legend_ncol": 2,
    "groups_data": [
        {
            "data": load_data("FastSVERL-OffPolicy_Parallel_Mastermind", group_1_w_exact),
            "label": "1:1",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_UpdateRate", group_2_w_exact),
            "label": "2:1",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_UpdateRate", group_10_w_exact),
            "label": "10:1",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_UpdateRate", group_50_w_exact),
            "label": "50:1",
        },
    ],
}

# Mastermind-222 Value Update Rate; Approx Characteristic
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_1_w_exact = None  # e.g. "1_w_exact_update_rate_dqn_value_mastermind_{timestamp}"
group_2_w_exact = None  # e.g. "2_w_exact_update_rate_dqn_value_mastermind_{timestamp}"
group_10_w_exact = None  # e.g. "10_w_exact_update_rate_dqn_value_mastermind_{timestamp}"
group_50_w_exact = None  # e.g. "50_w_exact_update_rate_dqn_value_mastermind_{timestamp}"

def mastermind_222_value_update_rate_approx(): return {
    "tag": "agent/ValueShapley_exact_loss",
    "ylim": (-.0005, 0.023),
    "figure_name": "app_mastermind_222_parallel_value_approx",
    "title": "Prediction Shapley",
    "ylabel": "MSE",
    "xticks": (0, 5_000, 10_000),
    "groups_data": [
        {
            "data": load_data("FastSVERL-OffPolicy_Parallel_Mastermind", group_1_w_exact),
            "label": "1:1",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_UpdateRate", group_2_w_exact),
            "label": "2:1",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_UpdateRate", group_10_w_exact),
            "label": "10:1",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_UpdateRate", group_50_w_exact),
            "label": "50:1",
        },
    ],
}

# Mastermind-222 Performance Update Rate; Approx Performance Characteristic + Approx Policy Characteristic
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_1_w_exact = None  # e.g. "1_w_policy_update_rate_dqn_performance_mastermind_{timestamp}"
group_2_w_exact = None  # e.g. "2_w_policy_update_rate_dqn_performance_mastermind_{timestamp}"
group_10_w_exact = None  # e.g. "10_w_policy_update_rate_dqn_performance_mastermind_{timestamp}"
group_50_w_exact = None  # e.g. "50_w_policy_update_rate_dqn_performance_mastermind_{timestamp}"

def mastermind_222_performance_update_rate_approx_perf_approx_policy(): return {
    "tag": "agent/PerformanceShapley_exact_loss",
    "figure_name": "app_mastermind_222_parallel_perf_approx_behaviour",
    "title": "Outcome Shapley",
    "ylabel": "MSE",
    "ylim": (-.0005, 0.08),
    "legend_ncol": 2,
    "groups_data": [
        {
            "data": load_data("FastSVERL-OffPolicy_Parallel_Mastermind", group_1_w_exact),
            "label": "1:1",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_UpdateRate", group_2_w_exact),
            "label": "2:1",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_UpdateRate", group_10_w_exact),
            "label": "10:1",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_UpdateRate", group_50_w_exact),
            "label": "50:1",
        },
    ],
}

# Loop over each
for plot_spec_func in [
    gwb_expected_return,
    gwb_dqn_loss,
    gwb_policy_update_rate_approx,
    gwb_value_update_rate_approx,
    gwb_performance_update_rate_approx_perf_approx_policy,
    mastermind_222_expected_return,
    mastermind_222_dqn_loss,
    mastermind_222_policy_update_rate_approx,
    mastermind_222_value_update_rate_approx,
    mastermind_222_performance_update_rate_approx_perf_approx_policy,
]:
    
    # Plot
    plot_spec = plot_spec_func()
    plot_spec["xlabel"] = "Training updates"
    plot_spec["legend_loc"] = "upper right"
    plot_spec["fontsize"] = 18
    plot_spec["n_batches"] = 0.1 # Scaling because models updated every 10 env steps.

    plot(**plot_spec)