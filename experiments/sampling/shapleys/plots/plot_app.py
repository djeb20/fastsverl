"""
Plots for appendix sampling experiments (Shapleys).
"""

from fastsverl.utils import load_data, plot

# GWB Policy 
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_wo_exact_model = None  # e.g. "wo_exact_model_shapleys_dqn_policy_gwb_{timestamp}"
group_w_exact_sampling = None  # e.g. "w_exact_sampling_shapleys_dqn_policy_gwb_{timestamp}"
group_w_exact_model = None  # e.g. "w_exact_model_shapleys_dqn_policy_gwb_{timestamp}"

def gwb_policy(): return {
    "tag": "model/PolicyShapley_exact_loss",
    "legend_loc": "upper center",
    "legend_bbox_to_anchor": (0.4, 1),
    "figure_name": "app_gwb_policy_sampling",
    "title": "Behaviour Shapley",
    "n_batches": (0.8 * 10_000) // 128,
    "groups_data": [
        {
            "data": load_data("FastSVERL-Optimising_Stochastic_GWB", group_wo_exact_model),
            "label": "Exact",
            "start_epoch": 0,
        },
        {
            "data": load_data("FastSVERL-Optimising_Stochastic_GWB", group_w_exact_sampling),
            "label": "Sample",
            "start_epoch": 0,
        },
        {
            "data": load_data("FastSVERL-Optimising_Stochastic_GWB", group_w_exact_model),
            "label": "Model",
            "start_epoch": 50,
        },
    ],
}

# GWB Value
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_wo_exact_model = None  # e.g. "wo_exact_model_shapleys_dqn_value_gwb_{timestamp}"
group_w_exact_sampling = None  # e.g. "w_exact_sampling_shapleys_dqn_value_gwb_{timestamp}"
group_w_exact_model = None  # e.g. "w_exact_model_shapleys_dqn_value_gwb_{timestamp}"

def gwb_value(): return {
    "tag": "model/ValueShapley_exact_loss",
    "figure_name": "app_gwb_value_sampling",
    "title": "Prediction Shapley",
    "legend_loc": "upper center",
    "legend_bbox_to_anchor": (0.35, 1),
    "n_batches": (0.8 * 10_000) // 128,
    "groups_data": [
        {
            "data": load_data("FastSVERL-Optimising_Stochastic_GWB", group_wo_exact_model),
            "label": "Exact",
            "start_epoch": 0,
        },
        {
            "data": load_data("FastSVERL-Optimising_Stochastic_GWB", group_w_exact_sampling),
            "label": "Sample",
            "start_epoch": 0,
        },
        {
            "data": load_data("FastSVERL-Optimising_Stochastic_GWB", group_w_exact_model),
            "label": "Model",
            "start_epoch": 30,
        },
    ],
}

# Mastermind-222 Policy
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_wo_exact_model = None  # e.g. "222_wo_exact_model_shapleys_dqn_policy_mastermind_{timestamp}"
group_w_exact_sampling = None  # e.g. "222_w_exact_sampling_shapleys_dqn_policy_mastermind_{timestamp}"
group_w_exact_model = None  # e.g. "222_w_exact_model_shapleys_dqn_policy_mastermind_{timestamp}"

def mastermind_222_policy(): return {
    "tag": "model/PolicyShapley_exact_loss",
    "ylabel": "MSE",
    "legend_loc": "upper right",
    "xlim": (-100, 7000),
    "figure_name": "app_mastermind_222_policy_sampling",
    "title": "Behaviour Shapley",
    "n_batches": (0.8 * 10_000) // 128,
    "groups_data": [
        {
            "data": load_data("FastSVERL-Optimising_Stochastic_Mastermind", group_wo_exact_model),
            "label": "Exact",
            "start_epoch": 0,
            
        },
        {
            "data": load_data("FastSVERL-Optimising_Stochastic_Mastermind", group_w_exact_sampling),
            "label": "Sample",
            "start_epoch": 0,
        },
        {
            "data": load_data("FastSVERL-Optimising_Stochastic_Mastermind", group_w_exact_model),
            "label": "Model",
            "start_epoch": 50,
        },
    ],
}

# Mastermind-222 Value
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_wo_exact_model = None  # e.g. "222_wo_exact_model_shapleys_dqn_value_mastermind_{timestamp}"
group_w_exact_sampling = None  # e.g. "222_w_exact_sampling_shapleys_dqn_value_mastermind_{timestamp}"
group_w_exact_model = None  # e.g. "222_w_exact_model_shapleys_dqn_value_mastermind_{timestamp}"

def mastermind_222_value(): return {
    "tag": "model/ValueShapley_exact_loss",
    "figure_name": "app_mastermind_222_value_sampling",
    "title": "Prediction Shapley",
    "ylabel": "MSE",
    "legend_loc": "upper right",
    "n_batches": (0.8 * 10_000) // 128,
    "groups_data": [
        {
            "data": load_data("FastSVERL-Optimising_Stochastic_Mastermind", group_wo_exact_model),
            "label": "Exact",
            "start_epoch": 0,
        },
        {
            "data": load_data("FastSVERL-Optimising_Stochastic_Mastermind", group_w_exact_sampling),
            "label": "Sample",
            "start_epoch": 0,
        },
        {
            "data": load_data("FastSVERL-Optimising_Stochastic_Mastermind", group_w_exact_model),
            "label": "Model",
            "start_epoch": 50,
        },
    ],
}

# Mastermind-333 Policy
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_wo_exact_model = None  # e.g. "333_wo_exact_model_shapleys_dqn_policy_mastermind_{timestamp}"
group_w_exact_sampling = None  # e.g. "333_w_exact_sampling_shapleys_dqn_policy_mastermind_{timestamp}"
group_w_exact_model = None  # e.g. "333_w_exact_model_shapleys_dqn_policy_mastermind_{timestamp}"

def mastermind_333_policy(): return {
    "tag": "model/PolicyShapley_exact_loss",
    "legend_loc": "upper center",
    "figure_name": "app_mastermind_333_policy_sampling",
    "title": "Behaviour Shapley",
    "n_batches": (0.8 * 10_000) // 128,
    "groups_data": [
        {
            "data": load_data("FastSVERL-Optimising_Stochastic_Mastermind", group_wo_exact_model),
            "label": "Exact",
            "start_epoch": 0,
        },
        {
            "data": load_data("FastSVERL-Optimising_Stochastic_Mastermind", group_w_exact_sampling),
            "label": "Sample",
            "start_epoch": 0,
        },
        {
            "data": load_data("FastSVERL-Optimising_Stochastic_Mastermind", group_w_exact_model),
            "label": "Model",
            "start_epoch": 1000,
        },
    ],
}

# Mastermind-333 Value
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_wo_exact_model = None  # e.g. "333_wo_exact_model_shapleys_dqn_value_mastermind_{timestamp}"
group_w_exact_sampling = None  # e.g. "333_w_exact_sampling_shapleys_dqn_value_mastermind_{timestamp}"
group_w_exact_model = None  # e.g. "333_w_exact_model_shapleys_dqn_value_mastermind_{timestamp}"

def mastermind_333_value(): return {
    "tag": "model/ValueShapley_exact_loss",
    "figure_name": "app_mastermind_333_value_sampling",
    "title": "Prediction Shapley",
    "legend_loc": "upper center",
    "n_batches": (0.8 * 10_000) // 128,
    "groups_data": [
        {
            "data": load_data("FastSVERL-Optimising_Stochastic_Mastermind", group_wo_exact_model),
            "label": "Exact",
            "start_epoch": 0,
        },
        {
            "data": load_data("FastSVERL-Optimising_Stochastic_Mastermind", group_w_exact_sampling),
            "label": "Sample",
            "start_epoch": 0,
        },
        {
            "data": load_data("FastSVERL-Optimising_Stochastic_Mastermind", group_w_exact_model),
            "label": "Model",
            "start_epoch": 2000,
        },
    ],
}

# Loop over each
for plot_spec_func in [
    gwb_policy,
    gwb_value,
    mastermind_222_policy,
    mastermind_222_value,
    mastermind_333_policy,
    mastermind_333_value,
]:
    
    # Plot
    plot_spec = plot_spec_func()
    plot_spec["xlabel"] = "Training updates"
    plot_spec["fontsize"] = 18

    plot(**plot_spec)