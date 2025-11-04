"""
Plots for the appendix off-policy experiments.
"""

from fastsverl.utils import load_data, plot

# Mastermind-222 Policy
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_wo_weight = None  # e.g. "wo_weight_new_losses_dqn_policy_mastermind_{timestamp}"
group_w_weight = None  # e.g. "w_weight_new_losses_dqn_policy_mastermind_{timestamp}"
group_exact = None  # e.g. "exact_new_losses_dqn_policy_mastermind_{timestamp}"

def mastermind_222_policy(): return {
    "tag": "model/PolicyCharacteristicModel_exact_loss",
    "ylabel": "MSE",
    "ylim": (-.003, 0.03),
    "figure_name": "app_mastermind_222_offpolicy_policy",
    "n_batches": (0.8 * 30_000) // 128,
    "groups_data": [
        {
            'data': load_data("FastSVERL-OffPolicy_NewLoss_Mastermind", group_wo_weight),
            'label': "Off-policy (without IS)",
        },
        {
            'data': load_data("FastSVERL-OffPolicy_NewLoss_Mastermind", group_w_weight),
            'label': "Off-policy (with IS)",
        },
        {
            'data': load_data("FastSVERL-OffPolicy_NewLoss_Mastermind", group_exact),
            'label': "On-policy",
        },
    ]
}

# Mastermind-222 Value
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_wo_weight = None  # e.g. "wo_weight_new_losses_dqn_value_mastermind_{timestamp}"
group_w_weight = None  # e.g. "w_weight_new_losses_dqn_value_mastermind_{timestamp}"
group_exact = None  # e.g. "exact_new_losses_dqn_value_mastermind_{timestamp}"

def mastermind_222_value(): return {
    "tag": "model/ValueCharacteristicModel_exact_loss",
    "figure_name": "app_mastermind_222_offpolicy_value",
    "ylim": (-0.005, 0.05),
    "n_batches": (0.8 * 30_000) // 128,
    "groups_data": [
        {
            "data": load_data("FastSVERL-OffPolicy_NewLoss_Mastermind", group_wo_weight),
            "label": "Off-policy (without IS)",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_NewLoss_Mastermind", group_w_weight),
            "label": "Off-policy (with IS)",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_NewLoss_Mastermind", group_exact),
            "label": "On-policy",
        },
    ],
}

# GWB Policy Clipping
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_0_9 = None  # e.g. "0.9_clipping_dqn_policy_gwb_{timestamp}"
group_0_99 = None  # e.g. "0.99_clipping_dqn_policy_gwb_{timestamp}"
group_0_995 = None  # e.g. "0.995_clipping_dqn_policy_gwb_{timestamp}"
group_0_998 = None  # e.g. "0.998_clipping_dqn_policy_gwb_{timestamp}"
group_1 = None  # e.g. "1_clipping_dqn_policy_gwb_{timestamp}"
group_exact = None  # e.g. "exact_clipping_dqn_policy_gwb_{timestamp}"

def gwb_policy_clipping(): return {
    "tag": "model/PolicyCharacteristicModel_exact_loss",
    "ylabel": "MSE",
    "xlim": (0, 5500),
    "legend_bbox_to_anchor": (1, 1.05),
    "figure_name": "app_gwb_offpolicy_clipping_policy",
    "n_batches": (0.8 * 15_000) // 128,
    "groups_data": [
        {
            "data": load_data("FastSVERL-OffPolicy_GWB_Clipping", group_0_9, '../clipping/runs'),
            "label": "Without IS",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_GWB_Clipping", group_0_99, '../clipping/runs'),
            "label": "With IS (clip = 0.99)",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_GWB_Clipping", group_0_995, '../clipping/runs'),
            "label": "With IS (clip = 0.995)",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_GWB_Clipping", group_0_998, '../clipping/runs'),
            "label": "With IS (clip = 0.998)",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_GWB_Clipping", group_1, '../clipping/runs'),
            "label": "With IS (no clipping)",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_GWB_Clipping", group_exact, '../clipping/runs'),
            "label": "On-policy",
        },
    ]
}

# GWB Value Clipping
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_0_9 = None  # e.g. "0.9_clipping_dqn_value_gwb_{timestamp}"
group_0_99 = None  # e.g. "0.99_clipping_dqn_value_gwb_{timestamp}"
group_0_995 = None  # e.g. "0.995_clipping_dqn_value_gwb_{timestamp}"
group_0_998 = None  # e.g. "0.998_clipping_dqn_value_gwb_{timestamp}"
group_1 = None  # e.g. "1_clipping_dqn_value_gwb_{timestamp}"
group_exact = None  # e.g. "exact_clipping_dqn_value_gwb_{timestamp}"

def gwb_value_clipping(): return {
    "tag": "model/ValueCharacteristicModel_exact_loss",
    "figure_name": "app_gwb_offpolicy_clipping_value",
    "xlim": (0, 10000),
    "ylim": (-.1, 6.6),
    "legend_bbox_to_anchor": (1, 1.05),
    "n_batches": (0.8 * 15_000) // 128,
    "groups_data": [
        {
            "data": load_data("FastSVERL-OffPolicy_GWB_Clipping", group_0_9, '../clipping/runs'),
            "label": "Without IS",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_GWB_Clipping", group_0_99, '../clipping/runs'),
            "label": "With IS (clip = 0.99)",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_GWB_Clipping", group_0_995, '../clipping/runs'),
            "label": "With IS (clip = 0.995)",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_GWB_Clipping", group_0_998, '../clipping/runs'),
            "label": "With IS (clip = 0.998)",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_GWB_Clipping", group_1, '../clipping/runs'),
            "label": "With IS (no clipping)",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_GWB_Clipping", group_exact, '../clipping/runs'),
            "label": "On-policy",
        },
    ],
}    

# GWB Policy Weighting
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_w_weight = None  # e.g. "w_weight_weighting_dqn_policy_gwb_{timestamp}"
group_wo_weight = None  # e.g. "wo_weight_weighting_dqn_policy_gwb_{timestamp}"
group_weighted_weight = None  # e.g. "weighted_weight_weighting_dqn_policy_gwb_{timestamp}"
group_exact = None  # e.g. "exact_weighting_dqn_policy_gwb_{timestamp}"

def gwb_policy_weighting(): return {
    "tag": "model/PolicyCharacteristicModel_exact_loss",
    "ylabel": "MSE",
    "figure_name": "app_gwb_offpolicy_weighting_policy",
    "n_batches": (0.8 * 15_000) // 128,
    "groups_data": [
        {
            "data": load_data("FastSVERL-OffPolicy_GWB_Weighting", group_w_weight, '../weighting/runs'),
            "label": "Unnormalised IS",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_GWB_Weighting", group_wo_weight, '../weighting/runs'),
            "label": "Without IS",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_GWB_Weighting", group_weighted_weight, '../weighting/runs'),
            "label": "Normalised IS",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_GWB_Weighting", group_exact, '../weighting/runs'),
            "label": "On-policy",
        },
    ]
}

# GWB Value Weighting
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_w_weight = None  # e.g. "w_weight_weighting_dqn_value_gwb_{timestamp}"
group_wo_weight = None  # e.g. "wo_weight_weighting_dqn_value_gwb_{timestamp}"
group_weighted_weight = None  # e.g. "weighted_weight_weighting_dqn_value_gwb_{timestamp}"
group_exact = None  # e.g. "exact_weighting_dqn_value_gwb_{timestamp}"

def gwb_value_weighting(): return {
    "tag": "model/ValueCharacteristicModel_exact_loss",
    "figure_name": "app_gwb_offpolicy_weighting_value",
    "ylim": (-.5, 10),
    "yticks": (0, 5, 10),
    "n_batches": (0.8 * 15_000) // 128,
    "groups_data": [
        {
            "data": load_data("FastSVERL-OffPolicy_GWB_Weighting", group_w_weight, '../weighting/runs'),
            "label": "Unnormalised IS",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_GWB_Weighting", group_wo_weight, '../weighting/runs'),
            "label": "Without IS",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_GWB_Weighting", group_weighted_weight, '../weighting/runs'),
            "label": "Normalised IS",
        },
        {
            "data": load_data("FastSVERL-OffPolicy_GWB_Weighting", group_exact, '../weighting/runs'),
            "label": "On-policy",
        },
    ]
}
    

# Loop over each
for plot_spec_func in [
    mastermind_222_policy,
    mastermind_222_value,
    gwb_policy_clipping,
    gwb_value_clipping,
    gwb_policy_weighting,
    gwb_value_weighting,
]:
    
    # Plot
    plot_spec = plot_spec_func()

    plot_spec["xlabel"] = "Training updates"
    plot_spec["legend_loc"] = "upper right"
    plot_spec["fontsize"] = 18
    plot_spec["sem_correction"] = True

    plot(**plot_spec)
