"""
Plots for the main off-policy experiments.
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
    "figure_name": "main_mastermind_222_offpolicy_policy",
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

# Loop over each
for plot_spec_func in [
    mastermind_222_policy,
]:
    
    # Plot
    plot_spec = plot_spec_func()

    plot_spec["xlabel"] = "Training updates"
    plot_spec["legend_loc"] = "upper right"
    plot_spec["fontsize"] = 18
    plot_spec["sem_correction"] = True

    plot(**plot_spec)
