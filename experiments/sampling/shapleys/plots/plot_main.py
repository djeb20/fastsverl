"""
Plots for main sampling experiments (Shapleys).
"""

from fastsverl.utils import load_data, plot

# Mastermind-222 Policy
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_wo_exact_model = None  # e.g. "222_wo_exact_model_shapleys_dqn_policy_mastermind_{timestamp}"
group_w_exact_sampling = None  # e.g. "222_w_exact_sampling_shapleys_dqn_policy_mastermind_{timestamp}"
group_w_exact_model = None  # e.g. "222_w_exact_model_shapleys_dqn_policy_mastermind_{timestamp}"

def mastermind_222_policy(): return {
    "tag": "model/PolicyShapley_exact_loss",
    "legend_loc": "upper right",
    "xlim": (-100, 7000),
    "figure_name": "main_mastermind_222_policy_sampling",
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

# Loop over each
for plot_spec_func in [
    mastermind_222_policy,
]:
    
    # Plot
    plot_spec = plot_spec_func()
    plot_spec["xlabel"] = "Training updates"
    plot_spec["fontsize"] = 18

    plot(**plot_spec)