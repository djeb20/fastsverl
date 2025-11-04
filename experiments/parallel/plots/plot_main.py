"""
Plot main results for parallel experiments.
"""

from fastsverl.utils import load_data, plot

# Mastermind-222 Value Update Rate; Approx Characteristic
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_1_w_exact = None  # e.g. "1_w_exact_parallel_training_dqn_value_mastermind_{timestamp}"
group_2_w_exact = None  # e.g. "2_w_exact_update_rate_dqn_value_mastermind_{timestamp}"
group_10_w_exact = None  # e.g. "10_w_exact_update_rate_dqn_value_mastermind_{timestamp}"
group_50_w_exact = None  # e.g. "50_w_exact_update_rate_dqn_value_mastermind_{timestamp}"

def mastermind_222_value_update_rate_approx(): return {
    "tag": "agent/ValueShapley_exact_loss",
    "ylim": (-.0005, 0.023),
    "figure_name": "main_mastermind_222_parallel_value_approx",
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

# Loop over each
for plot_spec_func in [
    mastermind_222_value_update_rate_approx,
]:
    
    # Plot
    plot_spec = plot_spec_func()
    plot_spec["xlabel"] = "Training updates"
    plot_spec["legend_loc"] = "upper right"
    plot_spec["fontsize"] = 18
    plot_spec["n_batches"] = 0.1 # Scaling because models updated every 10 env steps.

    plot(**plot_spec)