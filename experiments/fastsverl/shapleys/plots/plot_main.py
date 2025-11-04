"""
Plot main figures for Shapley experiments.
"""

from fastsverl.utils import calculate_final_stats, load_data, plot

# Mastermind-222 Performance
# Set `group` (from W&B) to match the experiment.
# Example: group = "222_shapleys_dqn_performance_mastermind_{timestamp}"
def mastermind_222_performance(): 
    group = None
    return {
    "tag": "model/PerformanceShapley_exact_loss",
    "xlim": (-100, 7000),
    "figure_name": "main_mastermind_222_perf_shapley",
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

# Loop over each
for plot_spec_func in [
    mastermind_222_performance,
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

# Mastermind Large Scale Behaviour
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_443_char = None  # e.g. "443_characteristics_dqn_policy_mastermind_{timestamp}"
group_453_char = None  # e.g. "453_characteristics_dqn_policy_mastermind_{timestamp}"
group_463_char = None  # e.g. "463_characteristics_dqn_policy_mastermind_{timestamp}"
group_443_shap = None  # e.g. "443_shapleys_dqn_policy_mastermind_{timestamp}"
group_453_shap = None  # e.g. "453_shapleys_dqn_policy_mastermind_{timestamp}"
group_463_shap = None  # e.g. "463_shapleys_dqn_policy_mastermind_{timestamp}"

def mastermind_scale_policy_table(): 
    return {
    "figure_name": "main_large_policy_shapley",
    "groups_data": [
        {
            "tag": "model/PolicyCharacteristicModel_epoch_loss",
            "data": load_data("FastSVERL_Mastermind", group_443_char, "../../characteristics/runs/"),
            "label": "Mastermind-443 Characteristic",
        },
        {
            "tag": "model/PolicyShapley_epoch_loss",
            "data": load_data("FastSVERL_Mastermind", group_443_shap),
            "label": "Mastermind-443 Shapley",
        },
        {
            "tag": "model/PolicyCharacteristicModel_epoch_loss",
            "data": load_data("FastSVERL_Mastermind", group_453_char, "../../characteristics/runs/"),
            "label": "Mastermind-453 Characteristic",
        },
        {
            "tag": "model/PolicyShapley_epoch_loss",
            "data": load_data("FastSVERL_Mastermind", group_453_shap),
            "label": "Mastermind-453 Shapley",
        },
        {
            "tag": "model/PolicyCharacteristicModel_epoch_loss",
            "data": load_data("FastSVERL_Mastermind", group_463_char, "../../characteristics/runs/"),
            "label": "Mastermind-463 Characteristic",
        },
        {
            "tag": "model/PolicyShapley_epoch_loss",
            "data": load_data("FastSVERL_Mastermind", group_463_shap),
            "label": "Mastermind-463 Shapley",
        },
        
    ],
}

# Loop over each
for table_spec_func in [
    mastermind_scale_policy_table,
]:

    # Plot
    table_spec = table_spec_func()
    table_spec["n_batches"] = (0.8 * 10_000) // 128,
    calculate_final_stats(**table_spec)