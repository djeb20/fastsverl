"""
Plot appendix characteristics for sampling experiments.
"""

from fastsverl.utils import load_data, plot

# GWB Performance
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_wo_exact_model = None  # e.g. "wo_exact_model_characteristics_dqn_performance_gwb_{timestamp}"
group_w_exact_sampling = None  # e.g. "w_exact_sampling_characteristics_dqn_performance_gwb_{timestamp}"
group_w_exact_model = None  # e.g. "w_exact_model_characteristics_dqn_performance_gwb_{timestamp}"

def gwb_performance(): return {
    "tag": "model/PerformanceCharacteristic_exact_loss",
    "figure_name": "app_gwb_perf_sampling",
    "title": "Outcome characteristic",
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
            "start_epoch": 50 * (0.8 * 10_000) // 128,
        },
    ]
}

# Mastermind-222 Performance
# Set group names (from W&B) to match the experiments.
# Examples shown below.
group_wo_exact_model = None  # e.g. "222_wo_exact_model_characteristics_dqn_performance_mastermind_{timestamp}" 
group_w_exact_sampling = None  # e.g. "222_w_exact_sampling_characteristics_dqn_performance_mastermind_{timestamp}"
group_w_exact_model = None  # e.g. "222_w_exact_model_characteristics_dqn_performance_mastermind_{timestamp}"

def mastermind_222_performance(): return {
    "tag": "model/PerformanceCharacteristic_exact_loss",
    "figure_name": "app_mastermind_222_perf_sampling",
    "title": "Outcome characteristic",
    "ylabel": "MSE",
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
            "start_epoch": 50 * (0.8 * 10_000) // 128,
        },
    ]
}

# Loop over each
for plot_spec_func in [
    gwb_performance,
    mastermind_222_performance,
]:

    # Plot
    plot_spec = plot_spec_func()
    plot_spec["xlabel"] = "Training updates"
    plot_spec["legend_loc"] = "upper right"
    plot_spec["fontsize"] = 18

    plot(**plot_spec)
