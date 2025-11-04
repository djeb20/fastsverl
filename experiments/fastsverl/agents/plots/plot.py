"""
Plotting scripts for FastSVERL agents' learning curves.
"""

from fastsverl.utils import load_data, plot

# GWB
# Set `group_name` (from W&B) to match the experiment.
# Example: group_name = "agents_dqn_gwb_{timestamp}"
group_name = None

def gwb(): return {
    "tag": "agent/episodic_return",
    "title": "GWB Episodic Return",
    "xlabel": "Environment Steps",
    "ylabel": "Episodic Return",
    "legend_loc": "lower right",
    "figure_name": "gwb_episodic_return",
    "groups_data": {
        "GWB": load_data("FastSVERL_GWB", group_name),
    },
}

# Masterind-222
# Set `group_name` (from W&B) to match the experiment.
# Example: group_name = "222_agents_dqn_mastermind_{timestamp}"
group_name = None

def mastermind_222(): return {
    "tag": "agent/episodic_return",
    "title": "Mastermind-222 Episodic Return",
    "xlabel": "Environment Steps",
    "ylabel": "Episodic Return",
    "legend_loc": "lower right",
    "figure_name": "mastermind_222_episodic_return",
    "groups_data": {
        "Mastermind-222": load_data("FastSVERL_Mastermind", group_name),
    },
}

# Mastermind-333
# Set `group_name` (from W&B) to match the experiment.
# Example: group_name = f"333_agents_dqn_mastermind_{timestamp}"
group_name = None

def mastermind_333(): return {
    "tag": "agent/episodic_return",
    "title": "Mastermind-333 Episodic Return",
    "xlabel": "Environment Steps",
    "ylabel": "Episodic Return",
    "legend_loc": "lower right",
    "figure_name": "mastermind_333_episodic_return",
    "groups_data": {
        "Mastermind-333": load_data("FastSVERL_Mastermind", group_name),
    },
}

# Loop over each
for plot_spec in [
    gwb,
    mastermind_222,
    mastermind_333,
]:

    # Plot
    plot(**plot_spec())