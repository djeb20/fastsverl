from collections import defaultdict, namedtuple
from types import SimpleNamespace
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.distributions import Categorical
import gymnasium as gym
import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import wandb
import matplotlib.patches as mpatches
from scipy.stats import t as t_dist

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_env(env_id, seed, idx, capture_video, run_name, **kwargs):
    """
    Utility function for creating a gym environment, for use with
    vectorized environments.
    """
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", seed=seed, **kwargs)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, seed=seed, **kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        return env

    return thunk

def value_iteration(env, gamma, policy='greedy'):
    """
    Performs value iteration.
    """

    V = defaultdict(float)

    # Perform value iteration
    delta = float('inf')
    while delta > 1e-10:
        delta = 0.
        # Iterate over all states
        for s in env.unwrapped.P:
            v_prev = V[s]
            Q_as = np.array([np.sum([transition[0] * (transition[2] + (1 - transition[3]) * gamma * V[transition[1]]) for transition in env.unwrapped.P[s][a]]) for a in range(env.unwrapped.action_space.n)])
            if policy == 'greedy': 
                V[s] = Q_as.max()
            else:
                V[s] = (policy(env.unwrapped.decode(s)) * Q_as).sum()

            delta = max(delta, abs(v_prev-V[s]))

    # Return state-action values
    return {tuple(env.unwrapped.decode(s)): np.array([np.sum([transition[0] * (transition[2] + (1 - transition[3]) * gamma * V[transition[1]]) 
                                                    for transition in env.unwrapped.P[s][a]]) 
                                                    for a in range(env.unwrapped.action_space.n)]) 
                                                    for s in env.unwrapped.P}

actionTuple = namedtuple('NamedTuple', [
            'action', 
            'logprob' 
            ]
        )

def random_action(obs, env):
    """
    Given a state returns a random action.
    """

    pi_rand = torch.ones(env.single_action_space.n) / env.single_action_space.n
    pi = pi_rand.repeat(obs.shape[0], 1) if obs.ndim > 1 else pi_rand

    policy = Categorical(pi)

    action = policy.sample()

    action_info = actionTuple(
        action=action.cpu().numpy(),
        logprob=policy.log_prob(action)
    )

    return action_info

def load_data(project: str, group: str, base_dir: str = "../runs"):
    """
    Load all scalar tags from a TensorBoard event file into pandas DataFrames.
    Precursor to plotting functions.

    Args:
        run_name (str): Folder name inside base_dir (e.g. '1_1744655871146973393')
        base_dir (str): Base directory containing the run folders (default: "../runs")

    Returns:
        dict[str, pd.DataFrame]: A dictionary mapping each tag to a DataFrame
    """

    print(f"Loading data from {project} - {group}...")

    # Collects run names for a given project and group
    api = wandb.Api()
    run_names = [run.name for run in api.runs(project, filters={"group": group})]

    data = []

    # Iterate over and load each run from local directory
    for run_name in run_names:
    
        # Construct the full path to the run directory
        run_dir = os.path.join(base_dir, run_name)

        # Find the first event file in the run directory
        event_file = next(os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.startswith("events"))

        # Load event data
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()

        # Extract all scalar tags and build a dictionary of DataFrames
        data.append({tag: pd.DataFrame(ea.Scalars(tag)) for tag in ea.Tags()["scalars"]})
    
    return data

def plot(
    groups_data: list[dict],
    tag: str,
    figure_name: str = None,
    figsize: tuple = (6, 4),
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    fontsize: int = None,
    legend_loc: str = "best",
    legend_bbox_to_anchor: tuple = None,
    legend_ncol: int = 1,
    legend: list = None,
    xlim: tuple = None,
    ylim: tuple = None,
    xticks: list = None,
    yticks: list = None,
    n_batches: int = None,
    confidence: float = 0.95,
    fill: str = "stderr",
    sem_correction: bool = False,
):
    """
    Plots scalar data with mean and confidence intervals from multiple runs.
    
    Example usage for outcome Shapley plot:
    
    def mastermind_222_performance(): 
    
        # wandb group name
        group = "shapleys_dqn_performance_mastermind_1746100179"

        return {
        "tag": "model/PerformanceShapley_exact_loss",
        "xlim": (-100, 7000),
        "figure_name": "main_mastermind_222_perf_shapley",
        "x_label": "Training updates",
        "y_label": "MSE",
        "legend_loc": "upper right",
        "fontsize": 16,
        "n_batches": (0.8 * 10_000) // 128,
        "title": "Outcome Shapley",
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

    plot(**plot_spec_func())
    """
    
    fig, ax = plt.subplots(figsize=figsize)

    # Start by merging all runs for each group
    merged_data = []
    steps = []

    for data_dict in groups_data:
        # Process all runs for this group
        dfs = [run_data[tag][["step", "value"]].dropna().reset_index(drop=True) for run_data in data_dict['data']]

        # Shift the step values if 'start_epoch' is specified
        if data_dict.get("start_epoch", False):
            for df in dfs:
                df["step"] += data_dict["start_epoch"]

        # Merge all runs on "step"
        merged = dfs[0].rename(columns={"value": 0})
        for i, df in enumerate(dfs[1:], 1):
            merged = pd.merge_asof(
                merged.sort_values("step"),
                df.rename(columns={"value": i}).sort_values("step"),
                on="step",
                direction="nearest"
            )

        merged = merged.sort_values("step").interpolate(method="linear", axis=0).dropna()

        # Store the merged data
        steps.append(np.array(merged["step"].values))
        merged_data.append(np.array(merged.drop(columns="step").values))

    # Optional: apply SEM correction technique
    if sem_correction:

        # Compute global mean
        global_mean = np.mean(merged_data, axis=(0, 2))

        # Compute treatment mean
        treatment_means = np.mean(merged_data, axis=0)

        # Adjust the data: y_hat_ij = y_ij + y_bar - y_bar_i
        merged_data = [
            merged + global_mean[:, None] - treatment_means
            for merged in merged_data
        ]

    # Plot each group's data
    for step, merged, data_dict in zip(steps, merged_data, groups_data):

        # Compute mean, stderr + confidence interval
        mean = merged.mean(axis=1)
        sem = merged.std(axis=1, ddof=1) / np.sqrt(merged.shape[1])
        ci = sem * t_dist.ppf((1 + confidence) / 2, df=merged.shape[1] - 1)

        # Plot
        ax.plot(step * (n_batches if n_batches else 1), 
                mean, 
                label=data_dict.get('label', None), 
                color=data_dict.get('colour', None), 
                linestyle=data_dict.get('linestyle', '-'))
        
        # Fill between mean +/- stderr
        ax.fill_between(step * (n_batches if n_batches else 1),
                        mean - (sem if fill == "stderr" else ci),
                        mean + (sem if fill == "stderr" else ci),
                        alpha=0.3)
        
        # Plot vertical line to indicate start if 'start_epoch' is specified
        if data_dict.get("start_epoch", False):
            ax.plot([data_dict["start_epoch"] * (n_batches if n_batches else 1)] * 2,
                    [0, mean.max()],
                    linestyle='--', color='gray', linewidth=1, alpha=0.7)

    # Set plot properties
    if xticks:
        ax.set_xticks(xticks)
    if yticks:
        ax.set_yticks(yticks)
    ax.set_xlabel(xlabel, fontsize=fontsize * 1.5 if fontsize else None)
    ax.set_ylabel(ylabel, fontsize=fontsize * 1.5 if fontsize else None)
    ax.set_title(title, fontsize=fontsize * 1.5 if fontsize else None)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.grid(True, alpha=0.3)

    # Create a legend only if there are multiple groups
    if len(groups_data) > 1:
        ax.legend(
            handles=legend if legend else None,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            ncol=legend_ncol,
            fontsize=fontsize,
            frameon=False
        )

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    fig.tight_layout(rect=[0, 0, 1, 0.95] if legend_bbox_to_anchor else None)
    if figure_name:
        fig.savefig(figure_name + ".pdf", transparent=True)
    else:
        plt.show()
    plt.close(fig)



def barcharts(
    groups_data: dict[tuple, list[dict[str, pd.DataFrame]]],
    tags: dict[str, str],
    figure_name: str = None,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    fontsize: int = None,
    ylim: tuple = None,
    xlim: tuple = None,
    bar_width: float = None,
    figsize_per_plot: tuple = (6, 2),  # width, height per plot
):
    """
    Stacked bar charts over (number of states) for different values of d.
    Used for plotting hypercube results.

    Args:
        groups_data: dict mapping (n, d) -> list of runs (each run is a dict of tag -> DataFrame).
        tag: scalar tag to extract losses.
        figure_name: save path for figure.
        xlabel, ylabel, title, xlim, ylim: plot settings.
        bar_width: width of bars.
        figsize_per_plot: (width, height) for each subplot.
    """
    # Group data by d
    d_to_entries = defaultdict(lambda: defaultdict(list))
    ns = np.unique(np.array(list(groups_data))[:, 0])
    
    for (n, d), data_list in groups_data.items():
        for model_idx, (label, tag) in enumerate(tags.items()):
        
            # Get the data for this tag
            steps = np.array([run_data[tag]["step"].iloc[-1:] for run_data in data_list])

            # Save number of states, average steps, and standard error
            d_to_entries[d][label].append((n ** d - 1, steps.mean(), steps.std(ddof=1) / np.sqrt(len(steps))))
    
    # Create the figure
    fig, axes = plt.subplots(
        nrows=len(d_to_entries),
        ncols=1,
        figsize=(figsize_per_plot[0], figsize_per_plot[1] * len(d_to_entries)),
        sharex=True,
    )

    # For distinguishing between different models
    hatches = ['', '////', 'o', 'O', '.', '*']

    for idx, (ax, (d, all_entries)) in enumerate(zip(reversed(axes), d_to_entries.items())):
        for k, (label, entries) in enumerate(all_entries.items()):

            entries = np.array(entries)
            width = bar_width * entries[:, 0] 

            ax.bar(entries[:, 0] + (k - len(all_entries) / 2) * width + width / 2,
                   entries[:, 1], 
                   width=width, 
                   color=plt.cm.viridis((ns[::-1] - 1) / len(ns)),
                   edgecolor="black",
                   hatch=hatches[k],
                   yerr=entries[:, 2],
                   error_kw={"ecolor": "black", "elinewidth": 1, "capsize": 4},
                   )

        if ylim:
            ax.set_ylim(ylim)
        if xlim:
            ax.set_xlim(xlim)

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.tick_params(labelsize=fontsize)
        ax.grid(True, alpha=0.3)

        # Add top-left label for each plot
        ax.text(
            0.02, 0.90, f"Dimension = {d}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=fontsize,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7)
        )

        # Only on the bottom plot: add legends
        if idx == 1:
            legend1 = ax.legend(
                handles=[mpatches.Patch(
                    facecolor=plt.cm.viridis((len(ns) + 2 - n) / len(ns)), 
                    edgecolor="black",
                    label=f"Length = {n}") 
                    for n in ns],
                loc="upper right",
                ncol=1,
                frameon=False,
                fontsize=fontsize,
            )

            # ax.add_artist(legend1)

        if idx == 0:

            ax.legend(
                handles=[mpatches.Patch(
                    hatch=hatches[k], 
                    facecolor="white",
                    edgecolor="black",
                    label=label
                    ) for k, label in enumerate(tags)],
                loc="upper right",
                # bbox_to_anchor=(0.73, 1.0),
                ncol=1,
                frameon=False,
                fontsize=fontsize,
            )

    # Common x label, y label, and title
    fig.suptitle(title, fontsize=fontsize * 1.5 if fontsize else None)
    fig.supxlabel(xlabel, fontsize=fontsize * 1.5 if fontsize else None)
    fig.supylabel(ylabel, fontsize=fontsize * 1.5 if fontsize else None)
    fig.tight_layout(rect=[0, 0, 1, 1])  # leave space for title

    if figure_name:
        fig.savefig(figure_name + ".pdf", transparent=True)
    else:
        plt.show()
    plt.close(fig)

# Helper function for scientific notation formatting
def _format_scientific(mean: float, std_err: float) -> str:
    """
    Formats a mean and standard error into scientific notation for LaTeX.
    Example: (1.234 ± 0.056) × 10^{3}
    Used for generating large-scale experiment summary tables.
    """
    if mean == 0:
        exponent = 0
    else:
        exponent = np.floor(np.log10(abs(mean)))

    scale_factor = 10**exponent
    scaled_mean = mean / scale_factor
    scaled_std_err = std_err / scale_factor

    # The double curly braces {{...}} are used to escape the braces in an f-string
    # so that LaTeX receives the correct superscript format.
    return f"$({scaled_mean:.3f} \\pm {scaled_std_err:.3f}) \\times 10^{{{int(exponent)}}}$"

def calculate_final_stats(groups_data: list[dict], n_batches=None, figure_name=None) -> pd.DataFrame:
    """
    Calculates the mean and standard error of the final value and total steps
    for each group of experiment runs.

    Args:
        groups_data (list[dict]): A list of dictionaries, where each dictionary
                                 represents a group and contains the loaded run data.
        tag (str): The specific scalar tag to analyze from the runs.

    Returns:
        pd.DataFrame: A DataFrame summarizing the statistics, formatted for easy
                      conversion to a LaTeX table.
    """
    summary_results = []

    # Iterate over each experimental group
    for group in groups_data:

        # Collect final values and steps from all runs in the group
        final_values = [run_data.get(group["tag"]).iloc[-1]["value"] for run_data in group["data"]]
        final_steps = [run_data.get(group["tag"]).iloc[-1]["step"] for run_data in group["data"]]
        final_steps = np.array(final_steps) * (n_batches if n_batches else 1)

        # Calculate statistics
        summary_results.append({
            "Group": group["label"],
            "Mean Final Loss": np.mean(final_values),
            "Std-Err Final Loss": np.std(final_values, ddof=1) / np.sqrt(len(final_values)),
            "Mean Updates": np.mean(final_steps),
            "Std-Err Updates": np.std(final_steps, ddof=1) / np.sqrt(len(final_steps)),
        })

    # Convert the results to a DataFrame for easy handling
    stats_df = pd.DataFrame(summary_results)

    # Format the columns into a "mean ± std" string for the final table
    stats_df["Updates"] = stats_df.apply(
        lambda row: _format_scientific(row['Mean Updates'], row['Std-Err Updates']),
        axis=1
    )
    stats_df["Final Loss"] = stats_df.apply(
        lambda row: _format_scientific(row['Mean Final Loss'], row['Std-Err Final Loss']),
        axis=1
    )

    summary_table = stats_df[["Group", "Updates", "Final Loss"]]

    if not figure_name:
        # Display the resulting DataFrame
        print("--- Summary DataFrame ---")
        print(summary_table)
        print("\n" + "="*25 + "\n")

    else:

        # Generate LaTeX code
        # The `index=False` removes the DataFrame index from the output.
        # The `escape=False` ensures that characters like '\' and '$' are rendered correctly.
        latex_code = summary_table.to_latex(index=False, escape=False)

        with open(f"{figure_name}.tex", 'w') as f:
            f.write(latex_code)

class Buffer:
    """
    For storing and retreiving agent experience.
    """

    def __init__(self, buffer_size, envs, *args):

        self.max_size = buffer_size

        # Name, shape and type of each possible buffer element
        self.buffer_info = {
            'obs': ((self.max_size, ) + envs.single_observation_space.shape, torch.float32),
            'action': ((self.max_size, ) + envs.single_action_space.shape, torch.int64),
            'reward': ((self.max_size, ), torch.float32),
            'n_obs': ((self.max_size, ) + envs.single_observation_space.shape, torch.float32),
            'terminated': ((self.max_size, ), torch.float32),
            'truncated': ((self.max_size, ), torch.float32),
            'logprob': ((self.max_size, ), torch.float32),
            'value': ((self.max_size, ), torch.float32),
            'n_value': ((self.max_size, ), torch.float32),
            'advantage': ((self.max_size, ), torch.float32),
            'returns': ((self.max_size, ), torch.float32),
            'e_obs': ((self.max_size, ) + envs.single_observation_space.shape, torch.float32),
            'C': ((self.max_size, ) + envs.single_observation_space.shape, torch.float32),
        }

        # Initiate empty buffer
        self.add_buffer(*args)

        # Track buffer size
        self.size = self.pointer = 0

    def add_buffer(self, *args, **kwargs):
        """
        Adds new element types in buffer.
        """

        if not hasattr(self, 'buffer'): 
            self.buffer = {}

        for buffer_element in args + tuple(kwargs.keys()):
            if buffer_element in self.buffer_info:
                if buffer_element in args:
                    self.buffer[buffer_element] = torch.empty(
                        self.buffer_info[buffer_element][0], 
                        dtype=self.buffer_info[buffer_element][1]
                    ).to(DEVICE)
                else:
                    # Ensure added buffer is of correct shape (flat).
                    self.buffer[buffer_element] = kwargs[buffer_element].view(-1, *self.buffer_info[buffer_element][0][1:])
            else:
                raise KeyError(f'{buffer_element} is not a valid buffer element.')    
            
    def add(self, **kwargs):
        """ 
        Stores experience in the buffer.
        Expects single step of experience from multiple parallel environments.
        Assumes experience is always of shape e.g. values.shape = (num_experience, *obs_shape).
        """

        # Not accounting for multiple identical kwargs keys.
        if not set(kwargs) <= set(self.buffer):
            raise KeyError(f'Keys: {list(kwargs)} do not match buffer elements: {list(self.buffer)}.')
        else:
            for key, values in kwargs.items():

                # Avoid duplicates in index - Torch can't handle this.
                values_t = torch.tensor(values[-self.max_size:], dtype=self.buffer_info[key][1]).to(DEVICE)
                index = torch.arange(self.pointer, self.pointer + values_t.shape[0]) % self.max_size
                self.buffer[key][index] = values_t

            # Update pointer and size
            self.pointer = (self.pointer + values_t.shape[0]) % self.max_size
            self.size = min(self.size + values_t.shape[0], self.max_size)

    def sample(self, batch_size, *args, start=None, replace=True):
        """
        Selects a batch of experience from the buffer.
        """
        # Random sampling
        if start is None: 
            index = np.random.choice(self.size, batch_size, replace=replace)
        
        # Sequential sampling
        else:
            if start + batch_size > self.size:
                raise ValueError('Batch size is greater than buffer size.')
            index = slice(start, start + batch_size)

        return [self.buffer[key][index] for key in args]

    def shuffle(self):
        """
        Shuffles the buffer.
        """

        index = torch.randperm(self.size)
        for value in self.buffer.values():
            value[:self.size] = value[:self.size][index]

    def snip(self, start, end):
        """
        Snips the buffer to a given size.
        """

        if end > self.size or start < 0:
            raise ValueError('Start and end indices must be within buffer size.')
        
        for key in self.buffer:
            self.buffer[key][:end-start] = self.buffer[key][start:end]

        self.size = self.pointer = end - start 
        

# Used to sample coalitions according to the weighted least squares distribution.
class ShapleySampler:
    '''
    For sampling player subsets from the Shapley distribution.

    ADAPTED FROM "Neil Jethani*, Mukund Sudarshan*, 
    Ian Covert*, Su-In Lee, Rajesh Ranganath. 
    "FastSHAP: Real-Time Shapley Value Estimation."

    Args:
      num_players: number of players.
    '''

    def __init__(self, num_players):
        arange = torch.arange(1, num_players)
        w = 1 / (arange * (num_players - arange))
        w = w / torch.sum(w)
        self.categorical = Categorical(probs=w)
        self.num_players = num_players
        self.tril = np.tril(
            np.ones((num_players - 1, num_players), dtype=np.float32), k=0
        )
        self.rng = np.random.default_rng(np.random.randint(0, 2**32))

    def sample(self, batch_size=1, paired_sampling=False):
        '''
        Generate samples from the Shapley distribution.

        Args:
          batch_size: number of samples.
          paired_sampling: whether to use paired sampling.
        '''

        num_included = 1 + self.categorical.sample([batch_size])
        S = self.tril[num_included - 1].reshape(batch_size, -1)
        S = self.rng.permuted(S, axis=1)  # Note: permutes each row.
        if paired_sampling:
            S[1::2] = 1 - S[0:(batch_size - 1):2]  # Note: allows batch_size % 2 == 1.
        return torch.from_numpy(S).to(DEVICE)
    
    def sample_rand(self, batch_size=1):
        """Generates random coalitions uniformly."""
        return torch.randint(2, size=(batch_size, self.num_players)).to(DEVICE)

# Dummy classes for argument parsing
class ExpArgs(SimpleNamespace): pass
class AgentArgs(SimpleNamespace): pass
class EnvArgs(SimpleNamespace): pass
class CharacteristicArgs(SimpleNamespace): pass
class ShapleyArgs(SimpleNamespace): pass
class PolicyCharacteristicArgs(SimpleNamespace): pass
class PerformanceCharacteristicArgs(SimpleNamespace): pass
class SVERLArgs(SimpleNamespace): pass