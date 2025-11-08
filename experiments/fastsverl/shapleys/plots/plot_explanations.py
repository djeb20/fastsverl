import json
import os
import pickle
from dataclasses import dataclass
from fastsverl.training import setup_envs
from fastsverl.utils import AgentArgs, EnvArgs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from fastsverl.envs.mastermind import Mastermind
from fastsverl.dqn import DQN
import torch

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Computer Modern Roman"
# })

# Converting state in readable codes.
move_dict = {-1: ' ', 1: 'A', 2: 'B', 3: 'C'}

textcolors = ("black", "white")

@dataclass
class ExpArgs:
    agent_args_f: str = None
    "where to load the agent's and the environment's hyperparameters from"

    # group: str = f"{os.path.basename(os.path.dirname(__file__))}_{os.path.basename(__file__)[: -len('.py')]}_{int(time.time())}"
    # """the group of this experiment"""
    num_runs: int = 1 # 20
    """the number of times to run the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "FastSVERL_Mastermind"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    env_id: str = "Mastermind-v0"
    """the id of the environment"""
    num_envs: int = 1
    """the number of parallel game environments"""

# Run names and corresponding agent args folders
agent_args_fs = [None, None, None]  # Fill in with actual agent args folder names if needed

explanations = {
    'Behaviour': [None, None, None],  # Fill in with actual run names for Behaviour explanations
    'Prediction': [None, None, None],  # Fill in with actual run names for Prediction explanations
    'OnPolicy_Outcome': [None, None, None],  # Fill in with actual run names for On-Policy Outcome explanations
    'OffPolicy_Outcome': [None, None, None]  # Fill in with actual run names for Off-Policy Outcome explanations
}

title_dict = {
    'Behaviour': 'Behaviour',
    'Prediction': 'Prediction',
    'OnPolicy_Outcome': 'On-Policy Outcome',
    'OffPolicy_Outcome': 'Off-Policy Outcome'
}

for explanation, run_names in explanations.items():

    for run_name, agent_args_f in zip(run_names, agent_args_fs):

        # Set up exp args
        exp_args = ExpArgs()
        exp_args.agent_args_f = agent_args_f

        # Set up env
        with open(f"../../agents/runs/{exp_args.agent_args_f}/EnvArgs.json", "r") as f:
            env_args = EnvArgs(**json.load(f))

        # Load in agent
        with open(f"../../agents/runs/{exp_args.agent_args_f}/AgentArgs.json", "r") as f:
            agent_args = AgentArgs(**json.load(f))
        envs = setup_envs(env_args, exp_args, 0, run_name, save_parameters=False, **vars(env_args))
        agent = DQN(envs, agent_args)
        agent.load_models(f"../../agents/runs/{exp_args.agent_args_f}", eval=True, epsilon=True)

        # Load in Shapley values
        with open(f"../runs/{run_name}/train_svs.pkl", 'rb') as file: 
            svs = pickle.load(file)

        for state_idx, (state, sv) in enumerate(svs.items()):

            # Initialise figure
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            fontsize = 15

            # This will store action info if we are plotting a Behaviour explanation
            action_info = None 

            # Behaviour explanation has shapley values for each action.
            if explanation == 'Behaviour':
                # Get the action for the state
                action = agent.choose_action(torch.tensor([state], dtype=torch.float32), exp=False).action[0]

                # Get the numerical index of the guess in the state.
                sv = sv[:, action]
                
                # Find the row index for the action (first non-filled guess)
                # We use the original 1D 'state' here before it's reshaped
                guess_idx = np.where(np.array(state) == -1)[0][0] // (env_args.code_size + 2)
                
                # Account for np.flipud by reversing the index
                flipped_guess_idx = env_args.num_guesses - 1 - guess_idx
                
                # Get the action as letters
                action_as_letters = [move_dict[val] for val in envs.envs[0].unwrapped.index_to_guess[action]]
                
                # Store info for plotting
                action_info = {'row': flipped_guess_idx, 'letters': action_as_letters}                

            # Reshape Shapley values and account for numerical errors with steady-state approximation.
            shapley_values = np.flipud(np.reshape(sv, (env_args.num_guesses, env_args.code_size + 2)))#.round(2)

            # Scale Shapley values between -1 and 1 (if not all zero)
            if shapley_values.max() - shapley_values.min() != 0:
                shapley_values = shapley_values / max(np.abs(shapley_values.max()), np.abs(shapley_values.min()))

            # Reshape state and change code indexes to readable ones.
            state = np.flipud(np.reshape(state, (env_args.num_guesses, env_args.code_size + 2))).astype(object)
            state[:, 1:-1] = np.reshape([move_dict[i] for i in state[:, 1:-1].flatten()], (env_args.num_guesses, env_args.code_size))
            state[:, 0] = [' ' if value == -1 else value for value in state[:, 0]]
            state[:, -1] = [' ' if value == -1 else value for value in state[:, -1]]

            # Create the colourmap (between -1 and 1).
            im = ax.imshow(shapley_values, cmap='RdBu', norm=TwoSlopeNorm(0, -1, 1))

            # Loop over data dimensions and create text annotations.
            for k in range(len(state)):
                for j in range(len(state[0])):
                    text = ax.text(j, k, state[k, j],
                        ha="center", va="center", 
                        color=textcolors[int(abs(shapley_values[k, j]) > 1/2)],
                        fontweight="bold", fontsize=fontsize)
                    
            # If this is a behaviour explanation, overlay the action
            if action_info:
                action_row = action_info['row']
                action_letters = action_info['letters']
                # Action letters go in columns 1 to code_size (e.g., j+1)
                for j, letter in enumerate(action_letters):
                    ax.text(j + 1, action_row, letter,
                            ha="center", va="center", color="#16bd31", # Color from your 2nd example
                            fontweight="bold", fontsize=fontsize)

            # Remove tickmarks
            ax.tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)

            # Turn spines off and create white grid.
            ax.spines[:].set_linewidth(2)

            # Set title, labels and ticks.
            ax.set_xticks(np.arange(state.shape[1]-1)+.5, minor=False)
            ax.set_yticks(np.arange(state.shape[0]-1)+.5, minor=False)
            ax.grid(which="major", color='k', linestyle='-', linewidth=0.5)
            ax.tick_params(which="minor", top=False, bottom=False, left=False, right=False)
            ax.set_title(title_dict[explanation], fontsize=fontsize, pad=8)

            # Add colour bar
            cbar = fig.colorbar(im, ax=ax, shrink=.5, fraction=0.05, orientation="vertical", anchor=(3, 0.5), pad=0)
            cbar.ax.set_yticks([-1, 0, 1])
            cbar.ax.tick_params(labelsize=fontsize)

            fig.supylabel('Shapley Value', x=1.03, fontsize=fontsize, rotation=-90, ha='right')

            # Adjust layout and save figure
            plt.subplots_adjust(bottom=-.1, left=.3)
            plt.tight_layout()

            # Save the figure
            os.makedirs(f'mastermind_explanations/{explanation.lower()}/{env_args.code_size}{env_args.num_guesses}{env_args.num_pegs}', exist_ok=True)
            plt.savefig(f'mastermind_explanations/{explanation.lower()}/{env_args.code_size}{env_args.num_guesses}{env_args.num_pegs}/mastermind_state-{state_idx}.png', bbox_inches='tight')
            plt.close(fig)