from dataclasses import dataclass
import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

from fastsverl.training import setup_envs
from fastsverl.utils import AgentArgs, EnvArgs
from fastsverl.envs.mastermind import Mastermind
from fastsverl.dqn import DQN

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
    cuda: bool = True
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
    num_envs: int = 16
    """the number of parallel game environments"""

exp_args = ExpArgs()

# --- 1. CONFIGURATION: Adjust these paths and settings ---

# Path to the directory where agent/explanation run data is stored
RUNS_DIR = "../runs/"
AGENTS_DIR = "../../agents/runs/"
OUTPUT_FILENAME = "../../README.md"

# This structure maps your domains to the specific run IDs needed.
# You'll need to fill this out with your actual run/folder names.
DOMAIN_CONFIG = {
    "Mastermind-443": {
        "agent_run_id": "1_1753520362364993335", # Agent for 443
        "explanations": {
            "Behaviour": "1_1754327171814823512",
            # "Performance (On-Policy)": "1_1757851984212157174",
            "Performance": "1_1757852311860287719",
            "Prediction": "1_1757524448855916224",
        }
    },
    "Mastermind-453": {
        "agent_run_id": "1_1753520364367597552", # Agent for 453
        "explanations": {
            "Behaviour": "1_1754327664370701736",
            # "Performance (On-Policy)": "1_1757852799250560205",
            "Performance": "1_1757853130928613104",
            "Prediction": "1_1757527102146519022",
        }
    },
    "Mastermind-463": {
        "agent_run_id": "1_1753520366370971122", # Agent for 463
        "explanations": {
            "Behaviour": "1_1754328163819393596",
            # "Performance (On-Policy)": "1_1757853893495242171",
            "Performance": "1_1757854221711682627",
            "Prediction": "1_1757529730632057966",
        }
    },
}

# --- 2. HELPER FUNCTIONS ---

MOVE_DICT = {-1: ' ', 1: 'A', 2: 'B', 3: 'C'}
TEXT_COLORS = ("black", "white") # For light and dark backgrounds respectively
ACTION_TEXT_COLOR = "#1C8C34" # Green for the chosen action

def load_sv_data(run_id):
    """Loads Shapley value dictionary from a pickle file."""
    path = os.path.join(RUNS_DIR, run_id, "train_svs.pkl")
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_color_from_gradient(value):
    """Maps a Shapley value from [-1, 1] to a hex color string."""
    if not np.isfinite(value): return "#FFFFFF" # Handle NaNs
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    cmap = plt.cm.RdBu # Red-White-Blue colormap
    rgba_color = cmap(norm(value))
    return mcolors.to_hex(rgba_color)

def format_state(state_array, env_args):
    """Converts a raw state array into a human-readable 2D grid."""
    grid = np.flipud(np.reshape(state_array, (env_args.num_guesses, env_args.code_size + 2))).astype(object)
    grid[:, 1:-1] = np.reshape([MOVE_DICT[i] for i in grid[:, 1:-1].flatten()], (env_args.num_guesses, env_args.code_size))
    grid[:, 0] = [' ' if v == -1 else int(v) for v in grid[:, 0]]
    grid[:, -1] = [' ' if v == -1 else int(v) for v in grid[:, -1]]
    return grid

def generate_board_html(state_grid, sv_grid, action_info=None):
    """Generates the HTML for a single colored Mastermind board."""
    html = '<table style="font-family: monospace; text-align: center; border-collapse: collapse;">'
    
    # Header Row
    html += '<tr>'
    headers = ["Clue 1"] + [f"Pos {i+1}" for i in range(state_grid.shape[1] - 2)] + ["Clue 2"]
    for h in headers:
        html += f'<th style="padding: 4px; border: 1px solid #ccc;">{h}</th>'
    html += '</tr>'
    
    # Data Rows
    for i, row in enumerate(state_grid):
        # guess_num = state_grid.shape[0] - i
        html += '<tr>'
        for j, cell_text in enumerate(row):
            sv = sv_grid[i, j]
            bg_color = get_color_from_gradient(sv)
            
            # Determine text color for contrast
            text_color = TEXT_COLORS[int(abs(sv) > 0.5)]
            
            # --- CHANGES ARE HERE ---
            # is_action_row = False
            is_action_row = action_info is not None and state_grid.shape[0] - i - 1 == action_info['guess_idx']

            # If this is the action row, override the text with the action letters
            if is_action_row and 1 <= j < state_grid.shape[1] - 1:

                display_text = action_info['letters'][j - 1]
                text_color = ACTION_TEXT_COLOR
            
            # Use a non-breaking space for empty cells to ensure consistent row height
            elif str(cell_text).strip() == '':
                display_text = '&nbsp;'

            else:
                display_text = cell_text

            style = f'background-color: {bg_color}; color: {text_color}; padding: 4px; border: 1px solid #ccc; font-weight: bold;'
            html += f'<td style="{style}">{display_text}</td>'
        html += '</tr>'
        
    html += '</table>'
    return html

# --- 3. MAIN SCRIPT LOGIC ---

intro_text = """
# FastSVERL Explanations for Mastermind

This document presents qualitative examples of Shapley value explanations generated using FastSVERL for trained DQN agents in various Mastermind domains.

Mastermind is a code-breaking game where, in these versions, an agent must guess a hidden 4-letter code, drawn from a 3-letter alphabet, within a limited number of turns. After each guess, the agent receives two clues: *Clue 2* for the number of correct letters in the correct position, and *Clue 1* for the number of correct letters in the wrong position. Full details of gameplay are provided in the paper.

## How to Read the Visualisations

For each state, three explanation types are shown side-by-side: *Behaviour*, *Performance*, and *Prediction*. The colour of each cell on the board represents its relative Shapley value, which indicates the feature's contribution to the explanation type.

* *Blue* cells indicate a *positive* contribution.
* *Red* cells indicate a *negative* contribution.
* The *intensity* of the colour corresponds to the magnitude of the influence.
* For the *Behaviour* explanation, the agent's next chosen action is marked in *green* for reference.

Full details of the FastSVERL methodology are available in the main paper.

## Navigating This Document

This document is organised by domain size. A table of contents at the top provides direct links to each domain section. Within each section, a grid of links allows you to jump to a specific state.

Please note that the state indices are for navigational purposes only and do not represent a sequential trajectory. The states presented are representative of those encountered by an optimal policy in each domain.
"""

def main():
    """Main function to generate the Markdown file."""
    md_content = [intro_text]

    # --- Create Header and Top-Level ToC ---
    md_content.append("\n## Experimental Domains")
    for domain_name in DOMAIN_CONFIG:
        anchor = domain_name.lower().replace(" ", "-")
        md_content.append(f"* [{domain_name}](#{anchor})")

    # --- Loop Through Each Domain ---
    for domain_name, config in DOMAIN_CONFIG.items():
        print(f"Processing domain: {domain_name}...")
        domain_anchor = domain_name.lower().replace(" ", "-")
        md_content.append(f"\n<br>\n\n---\n## {domain_name}")

        # --- Load Agent and Env Config for this domain ---
        with open(os.path.join(AGENTS_DIR, config['agent_run_id'], "EnvArgs.json"), "r") as f:
            env_args = EnvArgs(**json.load(f))
        with open(os.path.join(AGENTS_DIR, config['agent_run_id'], "AgentArgs.json"), "r") as f:
            agent_args = AgentArgs(**json.load(f))
        exp_args.agent_args_f = config['agent_run_id']
        
        envs = setup_envs(env_args, exp_args, 0, run_name=None, save_parameters=False, **vars(env_args))
        agent = DQN(envs, agent_args)
        agent.load_models(os.path.join(AGENTS_DIR, config['agent_run_id']), eval=True, epsilon=True)
        
        # --- Load all explanation data for this domain ---
        explanation_data = {name: load_sv_data(run_id) for name, run_id in config['explanations'].items()}
        
        # Use the first explanation's keys to define the state order
        state_keys = list(explanation_data["Behaviour"].keys())

        # --- Create State-Level ToC Grid (New Table-based version) ---
        md_content.append("\n### Jump to State")
        links = [f"[{i + 1}](#state-{i + 1}-{domain_anchor})" for i in range(len(state_keys))]
        num_columns = 10

        # Create the table header
        header = "| " + " | ".join([" "] * num_columns) + " |"
        separator = "|-" + "-|-".join(["-"] * num_columns) + "-|"
        md_content.append(header)
        md_content.append(separator)

        # Create the table rows
        current_row = []
        for i, link in enumerate(links):
            current_row.append(link)
            if (i + 1) % num_columns == 0:
                md_content.append("| " + " | ".join(current_row) + " |")
                current_row = []

        # Add the last row if it's not full
        if current_row:
            # Pad the last row to have the correct number of columns
            while len(current_row) < num_columns:
                current_row.append(" ")
            md_content.append("| " + " | ".join(current_row) + " |")
        

        # --- Loop Through Each State in the Domain ---
        for i, state_key in enumerate(state_keys):
            print(f"  - Generating state {i + 1}...")
            state_anchor = f"state-{i + 1}-{domain_anchor}"
            md_content.append(f"\n---\n<h3 id='{state_anchor}'>State {i + 1}</h3>")

            state_array = np.array(state_key)
            state_grid = format_state(state_array, env_args)

            # --- Generate the 4-column side-by-side table ---
            md_content.append("<table>")
            # Header row
            md_content.append("<tr>")
            for name in config['explanations']:
                md_content.append(f"<th>{name}</th>")
            md_content.append("</tr>")
            
            # Content row
            md_content.append("<tr>")
            for name, data in explanation_data.items():
                sv = data[state_key]
                action_info = None
                
                # Special handling for Behaviour
                if name == "Behaviour":
                    action = agent.choose_action(torch.tensor([state_array], dtype=torch.float32), exp=False).action[0]
                    sv = sv[:, action]
                    
                    # Find which guess slot this action corresponds to
                    guess_idx = np.where(state_array == -1)[0][0] // (env_args.code_size + 2)

                    action_as_letters = [MOVE_DICT[val] for val in envs.envs[0].unwrapped.index_to_guess[action]]
                    action_info = {'guess_idx': guess_idx, 'letters': action_as_letters}

                # Reshape and normalize SVs
                sv_grid = np.flipud(np.reshape(sv, (env_args.num_guesses, env_args.code_size + 2)))
                if sv_grid.max() - sv_grid.min() != 0:
                    sv_grid = sv_grid / max(abs(sv_grid.max()), abs(sv_grid.min()))

                board_html = generate_board_html(state_grid, sv_grid, action_info)
                md_content.append(f'<td valign="top">{board_html}</td>')
            
            md_content.append("</tr>")
            md_content.append("</table>")
    
    # --- Write to file ---
    with open(OUTPUT_FILENAME, "w") as f:
        f.write("\n".join(md_content))
    
    print(f"\nSuccessfully generated {OUTPUT_FILENAME}!")


if __name__ == "__main__":
    main()
