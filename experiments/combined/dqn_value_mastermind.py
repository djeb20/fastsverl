"""
Trains combined DQN, prediction characteristic and Shapley models on the Mastermind environment.
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass
from tqdm import tqdm

from fastsverl.envs.mastermind import Mastermind

os.environ["MKL_THREADING_LAYER"] = "GNU"

@dataclass
class ExpArgs:
    group: str = f"{os.path.basename(os.path.dirname(__file__))}_{os.path.basename(__file__)[: -len('.py')]}_{int(time.time())}"
    """the group of this experiment"""
    num_runs: int = 20
    """the number of times to run the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "FastSVERL-Optimising_XDQN_Mastermind"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    env_id: str = "Mastermind-v0"
    """the id of the environment"""
    num_envs: int = 16
    """the number of parallel game environments"""

@dataclass
class AgentArgs:
    shared_arch: dict = None
    """the shared layers' neural network architecture"""
    critic_arch: dict = None
    """the critic's neural network architecture"""
    total_timesteps: int = None
    """total timesteps of the experiments"""
    buffer_size: int = None
    """the replay memory buffer size"""

    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    gamma: float = 1
    """the discount factor gamma"""
    tau: float = 0.01
    """the target network update rate"""
    target_network_frequency: int = 1
    """the timesteps it takes to update the target network"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.25
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 0
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    eval_episodes: int = 0
    """the number of episodes to evaluate the agent"""
    verbose: int = 0
    """the verbosity of the agent training"""

    # For characteristic and Shapley
    exact_frequency: int = None
    """the frequency of computing the exact loss"""
    exact_buffer_size: int = None
    """the size of the buffer used to compute the exact loss"""

@dataclass
class CharacteristicArgs:
    model_arch: dict = None
    """the characteristic's neural network architecture"""
    exact_loss: bool = None
    """whether to record the exact loss"""

    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    policy: str = 'explore'
    """the policy to explain"""
    off_policy: bool = True
    """whether to use off-policy learning"""

@dataclass
class ShapleyArgs:
    with_exact_char: bool = None
    """whether to use the exact characteristic when computing the exact loss"""
    model_arch: dict = None
    """the Shapley neural network architecture"""
    exact_loss: bool = None
    """whether to record the exact loss"""

    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""

@dataclass
class EnvArgs:
    code_size: int = None
    """the code length"""
    num_guesses: int = None
    """the number of guesses"""
    num_pegs: int = None
    """the number of pegs"""

def run(agent):
    time.sleep(2) # Separate identifiers for each run
    return subprocess.Popen([
        "python", "run_policy_prediction.py", 
        str(seed),
        json.dumps(exp_args.__dict__),
        json.dumps(agent_args.__dict__),
        json.dumps(env_args.__dict__),
        json.dumps(char_args.__dict__),
        json.dumps(shapley_args.__dict__),
        "ValueCharacteristicModel",
        "ValueShapley",
        agent,
    ], preexec_fn=os.setsid)

if __name__ == "__main__":

    exp_args = ExpArgs()
    agent_args = AgentArgs()
    char_args = CharacteristicArgs()
    shapley_args = ShapleyArgs()
    env_args = EnvArgs()

    for seed in tqdm(range(1, exp_args.num_runs + 1), desc="Runs"):
        
        processes = []

        # --------- Exact loss (2 2 2) --------- #

        # Small agent
        agent_args.shared_arch = {
            'input': ['Linear', [None, 64]],
            'input_activation': ['ReLU', []],
            'hidden1': ['Linear', [64, 64]],
            'hidden1_activation': ['ReLU', []],
        }
        agent_args.critic_arch = {
            'critic_hidden1': ['Linear', [64, 64]],
            'critic_hidden1_activation': ['ReLU', []],
            'output': ['Linear', [64, None]],
        }
        agent_args.total_timesteps = 100_000
        agent_args.buffer_size = agent_args.total_timesteps
        agent_args.exact_frequency = 1_000
        agent_args.exact_buffer_size = 10_000

        # Small characteristic
        char_args.model_arch = {
            'hidden1': ['Linear', [64, 64]],
            'hidden1_activation': ['ReLU', []],
            'output': ['Linear', [64, None]],
        }
        char_args.exact_loss = True

        # Small Shapley
        shapley_args.model_arch = {
            'hidden1': ['Linear', [64, 64]],
            'hidden1_activation': ['ReLU', []],
            'output': ['Linear', [64, None]],
        }
        shapley_args.exact_loss = True

        # Small environment: 8 features, 4 actions, 53 states
        env_args.code_size = 2
        env_args.num_guesses = 2
        env_args.num_pegs = 2
        
        # ---- With exact char ---- #
        shapley_args.with_exact_char = True

        # All separate
        exp_args.group = f"222_w_exact_dqn_{ExpArgs.group}"
        processes.append(run('DQN'))

        # Combined characteristic
        exp_args.group = f"222_w_exact_dqn_c_{ExpArgs.group}"
        processes.append(run('XDQN_C'))

        # Combined Shapley
        exp_args.group = f"222_w_exact_dqn_s_{ExpArgs.group}"
        processes.append(run('XDQN_S'))

        # ---- Without exact char ---- #
        shapley_args.with_exact_char = False

        # All separate
        exp_args.group = f"222_wo_exact_dqn_{ExpArgs.group}"
        processes.append(run('DQN'))

        # Combined characteristic
        exp_args.group = f"222_wo_exact_dqn_c_{ExpArgs.group}"
        processes.append(run('XDQN_C'))

        # Combined Shapley
        exp_args.group = f"222_wo_exact_dqn_s_{ExpArgs.group}"
        processes.append(run('XDQN_S'))

        # Wait for all processes to finish
        for process in processes:
            process.wait()