"""
Training a single DQN agent for each configuration of the Mastermind environment.
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
    num_runs: int = 1
    """the number of times to run the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "FastSVERL-Optimising_Stochastic_Mastermind"
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
    critic_arch: dict = None
    """the critic's neural network architecture"""
    total_timesteps: int = None
    """total timesteps of the experiments"""
    buffer_size: int = None
    """the replay memory buffer size"""

    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    save_buffer: bool = True
    """whether to save buffer into the `runs/{run_name}` folder"""
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
    learning_starts: int = 10_000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    eval_episodes: int = 0
    """the number of episodes to evaluate the agent"""
    verbose: int = 0
    """the verbosity of the agent training"""

@dataclass
class EnvArgs:
    code_size: int = None
    """the code length"""
    num_guesses: int = None
    """the number of guesses"""
    num_pegs: int = None
    """the number of pegs"""

def run():
    time.sleep(2) # Separate identifiers for each run
    return subprocess.Popen([
        "python", "run.py",
        str(seed),
        json.dumps(exp_args.__dict__),
        json.dumps(agent_args.__dict__),
        json.dumps(env_args.__dict__),
        "DQN"
        ], preexec_fn=os.setsid)

if __name__ == "__main__":

    exp_args = ExpArgs()
    agent_args = AgentArgs()
    env_args = EnvArgs()

    for seed in tqdm(range(1, exp_args.num_runs + 1), desc="Runs"):

        processes = []

        # ---- Mastermind-222 ----
        exp_args.group = f"222_{ExpArgs.group}"

        # Small agent
        agent_args.critic_arch = {
            'input': ['Linear', [None, 64]],
            'input_activation': ['ReLU', []],
            'hidden1': ['Linear', [64, 64]],
            'hidden1_activation': ['ReLU', []],
            'output': ['Linear', [64, None]],
        }
        agent_args.total_timesteps = 100_000
        agent_args.buffer_size = agent_args.total_timesteps

        # Small environment: 8 features, 4 actions, 53 states
        env_args.code_size = 2
        env_args.num_guesses = 2
        env_args.num_pegs = 2
        
        # Run the experiment
        processes.append(run())

        # ---- Mastermind-333 ----
        exp_args.group = f"333_{ExpArgs.group}"

        # Large agent     
        agent_args.critic_arch = {
            'input': ['Linear', [None, 128]],
            'input_activation': ['ReLU', []],
            'hidden1': ['Linear', [128, 128]],
            'hidden1_activation': ['ReLU', []],
            'output': ['Linear', [128, None]],
        }
        agent_args.total_timesteps = 1_000_000
        agent_args.buffer_size = agent_args.total_timesteps

        # Big environment: 15 features, 27 actions, >= 104731 states
        env_args.code_size = 3
        env_args.num_guesses = 3
        env_args.num_pegs = 3

        # Run the experiment
        processes.append(run())

        # ----------- Non exact (new experiments) ---------------
        # Longer training, but same critic arch.
        agent_args.total_timesteps = 10_000_000
        agent_args.buffer_size = agent_args.total_timesteps

        # 24 features, 81 actions, >= 43_046_721 states
        exp_args.group = f"443_{ExpArgs.group}"
        env_args.code_size = 4
        env_args.num_guesses = 4
        env_args.num_pegs = 3
        processes.append(run())

        # 30 features, 81 actions, >= 3_486_784_401 states
        exp_args.group = f"453_{ExpArgs.group}"
        env_args.code_size = 4
        env_args.num_guesses = 5
        env_args.num_pegs = 3
        processes.append(run())

        # 36 features, 81 actions, >= 282_429_536_481 states
        exp_args.group = f"463_{ExpArgs.group}"
        env_args.code_size = 4
        env_args.num_guesses = 6
        env_args.num_pegs = 3
        processes.append(run())

        # Wait for all processes to finish
        for process in processes:
            process.wait()