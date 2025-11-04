"""
Trains combined DQN, behaviour characteristic and Shapley models on the GWB environment.
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from tqdm import tqdm

from fastsverl.envs.gwb import GWB

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
    wandb_project_name: str = "FastSVERL-Optimising_XDQN_GWB"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    env_id: str = "GWB-v0"
    """the id of the environment"""
    num_envs: int = 16
    """the number of parallel game environments"""

@dataclass
class AgentArgs:
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    shared_arch: dict = field(default_factory=lambda: {
        'input': ['Linear', [None, 64]],
        'input_activation': ['ReLU', []],
        'hidden1': ['Linear', [64, 64]],
        'hidden1_activation': ['ReLU', []],
    })
    """the shared layers' neural network architecture"""
    critic_arch: dict = field(default_factory=lambda: {
        'critic_hidden1': ['Linear', [64, 64]],
        'critic_hidden1_activation': ['ReLU', []],
        'output': ['Linear', [64, None]],
    })
    """the critic's neural network architecture"""
    total_timesteps: int = 100_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    buffer_size: int = total_timesteps
    """the replay memory buffer size"""
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
    exploration_fraction: float = 0.5
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
    exact_frequency: int = 1_000
    """the frequency of computing the exact loss"""
    exact_buffer_size: int = 10_000
    """the size of the buffer used to compute the exact loss"""

@dataclass
class CharacteristicArgs:
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    model_arch: dict = field(default_factory=lambda: {
        'hidden1': ['Linear', [64, 64]],
        'hidden1_activation': ['ReLU', []],
        'output': ['Linear', [64, None]],
        'output_activation': ['Softmax', [-1]],
    })
    """the characteristic's neural network architecture"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    exact_loss: bool = True
    """whether to record the exact loss"""
    policy: str = 'explore'
    """the policy to explain"""
    off_policy: bool = True
    """whether to use off-policy learning"""

@dataclass
class ShapleyArgs:
    with_exact_char: bool = None
    """whether to use the exact characteristic when computing the exact loss"""

    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    model_arch: dict = field(default_factory=lambda: {
        'hidden1': ['Linear', [64, 64]],
        'hidden1_activation': ['ReLU', []],
        'output': ['Linear', [64, None]],
    })
    """the characteristic's neural network architecture"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    exact_loss: bool = True
    """whether to record the exact loss"""

@dataclass
class EnvArgs:
    pass

def run(agent):
    time.sleep(2) # Separate identifiers for each run
    return subprocess.Popen([
        "python", "run_policy_prediction.py", 
        str(seed),
        json.dumps(exp_args.__dict__),
        json.dumps(AgentArgs().__dict__),
        json.dumps(EnvArgs().__dict__),
        json.dumps(CharacteristicArgs().__dict__),
        json.dumps(shapley_args.__dict__),
        "PolicyCharacteristicModel",
        "PolicyShapley",
        agent,
    ], preexec_fn=os.setsid)

if __name__ == "__main__":

    exp_args = ExpArgs()
    shapley_args = ShapleyArgs()

    for seed in tqdm(range(1, exp_args.num_runs + 1), desc="Runs"):
    
        processes = []

        # ------- Exact char -------
        shapley_args.with_exact_char = True

        # All separate
        exp_args.group = f"w_exact_dqn_{ExpArgs.group}"
        processes.append(run('DQN'))

        # Combined characteristic
        exp_args.group = f"w_exact_dqn_c_{ExpArgs.group}"
        processes.append(run('XDQN_C'))

        # Combined Shapley
        exp_args.group = f"w_exact_dqn_s_{ExpArgs.group}"
        processes.append(run('XDQN_S'))

        # ------- Without exact char -------
        shapley_args.with_exact_char = False

        # All separate
        exp_args.group = f"wo_exact_dqn_{ExpArgs.group}"
        processes.append(run('DQN'))

        # Combined characteristic
        exp_args.group = f"wo_exact_dqn_c_{ExpArgs.group}"
        processes.append(run('XDQN_C'))

        # Combined Shapley
        exp_args.group = f"wo_exact_dqn_s_{ExpArgs.group}"
        processes.append(run('XDQN_S'))

        # Wait for all processes to finish
        for process in processes:
            process.wait()