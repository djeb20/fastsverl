"""
Trains off-policy behaviour characteristic models with different weighting strategies in GWB. 
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
    wandb_project_name: str = "FastSVERL-OffPolicy_GWB_Weighting"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    env_id: str = "GWB-v0"
    """the id of the environment"""
    num_envs: int = 8
    """the number of parallel game environments"""

@dataclass
class AgentArgs:
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    critic_arch: dict = field(default_factory=lambda: {
        'input': ['Linear', [None, 64]],
        'input_activation': ['ReLU', []],
        'hidden1': ['Linear', [64, 64]],
        'hidden1_activation': ['ReLU', []],
        'output': ['Linear', [64, None]],
    })
    """the critic's neural network architecture"""
    total_timesteps: int = 15_000
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
    learning_starts: int = 10_000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    eval_episodes: int = 0
    """the number of episodes to evaluate the agent"""
    verbose: int = 0
    """the verbosity of the agent training"""

@dataclass
class CharacteristicArgs:
    off_policy: bool = None
    """whether to train the characteristic off-policy"""
    weighting: bool = None
    """whether to weight the importance sampling"""

    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    model_arch: dict = field(default_factory=lambda: {
        'input': ['Linear', [None, 64]],
        'input_activation': ['ReLU', []],
        'hidden1': ['Linear', [64, 64]],
        'hidden1_activation': ['ReLU', []],
        'output': ['Linear', [64, None]],
        'output_activation': ['Softmax', [-1]],
    })
    """the characteristic's neural network architecture"""
    epochs: int = 50
    """the number of epochs to train the model"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    buffer_size: int = AgentArgs.total_timesteps
    """the replay memory buffer size"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    val_fraction: float = 0.2
    """the fraction of data to be used for validation"""
    train_buffer_size: int = int(buffer_size * (1 - val_fraction))
    """the size of the training buffer"""
    val_buffer_size: int = int(buffer_size * val_fraction)
    """the size of the validation buffer"""
    patience: int = epochs
    """the number of epochs to wait before early stopping"""
    verbose: int = 0
    """the verbosity of characteristic training"""
    exact_loss: bool = True
    """whether to record the exact loss"""
    exact_buffer_size: int = train_buffer_size
    """the size of the buffer used to compute the exact loss"""
    policy: str = 'greedy'
    """the policy to explain"""

@dataclass
class EnvArgs:
    pass

def run():

    processes = []

    for seed in tqdm(range(1, exp_args.num_runs + 1), desc="Runs"):
        time.sleep(2) # Separate identifiers for each run
        processes.append(subprocess.Popen([
            "python", "../run_policy_prediction.py", 
            str(seed),
            json.dumps(exp_args.__dict__),
            json.dumps(AgentArgs().__dict__),
            json.dumps(EnvArgs().__dict__),
            json.dumps(char_args.__dict__),
            "DQN",
            "PolicyCharacteristicModel",
            ], preexec_fn=os.setsid))
    
    # Wait for all processes to finish
    for process in processes:
        process.wait()

if __name__ == "__main__":

    exp_args = ExpArgs()
    char_args = CharacteristicArgs()

    # Accurate steady-state
    exp_args.group = f"exact_{ExpArgs.group}"
    char_args.off_policy = False
    run()

    # Buffered steady-state without importance sampling
    exp_args.group = f"wo_weight_{ExpArgs.group}"
    char_args.off_policy = False
    run()

    # Buffered steady-state with importance sampling (no weighting)
    exp_args.group = f"w_weight_{ExpArgs.group}"
    char_args.off_policy = True
    char_args.weighting = False
    run()

    # Buffered steady-state with importance sampling (weighting)
    exp_args.group = f"weighted_weight_{ExpArgs.group}"
    char_args.off_policy = True
    char_args.weighting = True
    run()