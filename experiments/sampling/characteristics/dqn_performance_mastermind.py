"""
Trains outcome characteristic models for a given agent in the Mastermind environment,
comparing model-based and sampling-based approaches, with and without access to the exact behaviour characteristic.
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
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
    agent_args_f: str = None # Fill in wandb run name for Mastermind-222 agent here.
    """where to load the agent's and the environment's hyperparameters from"""

@dataclass
class PolicyCharacteristicArgs:
    # ----- Shared -----
    policy: str = 'greedy'
    """the policy to explain"""

    # ----- Sampling -----
    num_samples: int = 1
    """the number of samples to approximate the characteristic with"""
    pool_size: int = None
    """the size of the pool to sample from"""

    # ----- Model -----
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
    buffer_size: int = 10_000
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
    verbose: int = 1
    """the verbosity of characteristic training"""
    exact_loss: bool = True
    """whether to record the exact loss"""
    track_batch: bool = True
    """whether to save epoch losses at batch granularity"""
    save_batch: bool = True
    """whether to save batch losses"""

@dataclass
class PerformanceCharacteristicArgs:
    with_exact_char: bool = None
    """whether to use the exact policy characteristic when computing the exact loss"""

    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    model_arch: dict = field(default_factory=lambda: {
        'input': ['Linear', [None, 64]],
        'input_activation': ['ReLU', []],
        'hidden1': ['Linear', [64, 64]],
        'hidden1_activation': ['ReLU', []],
        'output': ['Linear', [64, None]],
    })
    """the characteristic's neural network architecture"""
    total_timesteps: int = 1_000_000
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
    start_e: float = None
    """the starting epsilon for exploration"""
    train_frequency: int = 1
    """the frequency of training"""
    eval_episodes: int = 0
    """the number of episodes to evaluate the agent"""
    verbose: int = 1
    """the verbosity of characteristic training"""
    sampling_frequency: str = "decision"
    """the frequency of sampling a new s_e and C"""
    exact_loss: bool = True
    """whether to record the exact loss"""

def run(policy_char):

    processes = []

    for seed in tqdm(range(1, exp_args.num_runs + 1), desc="Runs"):

        time.sleep(2) # Separate identifiers for each run
        processes.append(subprocess.Popen([
            "python", "run_performance.py",
            str(seed),
            json.dumps(exp_args.__dict__),
            json.dumps(policy_char_args.__dict__),
            "DQN",
            json.dumps(char_args.__dict__),
            policy_char,
            ], preexec_fn=os.setsid))
    
    # Wait for all processes to finish
    for process in processes:
        process.wait()

if __name__ == "__main__":

    exp_args = ExpArgs()
    policy_char_args = PolicyCharacteristicArgs()
    char_args = PerformanceCharacteristicArgs()

    # Sample from the whole buffer
    policy_char_args.pool_size = policy_char_args.train_buffer_size

    # ------------- With exact policy char -------------
    char_args.with_exact_char = True

    # Model version
    exp_args.group = f"222_w_exact_model_{ExpArgs.group}"
    run('PolicyCharacteristicModel')

    # Sampling version
    exp_args.group = f"222_w_exact_sampling_{ExpArgs.group}"
    run('PolicyCharacteristicSample')

    # ------------- Without exact policy char (only model version makes sense) -------------
    char_args.with_exact_char = False
    exp_args.group = f"222_wo_exact_model_{ExpArgs.group}"
    run('PolicyCharacteristicModel')