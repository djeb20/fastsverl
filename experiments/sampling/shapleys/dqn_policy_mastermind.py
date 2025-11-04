"""
Trains behaviour Shapley models for DQN agents on the Mastermind environment,
comparing model-based and sampling-based approaches.
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
    num_runs: int = None
    """the number of times to run the experiment"""
    agent_args_f: str = None
    "where to load the agent's and the environment's hyperparameters from"

    group: str = f"{os.path.basename(os.path.dirname(__file__))}_{os.path.basename(__file__)[: -len('.py')]}_{int(time.time())}"
    """the group of this experiment"""
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
class CharacteristicArgs:
    # ----- Shared -----
    policy: str = 'greedy'
    """the policy to explain"""

    # ----- Sampling -----
    num_samples: int = 1
    """the number of samples to approximate the characteristic with"""

    # ----- Model -----
    model_arch: dict = None
    """the characteristic's neural network architecture"""
    epochs: int = None
    """the number of epochs to train the model"""

    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
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
    patience: int = 10_000
    """the number of epochs to wait before early stopping"""
    verbose: int = 1
    """the verbosity of characteristic training"""
    exact_loss: bool = True
    """whether to record the exact loss"""

    save_val: bool = None
    """whether to save validation results into the `runs/{run_name}` folder"""

@dataclass
class ShapleyArgs:
    model_arch: dict = None
    """the model's neural network architecture"""
    epochs: int = None
    """the number of epochs to train the model"""
    with_exact_char: bool = None
    """whether to use the exact characteristic when computing the exact loss"""

    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
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
    patience: int = 10_000
    """the number of epochs to wait before early stopping"""
    verbose: int = 1
    """the verbosity of characteristic training"""
    exact_loss: bool = True
    """whether to record the exact loss"""

    save_val: bool = None
    """whether to save validation results into the `runs/{run_name}` folder"""

def run(char, seeds=None):

    processes = []

    for seed in tqdm(range(1, exp_args.num_runs + 1) if seeds is None else seeds, desc="Runs"):
        time.sleep(2) # Separate identifiers for each run
        processes.append(subprocess.Popen([
            "python", "run_policy_prediction.py", 
            str(seed),
            json.dumps(exp_args.__dict__),
            json.dumps(char_args.__dict__),
            json.dumps(shapley_args.__dict__),
            "DQN",
            "PolicyShapley",
            char,
        ], preexec_fn=os.setsid))
    
    # Wait for all processes to finish
    for process in processes:
        process.wait()

if __name__ == "__main__":

    exp_args = ExpArgs()
    char_args = CharacteristicArgs()
    shapley_args = ShapleyArgs()

    # 20 runs for first lot of experiments
    exp_args.num_runs = 20

    # Don't save validation for first lot of experiments
    char_args.save_val = False
    shapley_args.save_val = False

    # ------------- Explanation (2 2 2) ------------- #
    exp_args.agent_args_f = None # Fill in wandb run name for agent here.

    # Small characteristic model
    char_args.model_arch = {
        'input': ['Linear', [None, 64]],
        'input_activation': ['ReLU', []],
        'hidden1': ['Linear', [64, 64]],
        'hidden1_activation': ['ReLU', []],
        'output': ['Linear', [64, None]],
        'output_activation': ['Softmax', [-1]],
    }
    char_args.epochs = 50

    # Small Shapley model
    shapley_args.model_arch = {
        'input': ['Linear', [None, 64]],
        'input_activation': ['ReLU', []],
        'hidden1': ['Linear', [64, 64]],
        'hidden1_activation': ['ReLU', []],
        'output': ['Linear', [64, None]],
    }
    shapley_args.epochs = 50

    # ---- With exact char ---- #
    shapley_args.with_exact_char = True

    # Model version
    exp_args.group = f"222_w_exact_model_{ExpArgs.group}"
    run('PolicyCharacteristicModel')

    # Sampling version
    exp_args.group = f"222_w_exact_sampling_{ExpArgs.group}"
    run('PolicyCharacteristicSample')

    # ---- Without exact char (only model version makes sense) ---- #
    shapley_args.with_exact_char = False
    exp_args.group = f"222_wo_exact_model_{ExpArgs.group}"
    run('PolicyCharacteristicModel')

    # ------------- Explanation (3 3 3) ------------- #
    exp_args.agent_args_f = None # Fill in wandb run name for agent here.

    # Big characteristic model
    char_args.model_arch = {
        'input': ['Linear', [None, 128]],
        'input_activation': ['ReLU', []],
        'hidden1': ['Linear', [128, 128]],
        'hidden1_activation': ['ReLU', []],
        'output': ['Linear', [128, None]],
        'output_activation': ['Softmax', [-1]],
    }
    char_args.epochs = 1_000

    # Big Shapley model
    shapley_args.model_arch = {
        'input': ['Linear', [None, 128]],
        'input_activation': ['ReLU', []],
        'hidden1': ['Linear', [128, 128]],
        'hidden1_activation': ['ReLU', []],
        'output': ['Linear', [128, None]],
    }
    shapley_args.epochs = 300

    # ---- With exact char ---- #
    shapley_args.with_exact_char = True

    # Model version
    exp_args.group = f"333_w_exact_model_{ExpArgs.group}"
    run('PolicyCharacteristicModel')

    # Sampling version
    exp_args.group = f"333_w_exact_sampling_{ExpArgs.group}"
    run('PolicyCharacteristicSample')

    # ---- Without exact char (only model version makes sense) ---- #
    shapley_args.with_exact_char = False
    exp_args.group = f"333_wo_exact_model_{ExpArgs.group}"
    run('PolicyCharacteristicModel')