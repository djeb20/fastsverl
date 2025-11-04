"""
Trains multiple outcome Shapley value models for DQN agents in the Mastermind environment.
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
    char_args_f: str = None
    """where to load the characteristic from"""

    group: str = f"{os.path.basename(os.path.dirname(__file__))}_{os.path.basename(__file__)[: -len('.py')]}_{int(time.time())}"
    """the group of this experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
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
    agent_args_f: str = None # Fill in wandb run name for Mastermind-222 agent here.
    """where to load the agent's and the environment's hyperparameters from"""

@dataclass
class ShapleyArgs:
    with_exact_char: bool = None
    """whether to use the exact characteristic model when computing the exact loss"""

    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    model_arch: dict = field(default_factory=lambda: {
        'input': ['Linear', [None, 64]],
        'input_activation': ['ReLU', []],
        'hidden1': ['Linear', [64, 64]],
        'hidden1_activation': ['ReLU', []],
        'output': ['Linear', [64, None]],
    })
    """the model's neural network architecture"""
    epochs: int = 100
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
    off_policy_char: bool = True
    """whether to use the off-policy characteristic model"""

    save_val: bool = None
    """whether to save validation results into the `runs/{run_name}` folder"""
    save_explanations: bool = None
    """whether to save explanations into the `runs/{run_name}` folder"""

def run(seeds=None):

    processes = []

    for seed in tqdm(range(1, exp_args.num_runs + 1) if seeds is None else seeds, desc="Runs"):

        # Only save explanations for first run
        shapley_args.save_explanations = (seed == 1)

        time.sleep(2) # Separate identifiers for each run
        processes.append(subprocess.Popen([
            "python", "run_performance.py", 
            str(seed),
            json.dumps(exp_args.__dict__),
            json.dumps(shapley_args.__dict__),
            "DQN",
        ], preexec_fn=os.setsid))

    # Wait for all processes to finish
    for process in processes:
        process.wait()

if __name__ == "__main__":

    exp_args = ExpArgs()
    shapley_args = ShapleyArgs()

    # 20 runs for first lot of experiments
    exp_args.num_runs = 20

    # Don't save validation for first lot of experiments
    shapley_args.save_val = False

    # ------ With exact performance char ------
    shapley_args.with_exact_char = True

    # With exact policy char
    exp_args.char_args_f = None # Fill in wandb run name for characteristic here.
    exp_args.group = f"222_w_policy_{ExpArgs.group}"
    run()

    # Without exact policy char
    exp_args.char_args_f = None # Fill in wandb run name for characteristic here.
    exp_args.group = f"222_wo_policy_{ExpArgs.group}"
    run()

    # ------ Without exact performance char ------
    shapley_args.with_exact_char = False
    exp_args.group = f"222_wo_perf_{ExpArgs.group}"
    run()

    # ----------- Non exact experiments ---------------

    # 10 runs for second lot of experiments
    exp_args.num_runs = 10

    # No exact loss
    shapley_args.exact_loss = False

    # New epochs
    shapley_args.epochs = 50_000

    # New buffer size
    shapley_args.buffer_size = 10_000
    shapley_args.train_buffer_size = int(shapley_args.buffer_size * (1 - shapley_args.val_fraction))
    shapley_args.val_buffer_size = int(shapley_args.buffer_size * shapley_args.val_fraction)
    shapley_args.patience = 5_000
    shapley_args.save_val = True

    # New model architecture
    shapley_args.model_arch = {
        'input': ['Linear', [None, 128]],
        'input_activation': ['ReLU', []],
        'hidden1': ['Linear', [128, 128]],
        'hidden1_activation': ['ReLU', []],
        'output': ['Linear', [128, None]],
    }

    # ---- Explanation (4 4 3) ----
    exp_args.agent_args_f = None # Fill in wandb run name for Mastermind-443 agent here.

    exp_args.group = f"443_onpolicy_{ExpArgs.group}"
    exp_args.char_args_f = None # Fill in wandb run name for characteristic here.
    shapley_args.off_policy_char = False
    run([1, 2, 3, 4, 5])  # Separate runs for memory issues
    run([6, 7, 8, 9, 10])

    exp_args.group = f"443_offpolicy_{ExpArgs.group}"
    exp_args.char_args_f = None # Fill in wandb run name for characteristic here.
    shapley_args.off_policy_char = True
    run([1, 2, 3, 4, 5]) 
    run([6, 7, 8, 9, 10])

    # ---- Explanation (4 5 3) ----
    exp_args.agent_args_f = None # Fill in wandb run name for Mastermind-453 agent here.

    exp_args.group = f"453_onpolicy_{ExpArgs.group}"
    exp_args.char_args_f = None # Fill in wandb run name for characteristic here.
    shapley_args.off_policy_char = False
    run([1, 2, 3, 4, 5])
    run([6, 7, 8, 9, 10])

    exp_args.group = f"453_offpolicy_{ExpArgs.group}"
    exp_args.char_args_f = None # Fill in wandb run name for characteristic here.
    shapley_args.off_policy_char = True
    run([1, 2, 3, 4, 5])
    run([6, 7, 8, 9, 10])

    # ---- Explanation (4 6 3) ----
    exp_args.agent_args_f = None # Fill in wandb run name for Mastermind-463 agent here.

    exp_args.group = f"463_onpolicy_{ExpArgs.group}"
    exp_args.char_args_f = None # Fill in wandb run name for characteristic here.
    shapley_args.off_policy_char = False
    run([1, 2, 3, 4, 5])
    run([6, 7, 8, 9, 10])

    exp_args.group = f"463_offpolicy_{ExpArgs.group}"
    exp_args.char_args_f = None # Fill in wandb run name for characteristic here.
    shapley_args.off_policy_char = True
    run([1, 2, 3, 4, 5])
    run([6, 7, 8, 9, 10])
