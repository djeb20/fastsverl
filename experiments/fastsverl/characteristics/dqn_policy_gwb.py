"""
Trains multiple behaviour characteristic models for a given agent in the GWB environment.
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
    wandb_project_name: str = "FastSVERL_GWB"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    env_id: str = "GWB-v0"
    """the id of the environment"""
    num_envs: int = 16
    """the number of parallel game environments"""
    agent_args_f: str = None # Fill in wandb run name for agent here.
    """where to load the agent's and the environment's hyperparameters from"""

@dataclass
class CharacteristicArgs:
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    save_train_buffer: bool = True
    """whether to save the train buffer"""
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
    verbose: int = 0
    """the verbosity of characteristic training"""
    exact_loss: bool = True
    """whether to record the exact loss"""
    save_exact_buffer: bool = True
    """whether to save the exact buffer"""
    policy: str = 'greedy'
    """the policy to explain"""

if __name__ == "__main__":

    for seed in tqdm(range(1, ExpArgs.num_runs + 1), desc="Runs"):
        subprocess.run([
            "python", "run_policy_prediction.py", 
            str(seed),
            json.dumps(ExpArgs().__dict__),
            json.dumps(CharacteristicArgs().__dict__),
            "DQN",
            "PolicyCharacteristicModel",
            ], preexec_fn=os.setsid)