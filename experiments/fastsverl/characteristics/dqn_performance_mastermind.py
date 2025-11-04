"""
Trains multiple outcome characteristic models for a given agent in a particular Mastermind configuration.
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
class CharacteristicArgs:
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
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    gamma: float = 1
    """the discount factor gamma"""
    tau: float = 0.01
    """the target network update rate"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    start_e: float = None
    """the starting epsilon for exploration"""
    verbose: int = 1
    """the verbosity of characteristic training"""
    exact_loss: bool = True
    """whether to record the exact loss"""
    policy_char_args_f: str = None # Fill in wandb run name for Mastermind-222 behaviour characteristic here.
    """where to load the policy characteristic's hyperparameters from"""

    # For on-policy version
    total_timesteps: int = 200_000
    """total timesteps of the experiments"""
    buffer_size: int = total_timesteps
    """the replay memory buffer size"""
    target_network_frequency: int = 1
    """the timesteps it takes to update the target network"""
    train_frequency: int = 1
    """the frequency of training"""
    eval_episodes: int = 0
    """the number of episodes to evaluate the agent"""
    sampling_frequency: str = "decision"
    """the frequency of sampling a new s_e and C"""

    # For off-policy version
    train_buffer_size: int = 100_000 # Size of agent's buffer.
    """the size of the training buffer"""
    val_buffer_size: int = ExpArgs.num_envs * 10
    """the size of the validation buffer""" 
    epochs: int = total_timesteps * batch_size // train_buffer_size
    """the number of epochs to train the model""" # 200_000 * 64 // 100_000 = 128
    patience: int = epochs
    """the number of epochs to wait before early stopping"""
    track_batch: bool = True
    """whether to save epoch losses at batch granularity"""

def run(perf_char, seeds=None):

    processes = []

    for seed in tqdm(range(1, exp_args.num_runs + 1) if seeds is None else seeds, desc="Runs"):
        time.sleep(2) # Separate identifiers for each run
        processes.append(subprocess.Popen([
            "python", "run_performance.py",
            str(seed),
            json.dumps(exp_args.__dict__),
            json.dumps(char_args.__dict__),
            "DQN",
            perf_char,
            ], preexec_fn=os.setsid))
        
    # Wait for all processes to finish
    for process in processes:
        process.wait()

if __name__ == "__main__":

    exp_args = ExpArgs()
    char_args = CharacteristicArgs()

    # 20 runs for first lot of experiments
    exp_args.num_runs = 20

    # ---- Using exact policy char when computing exact loss ----
    char_args.with_exact_char = True

    # On-policy
    exp_args.group = f"222_w_exact_onpolicy_{ExpArgs.group}"
    run('PerformanceCharacteristicOnPolicy')

    # Off-policy
    exp_args.group = f"222_w_exact_offpolicy_{ExpArgs.group}"
    run('PerformanceCharacteristicOffPolicy')

    # ---- Without exact policy char when computing exact loss ----
    char_args.with_exact_char = False

    # On-policy
    exp_args.group = f"222_wo_exact_onpolicy_{ExpArgs.group}"
    run('PerformanceCharacteristicOnPolicy')

    # Off-policy
    exp_args.group = f"222_wo_exact_offpolicy_{ExpArgs.group}"
    run('PerformanceCharacteristicOffPolicy')

    # ----------- Non exact experiments ---------------

    # 1 run for second lot of experiments
    exp_args.num_runs = 1

    # No exact loss
    char_args.exact_loss = False

    # New model architecture
    char_args.model_arch = {
        'input': ['Linear', [None, 128]],
        'input_activation': ['ReLU', []],
        'hidden1': ['Linear', [128, 128]],
        'hidden1_activation': ['ReLU', []],
        'output': ['Linear', [128, None]],
    }

    # New on-policy args
    char_args.total_timesteps = 10_000_000
    char_args.buffer_size = char_args.total_timesteps

    # New off-policy args
    char_args.train_buffer_size = 10_000_000 # Size of agent's buffer.
    char_args.epochs = char_args.total_timesteps * char_args.batch_size // char_args.train_buffer_size
    char_args.patience = char_args.epochs

    # ---- Explanation (4 4 3) ----
    exp_args.agent_args_f = None # Fill in wandb run name for Mastermind-443 agent here.
    char_args.policy_char_args_f = None # Fill in wandb run name for Mastermind-443 behaviour characteristic here.
    
    exp_args.group = f"443_onpolicy_{ExpArgs.group}"
    run('PerformanceCharacteristicOnPolicy')

    exp_args.group = f"443_offpolicy_{ExpArgs.group}"
    run('PerformanceCharacteristicOffPolicy')

    # ---- Explanation (4 5 3) ----
    exp_args.agent_args_f = None # Fill in wandb run name for Mastermind-453 agent here.
    char_args.policy_char_args_f = None # Fill in wandb run name for Mastermind-453 behaviour characteristic here.

    exp_args.group = f"453_onpolicy_{ExpArgs.group}"
    run('PerformanceCharacteristicOnPolicy')

    exp_args.group = f"453_offpolicy_{ExpArgs.group}"
    run('PerformanceCharacteristicOffPolicy')

    # ---- Explanation (4 6 3) ----
    exp_args.agent_args_f = None # Fill in wandb run name for Mastermind-463 agent here.
    char_args.policy_char_args_f = None # Fill in wandb run name for Mastermind-463 behaviour characteristic here.

    exp_args.group = f"463_onpolicy_{ExpArgs.group}"
    run('PerformanceCharacteristicOnPolicy')

    exp_args.group = f"463_offpolicy_{ExpArgs.group}"
    run('PerformanceCharacteristicOffPolicy')
