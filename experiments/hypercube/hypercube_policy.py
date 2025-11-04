"""
Trains a DQN agent, behaviour characteristic model, and Shapley model in the Hypercube environment.
Training terminates when given exact loss condition is met or max epochs is reached.
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass, field

from tqdm import tqdm

os.environ["MKL_THREADING_LAYER"] = "GNU"

@dataclass
class ExpArgs:
    group: str = f"{os.path.basename(__file__)[: -len('.py')]}_{int(time.time())}"
    """the group of this experiment"""
    num_runs: int = 20
    """the number of times to run the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "FastSVERL_Hypercube"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    env_id: str = "Hypercube-v0"
    """the id of the environment"""
    num_envs: int = 16
    """the number of parallel game environments"""

@dataclass
class AgentArgs:
    total_timesteps: int = None
    """total timesteps of the experiments"""
    buffer_size: int = None
    """the replay memory buffer size"""

    critic_arch: dict = field(default_factory=lambda: {
        'input': ['Linear', [None, 128]],
        'input_activation': ['ReLU', []],
        'hidden1': ['Linear', [128, 128]],
        'hidden1_activation': ['ReLU', []],
        'output': ['Linear', [128, None]],
    })
    """the critic's neural network architecture"""
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
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: float = 1/50
    """timestep to start learning (here a fraction of `total-timesteps`)"""
    train_frequency: int = 10
    """the frequency of training"""
    eval_episodes: int = 0
    """the number of episodes to evaluate the agent"""
    verbose: int = 0
    """the verbosity of the agent training"""

@dataclass
class BaseSVERLArgs:
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    buffer_size: int = 10_000
    """the replay memory buffer size"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    verbose: int = 0
    """the verbosity of training"""
    exact_loss: bool = True
    """whether to record the exact loss"""
    exact_buffer_size: int = 100_000
    """the size of the buffer to record the exact loss"""
    epochs: int = 10_000
    """the number of epochs to train the model"""
    val_fraction: float = ExpArgs.num_envs * 10 / buffer_size
    """the fraction of data to be used for validation"""
    train_buffer_size: int = int(buffer_size * (1 - val_fraction))
    """the size of the training buffer"""
    val_buffer_size: int = int(buffer_size * val_fraction)
    """the size of the validation buffer"""
    patience: int = epochs
    """the number of epochs to wait before early stopping"""
    policy: str = 'greedy'
    """the policy to explain"""
    track_batch: bool = True
    """if toggled, epochs losses are saved as a function of the batch number"""

@dataclass
class CharacteristicArgs(BaseSVERLArgs):
    model_arch: dict = field(default_factory=lambda: {
        'input': ['Linear', [None, 128]],
        'input_activation': ['ReLU', []],
        'hidden1': ['Linear', [128, 128]],
        'hidden1_activation': ['ReLU', []],
        'output': ['Linear', [128, None]],
        'output_activation': ['Softmax', [-1]],
    })
    """the characteristic's neural network architecture"""
    exact_condition: float = 0.01
    """the exact loss to break the training"""
    char_class: str = 'PolicyCharacteristicModel'
    """the class of the characteristic to use"""

@dataclass
class ShapleyArgs(BaseSVERLArgs):
    with_exact_char: bool = False
    """whether to use the exact characteristic when computing the exact loss"""

    model_arch: dict = field(default_factory=lambda: {
        'input': ['Linear', [None, 128]],
        'input_activation': ['ReLU', []],
        'hidden1': ['Linear', [128, 128]],
        'hidden1_activation': ['ReLU', []],
        'output': ['Linear', [128, None]],
    })
    """the characteristic's neural network architecture"""
    exact_condition: float = 0.001
    """the exact loss to break the training"""
    shapley_class: str = 'PolicyShapley'
    """the class of the Shapley value to use"""

@dataclass
class EnvArgs:
    n: int = None
    """the side length of the hypercube"""
    d: int = None
    """the number of dimensions of the hypercube"""

# [(n, d), total_timesteps] for each environment
timesteps = {           # Number of states = n ^ d - 1
    (2, 2): 100_000,    # 3
    (3, 2): 50_000,     # 8
    (4, 2): 50_000,     # 15
    (5, 2): 50_000,     # 24
    (6, 2): 50_000,     # 35
    (2, 3): 50_000,     # 7
    (3, 3): 50_000,     # 26
    (4, 3): 50_000,     # 63
    (5, 3): 100_000,    # 124
    (6, 3): 100_000,    # 215
    (2, 4): 50_000,     # 15
    (3, 4): 50_000,     # 80
    (4, 4): 100_000,    # 255
    (5, 4): 100_000,    # 624
    (6, 4): 500_000,    # 1295
    (2, 5): 50_000,     # 31
    (3, 5): 100_000,    # 242
    (4, 5): 500_000,    # 1023
    (5, 5): 500_000,    # 3124
    (6, 5): 1_000_000,  # 7775
}

def run():

    processes = []

    for seed in range(1, ExpArgs.num_runs + 1):
        time.sleep(2) # Separate identifiers for each run
        processes.append(subprocess.Popen([
            "python", "run.py",
            str(seed),
            json.dumps(exp_args.__dict__),
            json.dumps(agent_args.__dict__),
            json.dumps(env_args.__dict__),
            json.dumps(CharacteristicArgs().__dict__),
            json.dumps(ShapleyArgs().__dict__),
            ], preexec_fn=os.setsid))
    
    # Wait for all processes to finish
    for process in processes:
        process.wait()

if __name__ == "__main__":

    exp_args = ExpArgs()
    agent_args = AgentArgs()
    env_args = EnvArgs()

    # Varying cube size
    for (n, d), total_timesteps in tqdm(timesteps.items(), desc="Cubes"):

        exp_args.group = f"{n}_{d}_{ExpArgs.group}"

        # Environment
        env_args.n = n
        env_args.d = d

        # Agent
        agent_args.total_timesteps = total_timesteps
        agent_args.buffer_size = agent_args.total_timesteps
        agent_args.learning_starts = int(AgentArgs.learning_starts * agent_args.total_timesteps)
        
        # Run the experiment
        run()