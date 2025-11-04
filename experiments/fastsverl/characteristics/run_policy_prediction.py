"""
Trains a single behaviour or prediction characteristic model for a given agent in the specified environment.
"""

import json
import sys
import time

from fastsverl.utils import Buffer, ExpArgs, EnvArgs, AgentArgs, CharacteristicArgs
from fastsverl.training import setup_envs, train_models, fill_buffer, setup
from fastsverl.characteristics import PolicyCharacteristicModel, ValueCharacteristicModel
from fastsverl.dqn import DQN
from fastsverl.ppo import PPO

if __name__ == "__main__":

    # Experiment setup
    seed = int(sys.argv[1])
    run_name = f"{seed}_{int(time.time_ns())}"

    # Arguments
    exp_args = ExpArgs(**json.loads(sys.argv[2]))    
    char_args = CharacteristicArgs(**json.loads(sys.argv[3])) 
    with open(f"../agents/runs/{exp_args.agent_args_f}/EnvArgs.json", "r") as f:
        env_args = EnvArgs(**json.load(f))
    with open(f"../agents/runs/{exp_args.agent_args_f}/AgentArgs.json", "r") as f:
        agent_args = AgentArgs(**json.load(f))

    # Set up writer for logging
    writer = setup(run_name, exp_args, [char_args], seed)

    # env setup
    envs = setup_envs(env_args, exp_args, seed, run_name, **vars(env_args))

    # agent setup
    agent = globals()[sys.argv[4]](envs, agent_args)
    agent.load_models(f"../agents/runs/{exp_args.agent_args_f}", eval=True, buffer=False, epsilon=True)

    # Train characteristic.
    train_buffer = Buffer(char_args.train_buffer_size, envs, 'obs')
    train_buffer, _ = fill_buffer(agent, envs, train_buffer, char_args.train_buffer_size, char_args.policy, 'obs')
    
    val_buffer = Buffer(char_args.val_buffer_size, envs, 'obs')
    val_buffer, _ = fill_buffer(agent, envs, val_buffer, char_args.val_buffer_size, char_args.policy, 'obs')

    char = globals()[sys.argv[5]](agent, envs, char_args, train_buffer, val_buffer, exact_buffer=train_buffer)
    train_models([char], char_args, writer, run_name)

    envs.close()
    writer.close()