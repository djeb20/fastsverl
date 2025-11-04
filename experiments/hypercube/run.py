"""
Trains a single DQN agent, characteristic, and Shapley model in Hypercube environment.
"""

import json
import sys
import time

from fastsverl.utils import Buffer, ExpArgs, EnvArgs, AgentArgs, ShapleyArgs, CharacteristicArgs
from fastsverl.training import fill_buffer, train_agent, setup, setup_envs, train_models
from fastsverl.dqn import DQN
from fastsverl.envs.hypercube import Hypercube
from fastsverl.characteristics import PolicyCharacteristicModel, ValueCharacteristicModel
from fastsverl.shapley import PolicyShapley, ValueShapley

if __name__ == "__main__":

    # Experiment setup
    seed = int(sys.argv[1])
    run_name = f"{seed}_{int(time.time_ns())}"

    # Arguments
    exp_args = ExpArgs(**json.loads(sys.argv[2]))
    agent_args = AgentArgs(**json.loads(sys.argv[3]))
    env_args = EnvArgs(**json.loads(sys.argv[4]))
    char_args = CharacteristicArgs(**json.loads(sys.argv[5]))
    shapley_args = ShapleyArgs(**json.loads(sys.argv[6]))

    # Set up writer for logging
    writer = setup(run_name, exp_args, [agent_args, char_args, shapley_args], seed)

    # env setup
    envs = setup_envs(env_args, exp_args, seed, run_name, **vars(env_args))

    # Train agent.
    agent = DQN(envs, agent_args)
    train_agent(agent, [], envs, agent_args, writer, seed, run_name)

    # Approximate steady state distribution
    train_buffer = Buffer(char_args.train_buffer_size, envs, 'obs')
    train_buffer, _ = fill_buffer(agent, envs, train_buffer, char_args.train_buffer_size, char_args.policy, 'obs')
    
    val_buffer = Buffer(char_args.val_buffer_size, envs, 'obs')
    val_buffer, _ = fill_buffer(agent, envs, val_buffer, char_args.val_buffer_size, char_args.policy, 'obs')

    exact_buffer = Buffer(char_args.exact_buffer_size, envs, 'obs')
    exact_buffer, _ = fill_buffer(agent, envs, exact_buffer, char_args.exact_buffer_size, char_args.policy, 'obs')

    # Train characteristic
    char = globals()[char_args.char_class](agent, envs, char_args, train_buffer, val_buffer, exact_buffer)
    train_models([char], char_args, writer, run_name)

    # Train Shapley
    shapley = globals()[shapley_args.shapley_class](envs, shapley_args, char, train_buffer, val_buffer)
    train_models([shapley], shapley_args, writer, run_name)

    envs.close()
    writer.close()