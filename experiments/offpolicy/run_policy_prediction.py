"""
Trains off-policy behaviour and prediction characteristic models in given environment.
"""

import json
import sys
import time
import torch

from fastsverl.utils import Buffer, ExpArgs, EnvArgs, AgentArgs, CharacteristicArgs
from fastsverl.training import setup_envs, train_models, fill_buffer, setup, train_agent
from fastsverl.characteristics import PolicyCharacteristicModel, ValueCharacteristicModel
from fastsverl.dqn import DQN
from fastsverl.ppo import PPO

def look_at_dists(agent, exact_buffer):

    e_obs, p_obs = torch.unique(agent.buffer.sample(agent.buffer.size, 'obs', start=0)[0], dim=0, return_counts=True)
    print(e_obs, p_obs / p_obs.sum())

    e_obs, p_obs = torch.unique(exact_buffer.sample(exact_buffer.size, 'obs', start=0)[0], dim=0, return_counts=True)
    print(e_obs, p_obs / p_obs.sum())

if __name__ == "__main__":

    # Experiment setup
    seed = int(sys.argv[1])
    run_name = f"{seed}_{int(time.time_ns())}"

    # Arguments
    exp_args = ExpArgs(**json.loads(sys.argv[2]))
    agent_args = AgentArgs(**json.loads(sys.argv[3]))
    env_args = EnvArgs(**json.loads(sys.argv[4]))
    char_args = CharacteristicArgs(**json.loads(sys.argv[5]))
    
    # Set up writer for logging
    writer = setup(run_name, exp_args, [agent_args, char_args], seed)

    # env setup
    envs = setup_envs(env_args, exp_args, seed, run_name, **vars(env_args))

    # Train agent.
    agent = globals()[sys.argv[6]](envs, agent_args)
    train_agent(agent, [], envs, agent_args, writer, seed, run_name)
    agent.buffer.shuffle()

    # Approximate steady-state
    train_buffer = Buffer(char_args.train_buffer_size, envs, 'obs', 'action', 'logprob')
    val_buffer = Buffer(char_args.val_buffer_size, envs, 'obs', 'action', 'logprob')

    # Accurate
    if exp_args.group[:5] == 'exact':
        train_buffer, _ = fill_buffer(agent, envs, train_buffer, char_args.train_buffer_size, char_args.policy, 'obs')
        val_buffer, _ = fill_buffer(agent, envs, val_buffer, char_args.val_buffer_size, char_args.policy, 'obs')
        
    # Buffered
    else:
        train_buffer.add(**{key: agent.buffer.buffer[key][:char_args.train_buffer_size] for key in train_buffer.buffer})
        val_buffer.add(**{key: agent.buffer.buffer[key][char_args.train_buffer_size:char_args.train_buffer_size + char_args.val_buffer_size] for key in val_buffer.buffer})

    exact_buffer = Buffer(char_args.exact_buffer_size, envs, 'obs')
    exact_buffer, _ = fill_buffer(agent, envs, exact_buffer, char_args.exact_buffer_size, char_args.policy, 'obs')

    # Train characteristic
    char = globals()[sys.argv[7]](agent, envs, char_args, train_buffer, val_buffer, exact_buffer)
    train_models([char], char_args, writer, run_name)

    look_at_dists(agent, exact_buffer)

    envs.close()
    writer.close()