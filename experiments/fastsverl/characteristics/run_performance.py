"""
Trains a single outcome characteristic model for a given agent in the specified environment.
"""

import json
import sys
import time

from fastsverl.utils import Buffer, ExpArgs, EnvArgs, AgentArgs, PolicyCharacteristicArgs, CharacteristicArgs
from fastsverl.training import setup_envs, train_V_on_policy, setup, train_models
from fastsverl.characteristics import PolicyCharacteristicModel, PerformanceCharacteristicOnPolicy, PerformanceCharacteristicOffPolicy
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
    with open(f"runs/{char_args.policy_char_args_f}/CharacteristicArgs.json", "r") as f:
        policy_char_args = PolicyCharacteristicArgs(**json.load(f))

    # Set up writer for logging
    writer = setup(run_name, exp_args, [char_args], seed)

    # env setup
    envs = setup_envs(env_args, exp_args, seed, run_name, **vars(env_args))

    # agent setup
    agent = globals()[sys.argv[4]](envs, agent_args)
    agent.load_models(f"../agents/runs/{exp_args.agent_args_f}", eval=True, buffer=True, epsilon=True)

    # policy char setup
    policy_char = PolicyCharacteristicModel(agent, envs, policy_char_args, train_buffer=None, val_buffer=None)
    policy_char.load_model(f"runs/{char_args.policy_char_args_f}", eval=True, train_buffer=policy_char_args.save_train_buffer, exact_buffer=policy_char_args.save_exact_buffer)

    # Train performance charateristic.
    if sys.argv[5] == 'PerformanceCharacteristicOnPolicy':
        char = PerformanceCharacteristicOnPolicy(agent, envs, char_args, policy_char)
        train_V_on_policy(char, envs, char_args, writer, seed, run_name)

    elif sys.argv[5] == 'PerformanceCharacteristicOffPolicy':
        train_buffer = Buffer(char_args.train_buffer_size, envs, 'obs', 'action', 'reward', 'n_obs', 'terminated')
        train_buffer.add(**{key: agent.buffer.buffer[key][:char_args.train_buffer_size] for key in train_buffer.buffer})

        val_buffer = Buffer(char_args.val_buffer_size, envs, 'obs', 'action', 'reward', 'n_obs', 'terminated')
        val_buffer.add(**{key: agent.buffer.buffer[key][:char_args.val_buffer_size] for key in val_buffer.buffer})

        char = PerformanceCharacteristicOffPolicy(agent, envs, char_args, policy_char, train_buffer, val_buffer)
        train_models([char], char_args, writer, run_name)

    envs.close()
    writer.close()