"""
Trains outcome characteristic models for a given agent in a specified environment.
"""

import json
import sys
import time

from fastsverl.utils import Buffer, ExpArgs, EnvArgs, AgentArgs, PerformanceCharacteristicArgs, PolicyCharacteristicArgs
from fastsverl.training import fill_buffer, setup_envs, train_models, train_V_on_policy, setup
from fastsverl.characteristics import PolicyCharacteristicModel, PolicyCharacteristicSample, PerformanceCharacteristicOnPolicy
from fastsverl.dqn import DQN
from fastsverl.ppo import PPO

if __name__ == "__main__":

    # Experiment setup
    seed = int(sys.argv[1])
    run_name = f"{seed}_{int(time.time_ns())}"

    # Arguments
    exp_args = ExpArgs(**json.loads(sys.argv[2]))
    policy_char_args = PolicyCharacteristicArgs(**json.loads(sys.argv[3]))
    char_args = PerformanceCharacteristicArgs(**json.loads(sys.argv[5]))
    with open(f"../agents/runs/{exp_args.agent_args_f}/EnvArgs.json", "r") as f:
        env_args = EnvArgs(**json.load(f))
    with open(f"../agents/runs/{exp_args.agent_args_f}/AgentArgs.json", "r") as f:
        agent_args = AgentArgs(**json.load(f))

    # Set up writer for logging
    writer = setup(run_name, exp_args, [agent_args, char_args, policy_char_args], seed)

    # env setup
    envs = setup_envs(env_args, exp_args, seed, run_name, **vars(env_args))

    # agent setup
    agent = globals()[sys.argv[4]](envs, agent_args)
    agent.load_models(f"../agents/runs/{exp_args.agent_args_f}", eval=True, buffer=True, epsilon=True)
    agent.buffer.shuffle()

    # Set up policy characteristic
    train_buffer = Buffer(policy_char_args.train_buffer_size, envs, 'obs')
    train_buffer, _ = fill_buffer(agent, envs, train_buffer, policy_char_args.train_buffer_size, policy_char_args.policy, 'obs')

    val_buffer = Buffer(policy_char_args.val_buffer_size, envs, 'obs')
    val_buffer, _ = fill_buffer(agent, envs, val_buffer, policy_char_args.val_buffer_size, policy_char_args.policy, 'obs')

    policy_char = globals()[sys.argv[6]](agent, envs, policy_char_args, train_buffer, val_buffer=val_buffer, exact_buffer=train_buffer)

    if sys.argv[6] == 'PolicyCharacteristicModel':
        train_models([policy_char], policy_char_args, writer, run_name)

    # Train performance charateristic.
    char = PerformanceCharacteristicOnPolicy(agent, envs, char_args, policy_char)
    train_V_on_policy(char, envs, char_args, writer, seed, run_name)

    envs.close()
    writer.close()