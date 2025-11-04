"""
Trains a single agent in a specified environment.
"""

import json
import sys
import time

from fastsverl.utils import ExpArgs, AgentArgs, EnvArgs
from fastsverl.training import train_agent, setup, setup_envs
from fastsverl.dqn import DQN
from fastsverl.ppo import PPO

if __name__ == "__main__":

    # Experiment setup
    seed = int(sys.argv[1])
    run_name = f"{seed}_{int(time.time_ns())}"

    # Arguments
    exp_args = ExpArgs(**json.loads(sys.argv[2]))
    agent_args = AgentArgs(**json.loads(sys.argv[3]))
    env_args = EnvArgs(**json.loads(sys.argv[4]))

    # Set up writer for logging
    writer = setup(run_name, exp_args, [agent_args], seed)

    # env setup
    envs = setup_envs(env_args, exp_args, seed, run_name, **vars(env_args))

    # Train agent.
    agent = globals()[sys.argv[5]](envs, agent_args)
    train_agent(agent, [], envs, agent_args, writer, seed, run_name)

    envs.close()
    writer.close()