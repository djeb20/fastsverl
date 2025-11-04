"""
Trains agent, outcome characteristic, and Shapley models in parallel on a given environment.
"""

import json
import sys
import time

from fastsverl.utils import ExpArgs, ShapleyArgs, EnvArgs, AgentArgs, PerformanceCharacteristicArgs, PolicyCharacteristicArgs
from fastsverl.training import setup_envs, train_agent, setup
from fastsverl.characteristics import PolicyCharacteristicModel, PerformanceCharacteristicOffPolicy
from fastsverl.shapley import PerformanceShapley
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
    char_args = PerformanceCharacteristicArgs(**json.loads(sys.argv[5]))
    policy_char_args = PolicyCharacteristicArgs(**json.loads(sys.argv[6]))
    sv_args = ShapleyArgs(**json.loads(sys.argv[7]))

    # Set up writer for logging
    writer = setup(run_name, exp_args, [agent_args, char_args, policy_char_args, sv_args], seed)

    # env setup
    envs = setup_envs(env_args, exp_args, seed, run_name, **vars(env_args))

    # Set up agent
    agent = globals()[sys.argv[8]](envs, agent_args)

    # Set up policy characteristic
    policy_char = PolicyCharacteristicModel(agent, envs, policy_char_args)

    # Set up performance charateristic.
    char = PerformanceCharacteristicOffPolicy(agent, envs, char_args, policy_char)

    # Set up Shapley values
    shapley = PerformanceShapley(envs, sv_args, char)

    # Train all models - in this order to ensure we have exact value for characteristics used by other models.
    train_agent(agent, [policy_char, char, shapley], envs, agent_args, writer, seed, run_name)

    envs.close()
    writer.close()