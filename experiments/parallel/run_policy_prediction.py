"""
Trains agent, behaviour and prediction characteristic, and Shapley models in parallel on a given environment.
"""

import json
import sys
import time

from fastsverl.utils import ExpArgs, ShapleyArgs, EnvArgs, AgentArgs, CharacteristicArgs
from fastsverl.training import setup_envs, setup, train_agent
from fastsverl.characteristics import PolicyCharacteristicModel, ValueCharacteristicModel
from fastsverl.shapley import PolicyShapley, ValueShapley
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
    char_args = CharacteristicArgs(**json.loads(sys.argv[5]))
    sv_args = ShapleyArgs(**json.loads(sys.argv[6]))
    
    # Set up writer for logging
    writer = setup(run_name, exp_args, [agent_args, char_args, sv_args], seed)

    # env setup
    envs = setup_envs(env_args, exp_args, seed, run_name, **vars(env_args))

    # Set up agent.
    agent = globals()[sys.argv[7]](envs, agent_args)

    # Set up characteristic
    char = globals()[sys.argv[8]](agent, envs, char_args)

    # Set up Shapley values
    shapley = globals()[sys.argv[9]](envs, sv_args, char)

    # Train all models - in this order to ensure we have exact value for characteristics used by other models.
    train_agent(agent, [char, shapley], envs, agent_args, writer, seed, run_name)

    envs.close()
    writer.close()