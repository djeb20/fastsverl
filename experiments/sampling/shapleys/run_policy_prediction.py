"""
Trains behaviour and prediction Shapley models for given agents on a given environment,
comparing model-based and sampling-based approaches.
"""

import json
import sys
import time

from fastsverl.utils import Buffer, ExpArgs, ShapleyArgs, EnvArgs, AgentArgs, CharacteristicArgs
from fastsverl.training import setup_envs, train_models, fill_buffer, setup
from fastsverl.characteristics import PolicyCharacteristicModel, ValueCharacteristicModel, PolicyCharacteristicSample, ValueCharacteristicSample
from fastsverl.shapley import PolicyShapley, ValueShapley
from fastsverl.dqn import DQN
from fastsverl.ppo import PPO

if __name__ == "__main__":

    # Experiment setup
    seed = int(sys.argv[1])
    run_name = f"{seed}_{int(time.time_ns())}"

    # Arguments
    exp_args = ExpArgs(**json.loads(sys.argv[2]))
    char_args = CharacteristicArgs(**json.loads(sys.argv[3]))
    sv_args = ShapleyArgs(**json.loads(sys.argv[4]))
    with open(f"../agents/runs/{exp_args.agent_args_f}/EnvArgs.json", "r") as f:
        env_args = EnvArgs(**json.load(f))
    with open(f"../agents/runs/{exp_args.agent_args_f}/AgentArgs.json", "r") as f:
        agent_args = AgentArgs(**json.load(f))

    # Set up writer for logging
    writer = setup(run_name, exp_args, [sv_args, char_args], seed)

    # env setup
    envs = setup_envs(env_args, exp_args, seed, run_name, **vars(env_args))

    # agent setup
    agent = globals()[sys.argv[5]](envs, agent_args)
    agent.load_models(f"../agents/runs/{exp_args.agent_args_f}", eval=True, epsilon=True)

    # Characteristic is a model
    if sys.argv[7] in ['PolicyCharacteristicModel', 'ValueCharacteristicModel']:
        
        # Train characteristic.
        train_buffer = Buffer(char_args.train_buffer_size, envs, 'obs')
        train_buffer, _ = fill_buffer(agent, envs, train_buffer, char_args.train_buffer_size, char_args.policy, 'obs')
        
        val_buffer = Buffer(char_args.val_buffer_size, envs, 'obs')
        val_buffer, _ = fill_buffer(agent, envs, val_buffer, char_args.val_buffer_size, char_args.policy, 'obs')

        char = globals()[sys.argv[7]](agent, envs, char_args, train_buffer, val_buffer, exact_buffer=train_buffer)
        train_models([char], char_args, writer, run_name)

        # Train Shalpey values
        train_buffer = Buffer(sv_args.train_buffer_size, envs, 'obs')
        train_buffer, _ = fill_buffer(agent, envs, train_buffer, sv_args.train_buffer_size, char_args.policy, 'obs')

        val_buffer = Buffer(sv_args.val_buffer_size, envs, 'obs')
        val_buffer, _ = fill_buffer(agent, envs, val_buffer, sv_args.val_buffer_size, char_args.policy, 'obs')    

    # Characteristic is sampled (e.g. PolicyCharacteristicSample or ValueCharacteristicSample)
    else:

        # Steady-state approximation
        train_buffer = Buffer(sv_args.train_buffer_size, envs, 'obs')
        train_buffer, _ = fill_buffer(agent, envs, train_buffer, sv_args.train_buffer_size, char_args.policy, 'obs')

        val_buffer = Buffer(sv_args.val_buffer_size, envs, 'obs')
        val_buffer, _ = fill_buffer(agent, envs, val_buffer, sv_args.val_buffer_size, char_args.policy, 'obs')

        # char setup
        char = globals()[sys.argv[7]](agent, envs, char_args, train_buffer, exact_buffer=train_buffer)

    # Train Shalpey values
    shapley = globals()[sys.argv[6]](envs, sv_args, char, train_buffer, val_buffer)
    train_models([shapley], sv_args, writer, run_name)

    envs.close()
    writer.close()