"""
Trains combined DQN, behaviour and prediction characteristic and Shapley models.
"""

import json
import sys
import time

from fastsverl.utils import ExpArgs, ShapleyArgs, EnvArgs, AgentArgs, CharacteristicArgs
from fastsverl.training import setup_envs, setup, train_agent
from fastsverl.characteristics import PolicyCharacteristicModel, ValueCharacteristicModel, XDQN_C
from fastsverl.shapley import PolicyShapley, ValueShapley, XDQN_S
from fastsverl.dqn import DQN

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

    # DQN agent
    if sys.argv[9] == 'DQN':

        # Set up agent.
        agent_args.critic_arch = agent_args.shared_arch | agent_args.critic_arch
        agent = DQN(envs, agent_args)

        # Set up characteristic
        char_args.model_arch = agent_args.shared_arch | char_args.model_arch
        char = globals()[sys.argv[7]](agent, envs, char_args)

        # Set up Shapley
        sv_args.model_arch = agent_args.shared_arch | sv_args.model_arch
        shapley = globals()[sys.argv[8]](envs, sv_args, char)

        # Train all models - in this order to ensure we have exact value for characteristics used by other models.
        train_agent(agent, [char, shapley], envs, agent_args, writer, seed, run_name)

    # DQN combined with characteristic
    elif sys.argv[9] == 'XDQN_C':

        # Set up agent.
        agent_args.exact_loss = getattr(char_args, 'exact_loss', False)
        agent = XDQN_C(envs, agent_args, **{globals()[sys.argv[7]].__name__: char_args})

        # Set up Shapley
        sv_args.model_arch = agent_args.shared_arch | sv_args.model_arch
        shapley = globals()[sys.argv[8]](envs, sv_args, agent.chars[globals()[sys.argv[7]].__name__])

        # Train all models
        train_agent(agent, [shapley], envs, agent_args, writer, seed, run_name)

    # DQN combined with Shapley (sys.argv[9] == 'XDQN_S')
    else:

        # Set up agent.
        agent_args.exact_loss = getattr(sv_args, 'exact_loss', False)
        agent = XDQN_S(envs, agent_args, **{globals()[sys.argv[8]].__name__: sv_args})

        # Set up characteristic
        char_args.model_arch = agent_args.shared_arch | char_args.model_arch
        char = globals()[sys.argv[7]](agent, envs, char_args)
        agent.add_chars({globals()[sys.argv[8]].__name__: char})

        # Train all models
        train_agent(agent, [char], envs, agent_args, writer, seed, run_name)

    envs.close()
    writer.close()