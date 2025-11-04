"""
Trains combined DQN, performance characteristic and Shapley models.
"""

import json
import sys
import time

from fastsverl.utils import ExpArgs, ShapleyArgs, EnvArgs, AgentArgs, PolicyCharacteristicArgs, PerformanceCharacteristicArgs
from fastsverl.training import setup_envs, setup, train_agent
from fastsverl.characteristics import PolicyCharacteristicModel, PerformanceCharacteristicOffPolicy
from fastsverl.shapley import PerformanceShapley, XDQN_S
from fastsverl.dqn import DQN

if __name__ == "__main__":

    # Experiment setup
    seed = int(sys.argv[1])
    run_name = f"{seed}_{int(time.time_ns())}"

    # Arguments
    exp_args = ExpArgs(**json.loads(sys.argv[2]))
    agent_args = AgentArgs(**json.loads(sys.argv[3]))
    env_args = EnvArgs(**json.loads(sys.argv[4]))
    policy_char_args = PolicyCharacteristicArgs(**json.loads(sys.argv[5]))
    char_args = PerformanceCharacteristicArgs(**json.loads(sys.argv[6]))
    sv_args = ShapleyArgs(**json.loads(sys.argv[7]))
    
    # Set up writer for logging
    writer = setup(run_name, exp_args, [agent_args, policy_char_args, char_args, sv_args], seed)

    # env setup
    envs = setup_envs(env_args, exp_args, seed, run_name, **vars(env_args))

    # DQN agent
    if sys.argv[8] == "DQN":

        # Set up agent.
        agent_args.critic_arch = agent_args.shared_arch | agent_args.critic_arch
        agent = DQN(envs, agent_args)

        # Set up policy characteristic
        policy_char_args.model_arch = agent_args.shared_arch | policy_char_args.model_arch
        policy_char = PolicyCharacteristicModel(agent, envs, policy_char_args)

        # Set up characteristic
        char_args.model_arch = agent_args.shared_arch | char_args.model_arch
        char = PerformanceCharacteristicOffPolicy(agent, envs, char_args, policy_char)

        # Set up Shapley
        sv_args.model_arch = agent_args.shared_arch | sv_args.model_arch
        shapley = PerformanceShapley(envs, sv_args, char)

        # Train all models - in this order to ensure we have exact value for characteristics used by other models.
        train_agent(agent, [policy_char, char, shapley], envs, agent_args, writer, seed, run_name)

    # DQN combined with Shapley (sys.argv[8] == 'XDQN_S')
    else:

        # Set up agent.
        agent_args.exact_loss = getattr(sv_args, 'exact_loss', False)
        agent = XDQN_S(envs, agent_args, **{PerformanceShapley.__name__: sv_args})

        # Set up policy characteristic
        policy_char_args.model_arch = agent_args.shared_arch | policy_char_args.model_arch
        policy_char = PolicyCharacteristicModel(agent, envs, policy_char_args)

        # Set up characteristic
        char_args.model_arch = agent_args.shared_arch | char_args.model_arch
        char = PerformanceCharacteristicOffPolicy(agent, envs, char_args, policy_char)
        agent.add_chars({PerformanceShapley.__name__: char})

        # Train all models
        train_agent(agent, [policy_char, char], envs, agent_args, writer, seed, run_name)

        envs.close()
        writer.close()