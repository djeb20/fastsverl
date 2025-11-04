"""
Training loops for agents and models.
"""

import copy
import json
import random
import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import gymnasium as gym
import pickle

from fastsverl.utils import Buffer, make_env, random_action

# TODO:
    # - In train model and train agent, we repreatedly call get_batch and get_obs. This is inefficient. Instead, call get_batch once and then pass the data to the models.

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------- General

def setup(run_name:str, exp_args, other_args, seed:int):
    """
    Sets up the training environment and logging.
    """

    if exp_args.track:
        import wandb

        # Initialize wandb
        wandb.init(
            project=exp_args.wandb_project_name,
            entity=exp_args.wandb_entity,
            sync_tensorboard=True,
            config={f"{args.__class__.__name__}": vars(args) for args in other_args},
            name=run_name,
            group=exp_args.group,
            save_code=True,
        )

    # Set up Tensorboard writer
    writer = SummaryWriter(f"runs/{run_name}")

    # Save hyperparameters
    for args in other_args:
        
        writer.add_text(
            f"hyperparameters_{args.__class__.__name__}",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        with open(f"runs/{run_name}/{args.__class__.__name__}.json", "w") as f:
            json.dump(vars(args), f)
            print(f"Hyperparameters saved to runs/{run_name}/{args.__class__.__name__}.json")

    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = exp_args.torch_deterministic
    torch.backends.cudnn.benchmark = not exp_args.torch_deterministic

    return writer

def setup_envs(env_args, exp_args, seed:int, run_name:str, save_parameters:bool=True, **kwargs):
    """
    Sets up vectorized environments and saves environment parameters.
    """

    # env setup
    envs = gym.vector.SyncVectorEnv( 
        [make_env(
            exp_args.env_id, 
            seed + i, 
            i, 
            exp_args.capture_video, 
            run_name, 
            **kwargs,
            ) for i in range(exp_args.num_envs)
        ]
    )

    # Save environment hyperparameters
    if save_parameters:
        with open(f"runs/{run_name}/{env_args.__class__.__name__}.json", "w") as f:
            json.dump(vars(env_args), f)
            print(f"Hyperparameters saved to runs/{run_name}/{env_args.__class__.__name__}.json")

    return envs

def get_data(model, buffer, batch_size, **kwargs):
    """
    Retrieves a batch of data from the buffer based on the model type.
    Different data is required for updating different models.
    """
    if model.__class__.__name__ in ['DQN', 'XDQN_S']:
        return buffer.sample(batch_size, 'obs', 'action', 'reward', 'n_obs', 'terminated')
    elif model.__class__.__name__ in ['PPO', 'XPPO']:
        return buffer.sample(batch_size, 'obs', 'action', 'logprob', 'value', 'advantage', 'returns', start=kwargs['start'])
    elif model.__class__.__name__ == 'PerformanceCharacteristicOnPolicy':
        if getattr(model.args, 'off_policy', False):
            return buffer.sample(batch_size, 'obs', 'reward', 'n_obs', 'terminated', 'e_obs', 'C', 'action', 'logprob')
        else: 
            return buffer.sample(batch_size, 'obs', 'reward', 'n_obs', 'terminated', 'e_obs', 'C')
    elif model.__class__.__name__ == 'PerformanceCharacteristicOffPolicy':
        if getattr(model.args, 'off_policy', False):
            return *buffer.sample(batch_size, 'obs', 'action', 'reward', 'n_obs', 'terminated', 'logprob'), *buffer.sample(batch_size, 'obs')
        else:
            return *buffer.sample(batch_size, 'obs', 'action', 'reward', 'n_obs', 'terminated'), *buffer.sample(batch_size, 'obs')
    elif model.__class__.__name__ in ['PolicyCharacteristicModel', 'ValueCharacteristicModel',
                                      'PolicyShapley', 'ValueShapley', 'PerformanceShapley']:
        # For off-policy importance sampling
        if getattr(model.args, 'off_policy', False):
            return buffer.sample(batch_size, 'obs', 'action', 'logprob')
        else:
            return buffer.sample(batch_size, 'obs')
    elif model.__class__.__name__ in ['XDQN_C']:
        return *buffer.sample(batch_size, 'obs', 'action', 'reward', 'n_obs', 'terminated', 'logprob'), *buffer.sample(batch_size, 'obs')
    else:
        raise ValueError(f"Model not recognised: {model.__class__.__name__}")
    
def fill_buffer(agent, envs, buffer, buffer_steps:int, policy:str, *args, choose_action=None):
    """
    Either top up a buffer until it is full or replace it with new experience.
    Policy used to choose actions: original policy, original policy with epsilon greedy, random or conditioned policy.

    *args are the names of the data to be stored in the buffer.
    
        Note: Steady-state approximation will be biased if episodes truncate.
    """

    if choose_action: 
        pass
    elif policy == 'greedy': 
        choose_action = lambda obs: agent.choose_action(obs, exp=False)
    elif policy == 'explore': 
        choose_action = lambda obs: agent.choose_action(obs, exp=True)
    elif policy == 'random': 
        choose_action = lambda obs: random_action(obs, envs)
    else: 
        raise ValueError('Policy not recognised.')

    # Reset environment
    obs, info = envs.reset()

    for _ in tqdm(range(buffer_steps // envs.num_envs), 'Filling Buffer') if getattr(agent.args, 'verbose', 0) == 1 else range(buffer_steps // envs.num_envs):
        
        # Select action according to policy
        action_info = choose_action(obs)
        action, logprob = action_info.action, action_info.logprob

        # Take step in environment
        n_obs, reward, terminated, truncated, info = envs.step(action)
        current_obs = n_obs.copy()

        if "final_info" in info:
            for idx, (info, final_observation) in enumerate(zip(info["final_info"], info["final_observation"])):
                if info and "episode" in info:

                    # Replace the next_obs with the final_observation if the episode terminates or truncates.
                    n_obs[idx] = final_observation

        # Store all relevant information in the buffer
        local_args = locals()
        buffer.add(**{k: local_args[k] for k in args})

        # Log buffer statistics
        if _ % int((buffer_steps // envs.num_envs) / 10) == 0 and getattr(agent.args, 'verbose', 0) == 1:
            print('\n', f"Unique obs={torch.unique(buffer.buffer['obs'][:buffer.size], dim=0).shape[0]}", '\n')

        obs = current_obs.copy()

    return buffer, buffer_steps

def train_agent(agent, *args, **kwargs):
    """Helper function to call the appropriate training loop based on agent type."""
    if agent.__class__.__name__ in ['DQN', 'XDQN_S', 'XDQN_C']:
        return train_dqn(agent, *args, **kwargs)
    elif agent.__class__.__name__ in ['PPO', 'XPPO']:
        return train_ppo(agent, *args, **kwargs)
    else:
        raise ValueError('Agent not recognised.')

# ------------------------- DQN

def train_dqn(dqn, updates, envs, agent_args, writer, seed:int, run_name:str):
    """
    Training loop for DQN agent.

    Args:
        dqn: DQN agent.
        updates: List of models to be trained alongside the DQN agent.
        envs: Vectorized environment.
        agent_args: Arguments for the agent.
        writer: Tensorboard writer.
        seed: Random seed.
        run_name: Name of the run.
    """

    # Reset the environment
    start_time = time.time()
    obs, _ = envs.reset(seed=seed)

    for step in tqdm(range(agent_args.total_timesteps // envs.num_envs), f"Training: {dqn.__class__.__name__}, " + ", ".join(update.__class__.__name__ for update in updates)):
        global_step = step * envs.num_envs
        
        # Choose action according to epsilon greedy policy
        dqn.linear_schedule(global_step)
        actions_info = dqn.choose_action(obs)

        # Execute action in the environment
        next_obs, rewards, terminations, truncations, infos = envs.step(actions_info.action)
        real_next_obs = next_obs.copy()

        # Log statistics for completed episodes
        if "final_info" in infos:
            for idx, (info, final_observation) in enumerate(zip(infos["final_info"], infos["final_observation"])):
                if info and "episode" in info:
                    if agent_args.verbose == 2:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}, episodic_length={info['episode']['l']}")
                        
                    writer.add_scalar("agent/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("agent/episodic_length", info["episode"]["l"], global_step)
                    writer.add_scalar("agent/epsilon", dqn.epsilon, global_step)

                    # Replace the next_obs with the final_observation if the episode terminates or truncates.
                    real_next_obs[idx] = final_observation

        # TRY NOT TO MODIFY: save data to reply buffer
        dqn.buffer.add(
            obs=obs, 
            action=actions_info.action, 
            reward=rewards, 
            n_obs=real_next_obs, 
            terminated=terminations, 
            logprob=actions_info.logprob,
            )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > agent_args.learning_starts:
            if global_step % agent_args.train_frequency < envs.num_envs:
                
                # Only for XDQN + parallel training. Reapproximate the steady-state 
                # distribution for computing the exact loss of the given characteristic
                # and Shapley values.
                if any([getattr(model.args, 'exact_loss', False) for model in [dqn] + updates]) and global_step % agent_args.exact_frequency < envs.num_envs:
                    exact_buffer = Buffer(agent_args.exact_buffer_size, envs, 'obs')
                    exact_buffer, _ = fill_buffer(dqn, copy.deepcopy(envs), exact_buffer, agent_args.exact_buffer_size, 'explore', 'obs')
                # -----------------------------------------------------------------------------------------
                
                # Update DQN and other models
                for model in [dqn] + updates:

                    # Perform multiple updates per step if specified
                    for _ in range(agent_args.update_rate) if model in updates and hasattr(agent_args, "update_rate") else range(1):
                        # All models share the same batch_size.
                        losses = model.update(*get_data(model, dqn.buffer, agent_args.batch_size))
                    
                    # Log losses
                    if global_step % 100 < envs.num_envs:
                        for name, loss in losses._asdict().items():
                            writer.add_scalar(f"agent/{name}", loss, global_step)
                        writer.add_scalar("agent/SPS", int(global_step / (time.time() - start_time)), global_step)

                    # Only for XDQN + parallel training: recompute the exact loss of the given characteristic and Shapley values.
                    if getattr(model.args, 'exact_loss', False) and global_step % agent_args.exact_frequency < envs.num_envs:
                        for name, loss in model.exact_loss(exact_buffer=exact_buffer)._asdict().items():
                            writer.add_scalar(f"agent/{name}", loss, global_step)
                    # -------------------------------------------------------------------------------------

            # update target network
            if global_step % agent_args.target_network_frequency < envs.num_envs:
                dqn.update_target()

        # Save models and buffer periodically
        if ((step + 1) * envs.num_envs) % (agent_args.total_timesteps // 10) < envs.num_envs:
            
            if getattr(agent_args, 'save_model', False):
                torch.save(dqn.model.state_dict(), f"runs/{run_name}/DQN.model")
                torch.save(dqn.target_critic.state_dict(), f"runs/{run_name}/DQN.target_critic")
                pickle.dump(dqn.epsilon, open(f"runs/{run_name}/epsilon.pkl", 'wb'))
                if agent_args.verbose == 2:
                    print(f"models saved to runs/{run_name}/DQN.model; runs/{run_name}/DQN.target_critic")

            if getattr(agent_args, 'save_buffer', False):
                with open(f"runs/{run_name}/buffer.pkl", 'wb') as f:
                    pickle.dump(dqn.buffer, f)
                if agent_args.verbose == 2:
                    print(f"buffer saved to runs/{run_name}/buffer.pkl")

    # Evaluation after training
    if agent_args.eval_episodes:
        episodic_returns = evaluate_dqn(dqn, envs, agent_args)

        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("agent/eval_episodic_return", episodic_return, idx)

    # Set models to eval mode after training
    dqn.model.eval()

def evaluate_dqn(dqn, envs, agent_args):
    """
    Evaluation loop for DQN agent.
    """

    obs, _ = envs.reset()
    episodic_returns = []

    while len(episodic_returns) < agent_args.eval_episodes:
        actions_info = dqn.choose_action(obs, exp=False)

        obs, _, _, _, infos = envs.step(actions_info.action)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                if agent_args.verbose == 1:
                    print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]

    return episodic_returns

# ------------------------- PPO

def train_ppo(ppo, updates, envs, agent_args, writer, seed:int, run_name:str):
    """
    Training loop for PPO agent.

    Args:
        ppo: PPO agent.
        updates: List of models to be trained alongside the PPO agent.
        envs: Vectorized environment.
        agent_args: Arguments for the agent.
        writer: Tensorboard writer.
        seed: Random seed.
        run_name: Name of the run.
    """

    # TODO:
        # - There are duplicates between values and n_values.
        # - get_action_and_value is sometimes being called twice for the same next_obs/real_next_obs.

    # TRY NOT TO MODIFY: initialization
    global_step = 0
    start_time = time.time()  
    obs, _ = envs.reset(seed=seed)
    obs = torch.Tensor(obs).to(DEVICE)

    for iteration in tqdm(range(1, agent_args.num_iterations + 1), f"Training: {ppo.__class__.__name__}, " + ", ".join(update.__class__.__name__ for update in updates)):
        
        # Annealing the rate if instructed to do so.
        if getattr(agent_args, "anneal_lr", False):
            ppo.anneal_lr(iteration)

        # Annealing the entropy coefficient if instructed to do so.
        if getattr(agent_args, "anneal_entropy", False):
            ppo.anneal_entropy(iteration)

        for step in range(0, agent_args.num_steps):
            global_step += envs.num_envs

            # ALGO LOGIC: action logic
            with torch.no_grad():
                actions, logprobs, _, values = ppo.get_action_and_value(obs)

            # TRY NOT TO MODIFY: execute the action in the environment
            next_obs, rewards, terminations, truncations, infos = envs.step(actions.cpu().numpy())
            real_next_obs = next_obs.copy()

            # Log statistics for completed episodes
            if "final_info" in infos:
                for idx, (info, final_observation) in enumerate(zip(infos["final_info"], infos["final_observation"])):
                    if info and "episode" in info:
                        if agent_args.verbose == 2:
                            print(f"global_step={global_step}, episodic_return={info['episode']['r']}, episodic_length={info['episode']['l']}")
                        writer.add_scalar("agent/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("agent/episodic_length", info["episode"]["l"], global_step)

                        # Replace the next_obs with the final_observation if the episode terminates or truncates.
                        real_next_obs[idx] = final_observation

            # Get n_values for GAE
            with torch.no_grad():
                _, _, _, n_values = ppo.get_action_and_value(torch.Tensor(real_next_obs).to(DEVICE))

            # Save experience to memory for GAE
            ppo.memory.add(obs=obs, action=actions, reward=rewards, terminated=terminations, truncated=truncations, logprob=logprobs, value=values.flatten(), n_value=n_values.flatten())

            # Saving experience to compute SVERL
            if hasattr(ppo, 'buffer'):
                ppo.buffer.add(obs=obs, action=actions, reward=rewards, n_obs=real_next_obs, terminated=terminations, logprob=logprobs)

            obs = torch.Tensor(next_obs).to(DEVICE)

        # ALGO LOGIC: compute returns and advantages
        rewards, terminations, truncations, values, n_values = ppo.memory.sample(
            ppo.memory.size, 'reward', 'terminated', 'truncated', 'value', 'n_value', start=0
            )
        returns, advantages = ppo.gae(rewards, terminations, truncations, values, n_values)
        ppo.memory.add_buffer(returns=returns, advantage=advantages)
    
        # Optimizing the policy and value network
        for epoch in range(agent_args.update_epochs):
            ppo.memory.shuffle()
            for start in range(0, agent_args.batch_size, agent_args.minibatch_size):
                ppo_losses = ppo.update(*get_data(ppo, ppo.memory, agent_args.minibatch_size, start=start))
                other_losses = [
                    model.update(*get_data(model, ppo.buffer, agent_args.minibatch_size)) for model in updates
                    ]

            # Early stopping if the KL divergence exceeds the target
            if agent_args.target_kl is not None and ppo_losses.approx_kl > agent_args.target_kl:
                break

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for name, loss in ppo_losses._asdict().items():
            writer.add_scalar(f"agent/{name}", loss, global_step)
        for losses in other_losses:
            for name, loss in losses._asdict().items():
                writer.add_scalar(f"agent/{name}", loss, global_step)
        writer.add_scalar("agent/actor_learning_rate", ppo.actor_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("agent/critic_learning_rate", ppo.critic_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("agent/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Only for XPPO: reapproximate the steady-state and compute the exact loss of the given characteristic and Shapley values.
        if any([getattr(model.args, 'exact_loss', False) for model in [ppo] + updates]) and global_step % agent_args.exact_frequency < envs.num_envs:
            exact_buffer = Buffer(agent_args.exact_buffer_size, envs, 'obs')
            exact_buffer, _ = fill_buffer(ppo, copy.deepcopy(envs), exact_buffer, agent_args.exact_buffer_size, 'explore', 'obs')

        for model in [ppo] + updates:
            if getattr(model.args, 'exact_loss', False) and global_step % agent_args.exact_frequency < (envs.num_envs * agent_args.num_steps):
                for name, loss in model.exact_loss(exact_buffer=exact_buffer)._asdict().items():
                    writer.add_scalar(f"agent/{name}", loss, global_step)
        # ------------------------------------------------------------------------------------------                
        # Save models and buffer periodically
        if iteration % (agent_args.num_iterations // 10) == 0:
            
            if getattr(agent_args, 'save_model', False):
                torch.save(ppo.model.state_dict(), f"runs/{run_name}/PPO.model")
                torch.save(ppo.critic.state_dict(), f"runs/{run_name}/PPO.critic")
                if agent_args.verbose == 2:
                    print(f"models saved to runs/{run_name}/PPO.model; runs/{run_name}/PPO.critic")

            if getattr(agent_args, 'save_buffer', False): 
                with open(f"runs/{run_name}/buffer.pkl", 'wb') as f:
                    pickle.dump(ppo.buffer, f)
                if agent_args.verbose == 2:
                    print(f"buffer saved to runs/{run_name}/buffer.pkl")

    # Set models to eval mode after training
    ppo.model.eval()
    ppo.critic.eval()

# ------------------------- Performance Characteristics

def train_V_on_policy(V, envs, char_args, writer, seed:int, run_name:str):
    """
    Learns V(s|C, s_e) "on-policy". Although joint environment interaction and value updates are not necessary. The same code would work if a buffer was first collected following pi hat and then experience was sampled without further interaction. 
    """

    # TODO:
        # - Sample new s_e and C every episode. Need to look into vectorized envs.
        # - Currently assume char has buffer that covers state space.

    # e_obs (s_e) and C are the state and coalition the policy is conditioned on.
    sample = lambda: (
        V.char.train_buffer.sample(envs.num_envs, 'obs')[0], 
        V.sampler.sample_rand(1).expand(envs.num_envs, -1)
        ) # Using different C accross envs might learn faster.
    e_obs, C = sample()
    
    # Initialization
    start_time = time.time()
    obs, _ = envs.reset(seed=seed)
    obs = torch.Tensor(obs).to(DEVICE)

    for step in tqdm(range(char_args.total_timesteps // envs.num_envs), 'Training V On Policy'):
        global_step = step * envs.num_envs

        # Choose action according to the current policy conditioned on s_e and C.
        actions_info = V.choose_action(obs, e_obs, C)

        # Execute action in the environment
        n_obs, rewards, terminations, truncations, infos = envs.step(actions_info.action)
        real_n_obs = n_obs.copy()

        # Replace the next_obs with the final_observation if the episode is truncated.
        if "final_info" in infos:
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_n_obs[idx] = infos["final_observation"][idx]

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`.
        # Action is not used.
        V.buffer.add(
            obs=obs, action=actions_info.action, reward=rewards, n_obs=real_n_obs, terminated=terminations, e_obs=e_obs, C=C
            )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = torch.Tensor(n_obs).to(DEVICE)

        # Sample new s^e and C if needed.
        if char_args.sampling_frequency == 'decision': # Currently always true.
            e_obs, C = sample()
            
        # ALGO LOGIC: training.
        if global_step % char_args.train_frequency < envs.num_envs:
            losses = V.update(*get_data(V, V.buffer, char_args.batch_size))

            # Compute and log losses
            if global_step % 100 < envs.num_envs:
                for name, loss in losses._asdict().items():
                    writer.add_scalar(f"model/{name}", loss, global_step)
                if getattr(char_args, 'exact_loss', False):
                    for name, exact_loss in V.exact_loss()._asdict().items():
                        writer.add_scalar(f"model/{name}", exact_loss, global_step)
                    # Exiting because of exact loss (only Hypercube).
                    if hasattr(char_args, 'exact_condition'):
                        if exact_loss < char_args.exact_condition:
                            break
                if char_args.verbose == 2:
                    print(f"global_step={global_step}, loss={loss}")
                writer.add_scalar("model/SPS", int(global_step / (time.time() - start_time)), global_step)

        # update target network
        if global_step % char_args.target_network_frequency < envs.num_envs:
            V.update_target()

        # Save model periodically
        if getattr(char_args, 'save_model', False) and ((step + 1) * envs.num_envs) % (char_args.total_timesteps // 10) < envs.num_envs:
                torch.save(V.model.state_dict(), f"runs/{run_name}/{V.__class__.__name__}.model")
                if char_args.verbose:
                    print(f"model saved to runs/{run_name}/{V.__class__.__name__}.model")

    # Set model to eval mode after training
    V.model.eval()

# ------------------------- Supervised Learning

def train_models(updates, args, writer, run_name:str):
    """
    Trains multiple models using supervised learning for a set number of steps.
    """

    # For early stopping.
    best_losses = torch.full((len(updates),), float('inf'), device=DEVICE)
    best_models = [None] * len(updates)
    best_epochs = torch.zeros(len(updates), dtype=torch.int64, device=DEVICE)
    
    n_batches = args.train_buffer_size // args.batch_size
    start_time = time.time()

    for epoch in tqdm(range(1, args.epochs + 1), "Training: " + ", ".join(update.__class__.__name__ for update in updates)):

        train_losses = np.zeros((n_batches, len(updates)), dtype=np.float32)
        for batch in range(1, n_batches + 1):

            # Train all models on the same batch.
            for i, model in enumerate(updates):

                train_losses[batch - 1][i] = getattr(
                    model.update(*get_data(model, model.train_buffer, args.batch_size)), 
                    f'{model.__class__.__name__}_loss'
                    ) # Models share batch_size.
                
                # Log batch losses for fair comparison if needed.
                if getattr(args, 'save_batch', False):
                    writer.add_scalar(f"model/{model.__class__.__name__}_batch_loss", train_losses[batch - 1][i], batch + epoch * n_batches)
        
        # Validation losses.
        val_losses = torch.tensor([getattr(
            model.forward(*get_data(model, model.val_buffer, model.args.val_buffer_size)), 
            f'{model.__class__.__name__}_loss') 
            for model in updates], device=DEVICE)
        
        # Compute and save losses.
        for model, train_loss, val_loss in zip(updates, np.mean(train_losses, axis=0), val_losses):
            save_step = (epoch + 1) * n_batches if getattr(args, 'track_batch', False) else epoch
            if getattr(model.args, 'exact_loss', False):
                for name, exact_loss in model.exact_loss()._asdict().items():
                    writer.add_scalar(f"model/{name}", exact_loss, save_step)
            writer.add_scalar(f"model/{model.__class__.__name__}_epoch_loss", train_loss, save_step)

            # Log validation loss if needed.
            if getattr(args, 'save_val', False):
                writer.add_scalar(f"model/{model.__class__.__name__}_val_loss", val_loss, save_step)

            if model.args.verbose == 2:
                print(f"epoch={epoch}; model={model.__class__.__name__}; train_loss={train_loss}; val_loss={val_loss}")

        # Early stopping.
        improved = val_losses < best_losses
        best_losses[improved] = val_losses[improved]
        best_epochs[improved] = epoch
        best_models = [copy.deepcopy(model.model) if improved[i] else best_models[i] for i, model in enumerate(updates)]

        # Check stopping condition across all models       
        if ((epoch - best_epochs) >= torch.tensor([model.args.patience for model in updates], device=DEVICE)).all():
            print(f"Stopping at epoch {epoch}, all models reached patience limits.")
            break

        # Exiting because of exact loss (only Hypercube).
        if hasattr(args, 'exact_condition'):
            if exact_loss < args.exact_condition:
                break

        # Save models periodically
        if getattr(args, 'save_model', False) and (args.epochs < 10 or (epoch + 1) % (args.epochs // 10) == 0):
            for model in updates:
                model_path = f"runs/{run_name}/{model.__class__.__name__}.model"
                torch.save(model.model.state_dict(), model_path)
                if args.verbose == 2:
                    print(f"model saved to {model_path}")

    # Clean up.
    for model, best_model in zip(updates, best_models):
        model.model.load_state_dict(best_model.state_dict())
        model.model.eval()

    # Save final models and buffers
    if getattr(args, 'save_model', False):
        for model in updates:
            torch.save(model.model.state_dict(), f"runs/{run_name}/{model.__class__.__name__}.model")
            print(f"model saved to runs/{run_name}/{model.__class__.__name__}.model")

    if getattr(args, 'save_train_buffer', False):
        with open(f"runs/{run_name}/train_buffer.pkl", 'wb') as f:
            pickle.dump(model.train_buffer, f)
        print(f"train_buffer saved to runs/{run_name}/train_buffer.pkl")

    if getattr(args, 'save_exact_buffer', False):
        with open(f"runs/{run_name}/exact_buffer.pkl", 'wb') as f:
            pickle.dump(model.exact_buffer, f)
        print(f"exact_buffer saved to runs/{run_name}/exact_buffer.pkl")