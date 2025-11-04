from collections import namedtuple
import pickle
import numpy as np
import torch
import torch.nn as nn
from fastsverl.utils import Buffer
from fastsverl.models import SimpleNN
from torch.distributions.categorical import Categorical

# TODO:
    # - Combine get_action_and_value and choose_action. 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PPO_Super:
    """
    Superclass for PPO agents.
    """

    def __init__(self, env, agent_args):

        # Hyperparameters
        self.args = agent_args

        # Entropy coefficient
        self.ent_coef = agent_args.ent_coef

        # Store experience for updating
        self.memory = Buffer(
            agent_args.num_steps * env.num_envs, 
            env,
            'obs', 
            'action', 
            'reward', 
            'terminated', 
            'truncated', 
            'logprob', 
            'value', 
            'n_value',
        )

        # Store experience for computing SVERL in parallel
        if hasattr(agent_args, 'sverl_buffer_size') and agent_args.sverl_buffer_size:
            self.buffer = Buffer(
                agent_args.sverl_buffer_size, 
                env,
                'obs', 
                'action', 
                'reward', 
                'n_obs', 
                'terminated', 
                'logprob',
            )

        # For converting flat experience to vectorised experience (for gae)
        self.resize = lambda x: x.view(agent_args.num_steps, env.num_envs, *x.shape[1:])

        # Separate optimizers for actor and critic
        self.actor_optimizer = torch.optim.Adam(self.model.parameters(), lr=agent_args.actor_learning_rate, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=agent_args.critic_learning_rate, eps=1e-5)

        # Track whether the model has been updated (for recomputing Shapley values)
        self.updated = False

    def load_models(self, model_path, eval=False, buffer=False):
        """
        Load models from the specified path.
        Assumes models and buffer are saved by FastSVERL's save_models function.
        """
        self.model.load_state_dict(torch.load(f"{model_path}/{self.__class__.__name__}.model", map_location=DEVICE, weights_only=True))
        self.critic.load_state_dict(torch.load(f"{model_path}/{self.__class__.__name__}.critic", map_location=DEVICE, weights_only=True))
        
        # Set to eval mode if specified
        if eval:
            self.model.eval()
            self.critic.eval()

        # Load buffer if specified
        if buffer and hasattr(self, 'buffer'):
            with open(f"{model_path}/buffer.pkl", 'rb') as f:
                self.buffer = pickle.load(f)

    def get_action_and_value(self, x, action=None):
        """
        Get action, value, and logprob from the model.
        """

        # Get policy
        policy = Categorical(self.model(x).PPO)

        # Sample action, unless specified
        if action is None:
            action = policy.sample()

        return action, policy.log_prob(action), policy.entropy(), self.critic(x).PPO
    
    def choose_action(self, obs, exp=None, action=None):
        """
        Given a state returns a chosen action given by policy.
        Has softmax exploration.
        """

        # Get policy
        if exp:
            policy = Categorical(self.pi_explore(torch.Tensor(obs).to(DEVICE)))
        else:
            policy = Categorical(self.pi_greedy(torch.Tensor(obs).to(DEVICE)))
        
        # Sample action, unless specified
        if action is None:
            action = policy.sample()

        return namedtuple('ActionTuple', ['action', 'logprob'])(action.cpu().numpy(), policy.log_prob(action))
    
    def V(self, obs):
        """Get value from the critic."""
        return self.critic(obs).PPO.detach()
    
    def pi_explore(self, obs):
        """Returns softmax policy"""
        return self.model(obs).PPO.detach()
    
    def pi_greedy(self, obs):
        """Returns greedy policy"""
        pi = self.model(obs).PPO.detach()
        return (pi == pi.max(dim=-1, keepdim=True).values).float()
    
    def anneal_lr(self, iteration):
        """
        Anneal learning rate based on current iteration.
        """

        # Compute new learning rates
        frac = 1.0 - (iteration - 1.0) / self.args.num_iterations
        actor_lrnow = frac * self.args.actor_learning_rate
        critic_lrnow = frac * self.args.critic_learning_rate
        
        # Update learning rates
        self.actor_optimizer.param_groups[0]["lr"] = actor_lrnow
        self.critic_optimizer.param_groups[0]["lr"] = critic_lrnow

    def anneal_entropy(self, iteration):
        """
        Anneal entropy coefficient based on current iteration.
        """

        # Compute new entropy coefficient
        frac = 1.0 - (iteration - 1.0) / self.args.num_iterations
        self.ent_coef = frac * self.args.ent_coef

    def gae(self, rewards, terminations, truncations, values, n_values):
        """
        Compute Generalized Advantage Estimation (GAE).
        Computes advantages and returns for given rewards and value estimates.
        """

        with torch.no_grad():

            # Resize tensors for parallel GAE computation
            resized_tensors = (self.resize(x) for x in (
                rewards, 
                1 - terminations, 
                1 - truncations, 
                values, 
                n_values
            ))
            rewards, nextnonterminal, nextnontruncated, values, n_values = resized_tensors

            # Don't bootstrap value if episode terminated
            # Don't update advantage if episode terminated/truncated
            nextnon_terminal_truncated = nextnonterminal * nextnontruncated
            advantages = torch.zeros_like(rewards).to(DEVICE)
            lastgaelam = 0

            for t in reversed(range(self.args.num_steps)):

                # Compute td error
                delta = rewards[t] + self.args.gamma * n_values[t] * nextnonterminal[t] - values[t]

                # Compute GAE advantage
                advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnon_terminal_truncated[t] * lastgaelam
            
            # Compute returns
            returns = advantages + values

        return returns, advantages
    
    def policy_loss(self, mb_advantages, ratio):
        """
        Compute PPO policy loss.
        """

        # Normalize advantages
        if self.args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # PPO clipped policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
        return torch.max(pg_loss1, pg_loss2).mean()
    
    def critic_loss(self, mb_values, mb_returns, newvalue):
        """
        Compute PPO critic loss.
        """

        newvalue = newvalue.view(-1)

        # PPO clipped value loss
        if self.args.clip_vloss:
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_clipped = mb_values + torch.clamp(
                newvalue - mb_values,
                -self.args.clip_coef,
                self.args.clip_coef,
            )
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            return 0.5 * v_loss_max.mean()
        else:
            # Unclipped value loss
            return 0.5 * ((newvalue - mb_returns) ** 2).mean()

    def update(self, *args):
        # Track whether the model has been updated (for recomputing Shapley values)
        self.updated = True
        return self.backward(self.forward(*args))

    def forward(self, mb_obs, mb_actions, mb_logprobs, mb_values, mb_advantages, mb_returns):
        """
        Function to update the actor and critic networks
        """

        # Get new logprobs, entropy, and value estimates
        _, newlogprob, entropy, newvalue = self.get_action_and_value(mb_obs, mb_actions.long())

        # Compute importance sampling ratio
        logratio = newlogprob - mb_logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()

        # Explained variance for critic
        if np.var(mb_returns.cpu().numpy()) == 0:
            explained_var = np.nan
        else:
            explained_var = 1 - np.var(mb_returns.cpu().numpy() - mb_values.reshape(-1).cpu().numpy()) / np.var(mb_returns.cpu().numpy())

        # Policy loss
        policy_loss = self.policy_loss(mb_advantages, ratio)

        # Value loss
        critic_loss = self.critic_loss(mb_values, mb_returns, newvalue)

        # Entropy loss
        entropy_loss = entropy.mean()

        # Total loss
        total_loss = policy_loss - self.ent_coef * entropy_loss + critic_loss * self.args.vf_coef

        return namedtuple('LossTuple', [
            'PPO_loss',
            'policy_loss', 
            'critic_loss',
            'entropy_loss',
            'approx_kl', 
            'old_approx_kl',
            'clipfrac', 
            'explained_var',
            'ent_coef',
            ])(
                PPO_loss=total_loss,
                policy_loss=policy_loss, 
                critic_loss=critic_loss, 
                entropy_loss=entropy_loss, 
                approx_kl=approx_kl,
                old_approx_kl=old_approx_kl, 
                clipfrac=clipfrac, 
                explained_var=explained_var,
                ent_coef=self.ent_coef,
            )

    def backward(self, loss):
        """
        Function to update the model
        """

        # Backward pass and optimization
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        getattr(loss, f'{self.__class__.__name__}_loss').backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

        self.critic_optimizer.step()
        self.actor_optimizer.step()

        return loss

class PPO(PPO_Super):
    """
    Proximal Policy Optimization (PPO) agent.
    """
    def __init__(self, env, agent_args):

        # Set input and output dimensions for actor and critic networks (flattened, not cnn)
        agent_args.actor_arch['input'][1][0] = np.prod(env.single_observation_space.shape)
        agent_args.actor_arch['output_shape'] = (env.single_action_space.n,)
        agent_args.critic_arch['input'][1][0] = np.prod(env.single_observation_space.shape)
        agent_args.critic_arch['output_shape'] = (1,)

        # Initialise models
        self.model = SimpleNN('PPO', **agent_args.actor_arch).to(DEVICE)
        self.critic = SimpleNN('PPO', **agent_args.critic_arch).to(DEVICE)

        super().__init__(env, agent_args)