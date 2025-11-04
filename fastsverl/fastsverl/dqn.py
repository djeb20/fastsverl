# Required libraries
import pickle
from collections import namedtuple
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from fastsverl.utils import Buffer
from fastsverl.models import SimpleNN
from torch.distributions.categorical import Categorical

# To do:
    # - Add and check CNN functionality properly. E.g. choose_action would not work.

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQN_Super:
    """
    Superclass for DQN agents and performance characteristics.
    """
    def __init__(self, env, agent_args):

        # Create random policy for exploration
        self.pi_rand = torch.ones(env.single_action_space.n).to(DEVICE) / env.single_action_space.n

        # Hyperparameters
        self.args = agent_args
        self.epsilon = agent_args.start_e # Epsilon for epsilon-greedy exploration
        
        # Create an empty replay buffer
        if hasattr(agent_args, 'buffer_size'):
            self.buffer = Buffer(
                agent_args.buffer_size, 
                env,
                'obs', 
                'action', 
                'reward', 
                'n_obs', 
                'terminated', 
                'logprob',
                )
        
        # Initialise target critic with critic weights.
        self.target_critic_nlayers = len(list(self.target_critic.parameters()))
        for w, w_t in zip(
            itertools.islice(self.model.parameters(), self.target_critic_nlayers), 
            self.target_critic.parameters()
            ):
            w_t.data.copy_(w.data)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=agent_args.learning_rate)

        # Track whether the model has been updated (for recomputing Shapley values)
        self.updated = False

    def load_models(self, model_path, eval=False, buffer=False, epsilon=True):
        """
        Load models from the specified path.
        Assumes models, epsilon, and buffer are saved by FastSVERL's save_models function.
        """
        self.model.load_state_dict(torch.load(f"{model_path}/{self.__class__.__name__}.model", map_location=DEVICE, weights_only=True))
        self.target_critic.load_state_dict(torch.load(f"{model_path}/{self.__class__.__name__}.target_critic", map_location=DEVICE, weights_only=True))

        # Set to eval mode if specified
        if eval:
            self.model.eval()
            self.target_critic.eval()

        # Load buffer if specified
        if buffer:
            with open(f"{model_path}/buffer.pkl", 'rb') as f:
                self.buffer = pickle.load(f)

        # Load epsilon if specified
        if epsilon:
            self.epsilon = pickle.load(open(f"{model_path}/epsilon.pkl", 'rb'))

    def choose_action(self, obs, exp=True, action=None):
        """
        Given a state returns a chosen action given by policy.
        Has epsilon greedy exploration.
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

    def Q(self, obs):
        """Returns Q values for all actions given observations."""
        return self.model(obs).DQN.detach()
    
    def V(self, obs):
        """Returns V values for all states given observations."""
        return self.Q(obs).detach().max(dim=-1, keepdim=True).values
    
    def pi_greedy(self, obs):
        """Returns greedy policy"""
        q_values = self.Q(obs).detach()
        policies_unnormalised = q_values == q_values.max(dim=-1, keepdim=True).values
        return policies_unnormalised / policies_unnormalised.sum(dim=-1, keepdim=True)
    
    def pi_explore(self, obs): # pi(a|s) = epsilon / |A| + (1 - epsilon) * pi(a|s)
        """Returns epsilon-greedy policy"""
        return self.pi_rand * self.epsilon + self.pi_greedy(obs) * (1 - self.epsilon)
    
    def update(self, *args):
        # Track whether the model has been updated (for recomputing Shapley values)
        self.updated = True
        return self.backward(self.forward(*args))
    
    def forward(self, obs, actions, rewards, n_obs, terminations):
        """
        Function to compute the loss
        """
        
        with torch.no_grad():
            target_max, _ = self.target_critic(n_obs).DQN.max(dim=1)
            td_target = rewards + self.args.gamma * target_max * (1 - terminations)

        old_val = self.model(obs).DQN.gather(1, actions.unsqueeze(1)).squeeze()

        return namedtuple('LossTuple', ['DQN_loss'])(F.mse_loss(td_target, old_val))
    
    def backward(self, loss):
        """
        Function to update the model
        """

        # Backward pass and optimization
        self.optimizer.zero_grad()
        getattr(loss, f'{self.__class__.__name__}_loss').backward()
        self.optimizer.step()

        return loss

    def update_target(self):
        """
        Soft update target network parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        where τ is the soft update parameter.
        """

        for w, w_t in zip(itertools.islice(self.model.parameters(), self.target_critic_nlayers), self.target_critic.parameters()):
            w_t.data.copy_(w.data * self.args.tau + w_t.data * (1 - self.args.tau))

    def linear_schedule(self, step):
        """ Linear schedule for epsilon greedy exploration. """

        # fraction of the decay that has completed
        fraction = step / (self.args.exploration_fraction * self.args.total_timesteps)

        # linearly interpolate between start and end epsilon
        new_epsilon = self.args.start_e + (self.args.end_e - self.args.start_e) * fraction

        # ensure epsilon does not go below the target end value
        self.epsilon = max(new_epsilon, self.args.end_e)

    
class DQN(DQN_Super):
    """
    Deep Q-Network agent.
    """
    def __init__(self, env, agent_args):

        # Set input and output dimensions
        if not getattr(agent_args, 'cnn', False):
            # If not using CNN, flatten input
            agent_args.critic_arch['input'][1][0] = np.prod(env.single_observation_space.shape)
        
        # Only discrete actions supported
        agent_args.critic_arch['output_shape'] = (env.single_action_space.n,)

        # Create model and target critic
        self.model = SimpleNN('DQN', **agent_args.critic_arch).to(DEVICE)
        self.target_critic = SimpleNN('DQN', **agent_args.critic_arch).to(DEVICE)

        super().__init__(env, agent_args)