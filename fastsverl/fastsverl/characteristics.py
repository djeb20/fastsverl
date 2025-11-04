from collections import defaultdict, namedtuple
import pickle

from tqdm import tqdm

import numpy as np
from fastsverl.ppo import PPO_Super
from fastsverl.utils import ShapleySampler, value_iteration
from fastsverl.models import MultiHeadNN, SimpleNN, ImportanceWeightedMSELoss
from fastsverl.dqn import DQN_Super
import torch
from torch.distributions import Categorical

# Note:
    # - The on-policy performance characteristic will sample experience that is never possible under the policy \hat\pi. For example, it could be in a state and sample some e_obs and C but with that e_obs and C it was never possibe to reach the state it's in. Similarly, the off-policy version might do the same and sample a combination that is not possible. For the model version of the policy characteristic, it still produces output for that obs, e_obs and C because it will always produce output, but the output is somewhat meaningless. For the sample version on the other hand, if the agent is in a state it should never have reached then it won't be in train buffer and so there will be no matches---it will not know how to act. This is a limitation of SVERL's outcome explanations in general: how does an agent act in a state it never expected to be in?
    # - The null is recomputed every time, but it won't change if the agent's behaviour model doesn't change. So can be cached and reused.
    # - For masking, it would probably be better to have all inputs as one-hot encodings and then the mask value would be 0. Something like -2 is too "close" to 1 or 2, since neural nets will just interpolate and treat them as similar.
    # - When the data used to approximate p^pi(s) has states that only appear once, the approximated characteristic values differ noticeably from the exact values. This is evident from comparing the exact and approximate in Mastermind for PPO, for all types of characteristics. The exact loss does not converge to zero and it is because we are comparing the characteristic values at these rare states. The approximate models are more accurate at states more likely to be visited (large p^pi(s)).

# TODO:
    # - Make sure every tensor is on the GPU---across all files.
    # - A unified version of off-policy forwards because there is the same change in all classes.

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Characteristic:
    """
    Parent class for all characteristic:
        - Subclassed by model-based and sampling-based characteristics.
        - Subclassed by behaviour, prediction and performance characteristics.
    """

    # Unknown features are set to -2 in inputs to characteristic models.
    MASK = -2

    def __init__(self, agent, env, char_args, train_buffer=None, val_buffer=None, exact_buffer=None):

        self.agent = agent
        self.env = env
        self.F_card = np.prod(env.single_observation_space.shape) # Number of features
        self.sampler = ShapleySampler(self.F_card) # Sampling coalitions
        self.args = char_args

        # Whether exact performance char is computed with exact policy char.
        self.with_exact_char = getattr(char_args, 'with_exact_char', True)

        # Buffers for training, validation and exact characteristic values.
        self.train_buffer = train_buffer
        self.val_buffer = val_buffer
        self.exact_buffer = exact_buffer

class CharacteristicModel(Characteristic):
    def __init__(self, agent, env, char_args, train_buffer=None, val_buffer=None, exact_buffer=None, model=None):
        super().__init__(agent, env, char_args, train_buffer, val_buffer, exact_buffer)

        if model: # Load model if given. For MixedShapley + XDQN + XPPO.
            self.model = model
        else:
            # Set input and output dimensions for the model.
            char_args.model_arch['input'][1][0] = self.input_shape(env)
            char_args.model_arch['output_shape'] = self.output_shape(env)
            
            # Initialize model and optimizer
            self.model = SimpleNN(self.__class__.__name__, **char_args.model_arch).to(DEVICE)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=char_args.learning_rate)
        
        # Loss functions
        self.exact_loss_fn = torch.nn.MSELoss()

        # Different loss and forward functions for on-policy and off-policy training
        if getattr(char_args, 'off_policy', False):
            self.loss_fn = ImportanceWeightedMSELoss()
            self.forward = self.forward_off_policy
        else:
            self.loss_fn = torch.nn.MSELoss()
            self.forward = self.forward_on_policy

    def load_model(self, model_path, eval=False, train_buffer=False, exact_buffer=False):

        # Assumes model was saved by FastSVERL's save_model function.
        self.model.load_state_dict(torch.load(f"{model_path}/{self.__class__.__name__}.model", map_location=DEVICE, weights_only=True))
        
        # Set to eval mode
        if eval:
            self.model.eval()

        # Load buffers if specified
        if train_buffer:
            with open(f"{model_path}/train_buffer.pkl", "rb") as f:
                self.train_buffer = pickle.load(f)
        
        if exact_buffer:
            with open(f"{model_path}/exact_buffer.pkl", "rb") as f:
                self.exact_buffer = pickle.load(f)

    # General predict, update and backward functions
    def predict(self, *args):
        return getattr(self.model(*args), self.__class__.__name__)
    
    def update(self, *args, **kwargs):
        return self.backward(self.forward(*args, **kwargs))
    
    def backward(self, loss):
        """
        Function to update the model
        """

        # Backward pass and optimization
        self.optimizer.zero_grad()
        getattr(loss, f'{self.__class__.__name__}_loss').backward()
        self.optimizer.step()

        return loss

# -------------------------------------------- Behaviour and prediction characteristics --------------------------------------------

class ValuePolicyCharacteristic:
    """
    Parent class for behaviour and prediction characteristics.
        - Subclassed by model-based versions and sampling-based versions.
    """

    MASK = Characteristic.MASK

    def __init__(self, char_args):

        # Whether the exploration or greedy policy is being explained.
        self.explore = char_args.policy == 'explore'

    def get_exact(self, exact_buffer=None):
        """
        Calculates the exact characteristic values for policy and prediction.
            - Exact assumes exact_buffer is good approximation of p^pi(s).
        """

        # Compute new exact if given new buffer (e.g. if training parallel).
        if not hasattr(self, 'exact') or (exact_buffer and exact_buffer != self.exact_buffer):

            if exact_buffer:
                self.exact_buffer = exact_buffer

            # Get unique observed states and their empirical steady-state distribution
            e_obs, dist = torch.unique(self.exact_buffer.sample(self.exact_buffer.size, 'obs', start=0)[0], dim=0, return_counts=True)
            dist = dist.float() / dist.sum()

            self.exact = {}

            # Compute exact characteristic values for all coalitions and observed states
            all_C = torch.cartesian_prod(*torch.tensor(self.F_card * [[0, 1]]).to(DEVICE)).int()
            for C in (tqdm(all_C, f'Exact {self.__class__.__name__}') if getattr(self.args, 'verbose', False) else all_C):

                # Masked observed states
                m_obs = e_obs.masked_fill(~C.bool(), self.MASK)
                for m_ob in torch.unique(m_obs, dim=0):

                    # Get indexes of matching observations, compute using Bayes rule
                    indexes = (m_ob == m_obs).all(axis=1)
                    cond_dist = dist[indexes] / dist[indexes].sum()
                    self.exact[*m_ob.cpu().numpy()] = (self.v_F(e_obs[indexes]).detach() * cond_dist[:, None]).sum(axis=0)

            # print(f'Exact characteristic values: {self.exact}')

    def get_exact_val(self, e_ob, C):
        """Returns the exact characteristic value."""
        return self.exact[*e_ob.masked_fill(~C.bool(), self.MASK).cpu().numpy()]
    
    def null_exact(self, obs):
        """Returns the null characteristic value."""
        return self.exact[(self.MASK,) * self.F_card]

# -------------------------------------------- Model-based behaviour and prediction characteristics --------------------------------------------

class ValuePolicyCharacteristicModel(CharacteristicModel, ValuePolicyCharacteristic):
    """Model for behaviour and prediction characteristics."""
    def __init__(self, agent, env, char_args, train_buffer=None, val_buffer=None, exact_buffer=None, model=None):

        CharacteristicModel.__init__(self, agent, env, char_args, train_buffer, val_buffer, exact_buffer, model)
        ValuePolicyCharacteristic.__init__(self, char_args)

        # Clip importance sampling weights, for off-policy training.
        self.clip_coef = getattr(char_args, 'clip_coef', float('inf'))

        # Whether to weight the importance sampling weights, for off-policy training.
        self.weight_func = (lambda ratio: ratio / ratio.sum()) if getattr(char_args, 'weighting', True) else (lambda ratio: ratio)

    def input_shape(self, env):
        return np.prod(env.single_observation_space.shape)

    def get_val(self, e_obs, Cs):
        """ Returns the characteristic value for given states and coalitions."""
        return self.predict(e_obs.masked_fill(~Cs.bool(), self.MASK)).detach()

    def forward_on_policy(self, obs, **kwargs):
        """ Forward pass for on-policy training. """

        # Target values
        targets = self.v_F(obs)

        # Forward pass
        Cs = self.sampler.sample_rand(obs.shape[0])
        predictions = self.predict(obs.masked_fill(~Cs.bool(), self.MASK))

        return namedtuple('LossTuple', [f'{self.__class__.__name__}_loss'])(self.loss_fn(predictions, targets))
    
    def forward_off_policy(self, obs, actions, old_logprobs, **kwargs):
        """ Forward pass for off-policy training. """

        # Did not work:
            # - KL: weights = (old_logits.exp() * (old_logits - new_logits)).sum(dim=-1)
            # - KL: weights = (new_logits.exp() * (new_logits - old_logits)).sum(dim=-1)
            # - "Expected" IS: weights = (new_logits.exp() * (new_logits - old_logits).exp()).sum(dim=-1)
            # - "Mean" IS: weights = torch.mean((new_logits - old_logits).exp())
            # - Actions sampled from new policy instead of from buffer.

        # Target values
        targets = self.v_F(obs)

        # Forward pass
        Cs = self.sampler.sample_rand(obs.shape[0])
        predictions = self.predict(obs.masked_fill(~Cs.bool(), self.MASK))  

        # Importance sampling weights
        new_logprobs = self.agent.choose_action(obs, action=actions, exp=self.explore).logprob        
        ratio = (new_logprobs - old_logprobs).exp()

        # Weighted estimate
        ratio = self.weight_func(ratio)

        # Clipped importance sampling
        ratio = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)

        return namedtuple('LossTuple', [f'{self.__class__.__name__}_loss'])(self.loss_fn(predictions, targets, ratio))

    def exact_loss(self, exact_buffer=None):
        """
        Calculates the loss of the exact characteristic values for behaviour and prediction.
        """

        # Compute exact characteristic values if characteristic changes
        self.get_exact(exact_buffer)

        # Fetch exact and approximate values
        exact_values = torch.stack(list(self.exact.values())).to(DEVICE)
        approx_values = self.predict(torch.tensor(list(self.exact)).to(DEVICE))

        # Compute exact loss
        exact_loss = self.exact_loss_fn(exact_values, approx_values)
        return namedtuple('LossTuple', [f'{self.__class__.__name__}_exact_loss'])(exact_loss)

    def null(self, obs):
        """Returns the null characteristic value."""
        return self.predict(torch.full_like(obs, self.MASK)).detach()
    
class ValueCharacteristicModel(ValuePolicyCharacteristicModel):
    def __init__(self, agent, env, char_args, train_buffer=None, val_buffer=None, exact_buffer=None, model=None):
        super().__init__(agent, env, char_args, train_buffer, val_buffer, exact_buffer, model)
        
        # Prediction characteristic uses value function
        self.v_F = self.agent.V

    def output_shape(self, env):
        return (1, )

class PolicyCharacteristicModel(ValuePolicyCharacteristicModel):
    def __init__(self, agent, env, char_args, train_buffer=None, val_buffer=None, exact_buffer=None, model=None):
        super().__init__(agent, env, char_args, train_buffer, val_buffer, exact_buffer, model)
        
        # Behaviour characteristic uses policy function
        self.v_F = self.agent.pi_explore if self.explore else self.agent.pi_greedy

    def output_shape(self, env):
        return (env.single_action_space.n,)

# -------------------------------------------- Sampling-based behaviour and prediction characteristics --------------------------------------------
    
class ValuePolicyCharacteristicSample(Characteristic, ValuePolicyCharacteristic):
    """
    Sampling-based model for behaviour and prediction characteristics.
    """
    def __init__(self, agent, env, char_args, train_buffer=None, val_buffer=None, exact_buffer=None):
        Characteristic.__init__(self, agent, env, char_args, train_buffer, val_buffer, exact_buffer)
        ValuePolicyCharacteristic.__init__(self, char_args)

        # Size of the pool to sample from.
        self.pool_size = getattr(char_args, 'pool_size', self.train_buffer.size)

    def get_val(self, e_obs, Cs):
        """
        Returns the single-sample characteristic value estimate 
        for given states and coalitions.

        Assumes:
            - num_samples < rnd.shape[0]
            - e_obs.dim() > 1
            - Cs's rows are identical.
        """

        # Train buffer to sample from
        train_buf = self.train_buffer.sample(self.pool_size, 'obs', replace=False)[0]

        # Indexes of buffer that match the coalition
        cols = Cs[0].bool().nonzero(as_tuple=True)[0]
        matches = (train_buf[:, cols].unsqueeze(0) == e_obs[:, cols].unsqueeze(1)).all(dim=2)

        # Sampling from matches
        rnd = torch.where(matches, torch.rand(matches.shape, device=train_buf.device), self.MASK)

        # Select top-k indices (num_samples per row)
        topk_indices = rnd.topk(self.args.num_samples, dim=1).indices

        # Gather sampled elements and take mean along the sampling dimension
        return self.v_F(train_buf[topk_indices]).mean(dim=1)
    
    def null(self, obs):
        """Returns the null characteristic estimate: average value over training buffer."""
        return self.v_F(self.train_buffer.buffer['obs'][:self.train_buffer.size]).mean(dim=0)
    
class ValueCharacteristicSample(ValuePolicyCharacteristicSample):
    def __init__(self, agent, env, char_args, train_buffer=None, val_buffer=None, exact_buffer=None):
        super().__init__(agent, env, char_args, train_buffer, val_buffer, exact_buffer)

        # Value characteristic uses value function
        self.v_F = self.agent.V

class PolicyCharacteristicSample(ValuePolicyCharacteristicSample):
    def __init__(self, agent, env, char_args, train_buffer=None, val_buffer=None, exact_buffer=None):
        super().__init__(agent, env, char_args, train_buffer, val_buffer, exact_buffer)
        
        # Behaviour characteristic uses policy function
        self.v_F = self.agent.pi_explore if self.explore else self.agent.pi_greedy

# -------------------------------------------- Outcome characteristics --------------------------------------------

class PerformanceCharacteristic(CharacteristicModel, DQN_Super):
    """
    Parent class for the outcome characteristic model.
    1. On-policy variant subclassed by PerformanceCharacteristicOnPolicy.
    2. Off-policy variant subclassed by PerformanceCharacteristicOffPolicy.

    Performance characteristics subclass DQN because they are trained with DQN-like updates.
    """
    def __init__(self, agent, env, char_args, policy_char, train_buffer=None, val_buffer=None, model=None):

        # Initialize parent classes and set target critic
        CharacteristicModel.__init__(self, agent, env, char_args, train_buffer, val_buffer, exact_buffer=policy_char.exact_buffer, model=model)
        self.target_critic = SimpleNN('DQN', **char_args.model_arch).to(DEVICE)
        DQN_Super.__init__(self, env, char_args)

        # Behaviour characteristic
        self.char = policy_char

        # Whether the exploration or greedy policy is being explained.
        self.explore = policy_char.args.policy == 'explore'

    def input_shape(self, env):
        return 3 * np.prod(env.single_observation_space.shape)

    def pi(self, obs, e_obs, Cs):
        # Conditioned policy \hat\pi(a|s, s^e, C)
        return torch.where((obs == e_obs).all(dim=-1, keepdim=True), self.char.get_val(e_obs, Cs), self.char.v_F(obs))
    
    def exact_pi(self, obs, e_obs, Cs):
        # Conditioned policy \hat\pi(a|s, s^e, C) using exact behaviour characteristic values
        return torch.where((obs == e_obs).all(dim=-1, keepdim=True), self.char.get_exact_val(e_obs, Cs), self.char.v_F(obs))

    def choose_action(self, obs, e_obs, Cs):
        # Sample action from conditioned policy \hat\pi(a|s, s^e, C)
        policy = Categorical(self.pi(obs, e_obs, Cs))
        action = policy.sample()

        return namedtuple('ActionTuple', ['action', 'logprob'])(action.cpu().numpy(), policy.log_prob(action))

    def get_exact(self, exact_buffer=None):
        """
        Calculates the exact outcome characteristic values.
            - Can treat C's and e_obs as part of state and train value iteration for single policy. Much faster.
        """

        # Compute new exact if given new buffer (e.g. if training parallel).
        if not hasattr(self, 'exact') or (exact_buffer and exact_buffer != self.exact_buffer):

            # Update exact buffer if given
            if exact_buffer:
                self.exact_buffer = exact_buffer

            # Compute exact behaviour characteristic values if needed
            if self.with_exact_char:
                self.char.get_exact(exact_buffer) 

            # Ensure transition dynamics are available
            if not hasattr(self.env.envs[0].unwrapped, 'P'):
                self.env.envs[0].unwrapped.get_P()

            # Get unique observed states
            e_obs = torch.unique(self.exact_buffer.sample(self.exact_buffer.size, 'obs', start=0)[0], dim=0)

            self.exact = defaultdict(dict)

            # Compute exact characteristic values for all coalitions and observed states
            all_C = torch.cartesian_prod(*torch.tensor(self.F_card * [[0, 1]]).to(DEVICE)).int()
            for C in tqdm(all_C, f'Exact {self.__class__.__name__}') if getattr(self.args, 'verbose', False) else all_C:

                # Get unique observed states
                for e_ob in e_obs:

                    # Exact conditioned policy or not
                    if self.with_exact_char:
                        policy = lambda ob: self.exact_pi(torch.Tensor(ob).to(DEVICE), e_ob, C).cpu().numpy()
                    else:
                        policy = lambda ob: self.pi(torch.Tensor(ob).to(DEVICE), e_ob, C).cpu().numpy()
                    
                    # Value iteration to get exact Q-values
                    Q_table = value_iteration(self.env.unwrapped.get_attr('env')[0], gamma=self.args.gamma, policy=policy) # self.env.unwrapped.get_attr('env')[0]
                    self.exact[*C.cpu().numpy()][*e_ob.cpu().numpy()] = torch.Tensor(policy(e_ob) * Q_table[*e_ob.cpu().numpy()]).sum(dim=0, keepdims=True).to(DEVICE)

            # print(f'Exact characteristic values: {self.exact}')

    def get_exact_val(self, e_ob, C):
        """Returns the exact characteristic value."""
        return self.exact[*C.cpu().numpy()][*e_ob.cpu().numpy()]

    def exact_loss(self, exact_buffer=None):
        """
        Calculates the loss of the exact characteristic values for performance.
        """

        self.get_exact(exact_buffer)

        # Inefficient loops but cleaner code.
        exact_values = torch.Tensor([values for inner_dict in self.exact.values() for values in inner_dict.values()]).to(DEVICE)
        Cs = torch.Tensor([C for C in self.exact for _ in self.exact[C]]).to(DEVICE)
        e_obs = torch.Tensor([e_obs for inner_dict in self.exact.values() for e_obs in inner_dict]).to(DEVICE)
        
        # Get approximate values
        approx_values = self.get_val(e_obs, Cs).squeeze()

        # Compute exact loss
        exact_loss = self.exact_loss_fn(exact_values, approx_values)
        return namedtuple('LossTuple', [f'PerformanceCharacteristic_exact_loss'])(exact_loss)

    def null(self, obs):
        """Returns the null characteristic value."""
        return self.get_val(obs, torch.zeros_like(obs).to(DEVICE))
    
    def null_exact(self, obs):
        """Returns the null exact characteristic value."""
        return self.exact[(0,) * self.F_card][*obs.cpu().numpy()]

class PerformanceCharacteristicOnPolicy(PerformanceCharacteristic):
    """On-policy variant of the outcome characteristic model."""
    def __init__(self, agent, env, char_args, policy_char, model=None):
        super().__init__(agent, env, char_args, policy_char, model)

        # Additional observations and coalitions to standard DQN buffer
        self.buffer.add_buffer('e_obs', 'C')

    def output_shape(self, env):
        return (1, )

    def get_val(self, e_obs, Cs):
        """Returns the characteristic value for given states and coalitions."""
        return self.predict(e_obs, e_obs, Cs).detach()

    def forward_on_policy(self, obs, rewards, n_obs, terminations, e_obs, Cs):
        """
        On-policy and goal-conditioned variant of DQN update.
        """

        # Target values
        targets = rewards + self.args.gamma * self.target_critic(n_obs, e_obs, Cs).DQN.squeeze().detach() * (1 - terminations)

        # Forward pass
        predictions = self.predict(obs, e_obs, Cs).squeeze()

        return namedtuple('LossTuple', [f'{self.__class__.__name__}_loss'])(self.loss_fn(predictions, targets))
    
class PerformanceCharacteristicOffPolicy(PerformanceCharacteristic):
    """Off-policy variant of the outcome characteristic model."""
    def __init__(self, agent, env, char_args, policy_char, train_buffer=None, val_buffer=None, model=None):
        super().__init__(agent, env, char_args, policy_char, train_buffer, val_buffer, model)

    def output_shape(self, env):
        return (env.single_action_space.n,)

    def get_val(self, e_obs, Cs):
        # V_s(C) = sum_a [pi^a_s(C) * Q(s, a|s^e, C)]
        return (self.pi(e_obs, e_obs, Cs) * self.predict(e_obs, e_obs, Cs)).sum(dim=-1, keepdims=True).detach()
    
    def forward_on_policy(self, obs, actions, rewards, n_obs, terminations, e_obs, **kwargs):
        """
        Off-policy and goal-conditioned variant of DQN update.
        """

        Cs = self.sampler.sample_rand(obs.shape[0])
    
        # Target values
        n_actions = Categorical(self.pi(n_obs, e_obs, Cs)).sample()
        targets = rewards + self.args.gamma * self.target_critic(n_obs, e_obs, Cs).DQN.gather(1, n_actions.unsqueeze(1)).squeeze().detach() * (1 - terminations)

        # Target values - Alternative formulation
        # n_v = (self.pi(n_obs, e_obs, Cs) * self.target_critic(n_obs, e_obs, Cs).DQN.detach()).sum(dim=-1).detach()
        # targets = rewards + self.args.gamma * n_v * (1 - terminations)

        # Forward pass
        predictions = self.predict(obs, e_obs, Cs).gather(1, actions.unsqueeze(1)).squeeze()

        return namedtuple('LossTuple', [f'{self.__class__.__name__}_loss'])(self.loss_fn(predictions, targets))

    def update(self, *args):
        losses = super().update(*args)
        self.update_target()
        return losses

# -------------------------------------------- XDQN and XPPO Characteristic --------------------------------------------
class MixedCharacteristic:
    """
    Wrapper for combining multiple characteristics.
    """

    def __init__(self, agent, env, **kwargs):
        """
        kwargs = {'__class__.__name__': {char_args:, train_buffer:, val_buffer:, exact_buffer:, model:}, ...}
        Each value may or may not contain...
        """

        # Characteristic models for each explanation
        self.chars = {class_name: globals()[class_name](agent, env, **values) for class_name, values in kwargs.items()}

    def load_model(self, model_paths, eval=False, train_buffer=False, exact_buffer=False):
        """ Loads models for all characteristics. """
        for char, model_path in zip(self.chars.values(), model_paths):
            char.load_model(model_path, eval, train_buffer, exact_buffer)
        
    def forward(self, **kwargs):
        """
        Forward pass for all characteristics.
        """

        # Collect loss names and values
        char_losses = {
            name: loss
            for char in self.chars.values()
            for name, loss in char.forward(**kwargs)._asdict().items()
        }

        return namedtuple('LossTuple', [
            f'MixedCharacteristic_loss',
            *char_losses.keys(),
        ])(
            sum(char_losses.values()),
            **char_losses,
        )
    
    def exact_loss(self, exact_buffer=None):
        """
        Computes the exact losses for the characteristics.
        """

        # Collect loss names and values
        losses = {}
        for char in self.chars.values():
            if char.args.exact_loss:
                losses |= char.exact_loss(exact_buffer)._asdict()

        return namedtuple('LossTuple', [*losses.keys()])(**losses)
    
# -------------------------------------------- XDQN-Characteristic --------------------------------------------
class XDQN_C(DQN_Super, MixedCharacteristic):
    """
    XDQN_C combines DQN model with multiple characteristic models.
    """
    def __init__(self, env, agent_args, **kwargs):
        """
        kwargs = {'__class__.__name__': char_args, ...}
        We assume the first head is the DQN head, that is what is copied to the target critic.
        Assume at least one Characteristic head is given in kwargs.
        """

        # Set input dimensions for the shared layers 
        if not getattr(agent_args, 'cnn', False):
            agent_args.shared_arch['input'][1][0] = np.prod(env.single_observation_space.shape)

        # Set model architectures and output shapes for the heads.
        heads = {'DQN': agent_args.critic_arch}
        heads_output_shapes = [(env.single_action_space.n,)]

        for class_name, char_args in kwargs.items():
            heads[class_name] = char_args.model_arch
            heads_output_shapes.append(globals()[class_name].output_shape(globals()[class_name], env))

        # Initialize the model and target critic
        self.model = MultiHeadNN(agent_args.shared_arch, heads, heads_output_shapes).to(DEVICE)
        self.target_critic = SimpleNN('DQN', **agent_args.shared_arch, **agent_args.critic_arch, output_shape=(env.single_action_space.n,)).to(DEVICE)

        # Set each explanation model to the agent model.
        for class_name, char_args in kwargs.items():
            kwargs[class_name] = {'char_args': char_args, 'model': self.model}

        # Initialise the super classes
        DQN_Super.__init__(self, env, agent_args)
        MixedCharacteristic.__init__(self, self, env, **kwargs)

    def forward(self, obs, actions, rewards, n_obs, terminations, old_logprobs, e_obs):
        # TODO:
            # - old_logprobs needs to be logprobs for this to work with PPO.

        # Compute all losses
        dqn_losses = DQN_Super.forward(self, obs, actions, rewards, n_obs, terminations)
        char_losses = MixedCharacteristic.forward(self, obs=obs, actions=actions, rewards=rewards, n_obs=n_obs, 
                                                  terminations=terminations, old_logprobs=old_logprobs, e_obs=e_obs)

        # Combine losses
        return namedtuple('LossTuple', [
            'XDQN_C_loss',
            *dqn_losses._fields,
            *char_losses._fields,
            ]
        )(
            dqn_losses.DQN_loss + char_losses.MixedCharacteristic_loss,
            **dqn_losses._asdict(),
            **char_losses._asdict(),
        )

# -------------------------------------------- XPPO-Characteristic --------------------------------------------

# TODO: Work in progress. XPPO_C combines PPO model with multiple characteristic models.
    # Will parallel XDQN_C closely.
    # - Decide if actor or critic should be "model" that is combined. Currently actor.

# class XPPO_C(PPO_Super, MixedCharacteristic):
#     def __init__(self, env, agent_args, **kwargs):
#         """
#         kwargs = {'__class__.__name__': char_args, ...}
#         Assume at least one Characteristic head is given in kwargs.
#         """

#         # Set input dimensions for the critic and the shared layers of the actor
#         agent_args.critic_arch['input'][1][0] = np.prod(env.single_observation_space.shape)
#         agent_args.shared_arch['input'][1][0] = np.prod(env.single_observation_space.shape)

#         # Set model architectures and output shapes for the heads.
#         heads = {'PPO': agent_args.actor_arch}
#         heads_output_shapes = [(env.single_action_space.n,)]

#         for class_name, char_args in kwargs.items():
#             heads[class_name] = char_args.model_arch
#             heads_output_shapes.append(globals()[class_name].output_shape(globals()[class_name], env))

#         # Initialize the shared actor model and the critic
#         self.model = MultiHeadNN(agent_args.shared_arch, heads, heads_output_shapes).to(DEVICE)
#         self.critic = SimpleNN('PPO', **agent_args.critic_arch, output_shape=(1,)).to(DEVICE)

#         # Set each explanation model to the agent model.
#         for class_name, char_args in kwargs.items():
#             kwargs[class_name] = {'char_args': char_args, 'model': self.model}

#         # Initialise the super classes
#         PPO_Super.__init__(self, env, agent_args)
#         MixedCharacteristic.__init__(self, env, **kwargs)

#     def forward(self, mb_obs, mb_actions, mb_logprobs, mb_values, mb_advantages, mb_returns):

#         ppo_losses = PPO_Super.forward(self, mb_obs, mb_actions, mb_logprobs, mb_values, mb_advantages, mb_returns)
        
#         # To do: Fix commented out
#         # char_losses = MixedCharacteristic.forward(self, ...)

#         return namedtuple('LossTuple', [
#             'XPPO_C_loss',
#             *ppo_losses._fields,
#             *char_losses._fields,
#             ]
#         )(
#             ppo_losses.PPO_loss + char_losses.MixedCharacteristic_loss,
#             **ppo_losses._asdict(),
#             **char_losses._asdict(),
#         )