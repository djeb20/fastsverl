from collections import namedtuple
import pickle
from fastsverl.dqn import DQN_Super
import torch
import numpy as np
from math import comb
from tqdm import tqdm
from fastsverl.ppo import PPO_Super
from fastsverl.models import MultiHeadNN, SimpleNN, ImportanceWeightedMSELoss

# Note:
    # - We use the same C for every observation in a batch. It would be better to have different each batch (converge to exact faster), but if they are the same then we can vectorise get_val for the sampling characteristics.

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Shapley:
    """
    Superclass for Shapley value models.
    Note: 1-d coalitions only (flat states).
    """
    def __init__(self, env, sv_args, char=None, train_buffer=None, val_buffer=None, model=None):

        self.args = sv_args

        # Whether exact Shapleys are computed using exact characteristics.
        self.with_exact_char = getattr(sv_args, 'with_exact_char', True)

        # Load characteristic if given, else using sampling-based characteristics.
        if char:
            self.char = char
            self.exact_buffer = char.exact_buffer
        # else: --- If code crashes, uncomment.
        #     self.exact_buffer = train_buffer
        
        # For approximating steady-state distribution.
        self.train_buffer = train_buffer
        self.val_buffer = val_buffer

        # Load model if given. For MixedShapley + XDQN + XPPO.
        if model:
            self.model = model
        else:
            # Set input and output dimensions for the model.
            sv_args.model_arch['input'][1][0] = np.prod(env.single_observation_space.shape)
            sv_args.model_arch['output_shape'] = self.output_shape(env)
            
            # Initialize model and optimizer.
            self.model = SimpleNN(self.__class__.__name__, **sv_args.model_arch).to(DEVICE)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=sv_args.learning_rate)

        # Loss functions, separate because one my using importance weighting.
        self.exact_loss_fn = torch.nn.MSELoss()
        self.loss_fn = torch.nn.MSELoss()

        # TODO: off-policy updates with sampling characteristics.
        # if getattr(sv_args, 'off_policy', False): -------- Work in progress
        #     self.loss_fn = ImportanceWeightedMSELoss()
        #     self.forward = self.forward_off_policy
        # else:
        #     self.loss_fn = torch.nn.MSELoss()
        #     self.forward = self.forward_on_policy
        
        
    def add_char(self, char):
        """
        Adds a characteristic model to the Shapley value model.
        """
        self.char = char

    def load_model(self, model_path, eval=False, train_buffer=False):
        """
        Load model from the specified path.
        Assumes model is saved by FastSVERL's save_models function.
        """
        self.model.load_state_dict(torch.load(f"{model_path}/{self.__class__.__name__}.model", map_location=DEVICE, weights_only=True))
        
        # Set to eval mode if specified
        if eval:
            self.model.eval()

        # Load train buffer if specified
        if train_buffer:
            with open(f"{model_path}/train_buffer.pkl", "rb") as f:
                self.train_buffer = pickle.load(f)

    def predict(self, *args):
        """ Predicts the Shapley values using the model. """
        return getattr(self.model(*args), self.__class__.__name__)

    def update(self, *args):
        return self.backward(self.forward(*args))
    
    def forward(self, obs):
        """
        On-policy forward pass to compute the Shapley value loss.
        """

        # Same C for entire batch. Using different C would likely converge to true values faster.
        Cs = self.char.sampler.sample(1).expand(obs.shape[0], -1)

        # Target values.
        targets = self.char.get_val(obs, Cs) - self.char.null(obs)

        # Forward pass
        forward = self.predict(obs)
        predictions = torch.sum(Cs.unsqueeze(-1) * forward, dim=1) # Sum over the coalition dimension.

        return namedtuple('LossTuple', [f'{self.__class__.__name__}_loss'])(self.loss_fn(predictions, targets))
    
    # TODO: off-policy updates with sampling characteristics.
    # def forward_off_policy(self, obs, actions, logprobs):
    #     # Cs = self.char.sampler.sample(obs.shape[0])
    #     Cs = self.char.sampler.sample(1).expand(obs.shape[0], -1)

    #     # Target values.
    #     targets = self.char.get_val(obs, Cs) - self.char.null(obs)

    #     # Forward pass
    #     forward = self.predict(obs)
    #     predictions = torch.sum(Cs.unsqueeze(-1) * forward, dim=1) # Sum over the coalition dimension.

    #     # Importance sampling weights
    #     new_logprobs = self.char.agent.choose_action(obs, action=actions, exp=self.char.explore).logprob        
    #     weights = (new_logprobs - logprobs).exp()

    #     return namedtuple('LossTuple', [f'{self.__class__.__name__}_loss'])(self.loss_fn(predictions, targets, weights))
    
    def backward(self, loss):
        """
        Function to update the model
        """

        # Backward pass and optimization
        self.optimizer.zero_grad()
        getattr(loss, f'{self.__class__.__name__}_loss').backward()
        self.optimizer.step()

        return loss

    def normalise(self, obs, sv):
        """
        Adds the normalisation step to the Shapley values.
        Note: Using get_val instead of v_F because:
            - v_F (grand coalition) should be expected return for performance, but the critic may be inaccurate (like with PPO).
            - Using the learnt characteristic model for the performance grand coalition tends to be more accurate.
            - For policy and value, they should be the same anyway.
        """

        norm_fix = (self.char.get_val(obs, torch.ones_like(obs).to(DEVICE)) - self.char.null(obs) - sv.sum(dim=-2)) / self.char.F_card

        return sv + norm_fix.unsqueeze(-2)
    
    def normalise_exact(self, obs, sv): 
        """
        Mirrors normalise but for exact Shapley values.
        """
        
        norm_fix = (self.char.get_exact_val(obs, torch.ones_like(obs).to(DEVICE)) - self.char.null_exact(obs) - sv.sum(dim=-2)) / self.char.F_card
        
        return sv + norm_fix.unsqueeze(-2)

    def sv(self, obs):
        """
        Returns the Shapley values predicted by the model.
        Predicts, then normalises.
        """
        return self.normalise(obs, self.predict(obs).detach())
    
    def get_exact(self, exact_buffer=None):
        """
        Returns the exact Shapley values using WLS.
        """

        # Compute new exact if given new buffer (e.g. if training parallel).
        if not hasattr(self, 'exact') or (exact_buffer and exact_buffer != self.exact_buffer):

            if exact_buffer:
                self.exact_buffer = exact_buffer
            
            # If using exact characteristics, ensure they are up to date.
            if self.with_exact_char:
                self.char.get_exact(exact_buffer) 

            # Generate coalition matrix (m x F_card)
            X = torch.cartesian_prod(*torch.Tensor(self.char.F_card * [[0, 1]]).to(DEVICE))

            # Calculate diagonal matrix of weights for each coalition based on its size.
            weight = lambda C: (self.char.F_card - 1) / (comb(self.char.F_card, C.sum()) * C.sum() * (self.char.F_card - C.sum()))
            W = torch.diag(torch.Tensor([weight(C) if (C.sum() != 0 and C.sum() != self.char.F_card) else 0 for C in X.int()])).to(DEVICE) # Weight matrix (m x m)

            # Sample unique observations from the exact buffer.
            e_obs = torch.unique(self.exact_buffer.sample(self.exact_buffer.size, 'obs', start=0)[0], dim=0)

            self.exact = {}
            for e_ob in tqdm(e_obs, 'Exact Shapley values') if getattr(self.args, 'verbose', False) else e_obs:

                # Compute exact Shapley values.
                if self.with_exact_char:
                    Y = torch.stack([self.char.get_exact_val(e_ob, C) for C in X]).T # Y (n x m)
                    solutions = torch.stack([torch.linalg.solve(X.T @ W @ X, X.T @ W @ y) for y in Y]).T # Solutions (F_card x n)
                    
                    # Normalise and store.
                    self.exact[*e_ob.cpu().numpy()] = self.normalise_exact(e_ob, solutions)

                # Using the approximate characteristic values instead of exact,
                # Exact given the approximate characteristic values.
                else:
                    Y = torch.stack([self.char.get_val(e_ob, C) for C in X]).T # Y (n x m)
                    solutions = torch.stack([torch.linalg.solve(X.T @ W @ X, X.T @ W @ y) for y in Y]).T # Solutions (F_card x n)
                    self.exact[*e_ob.cpu().numpy()] = self.normalise(e_ob, solutions)

            # print(f'Exact Shapley Values: {self.exact}')
    
    def exact_loss(self, exact_buffer=None):
        """
        Calculates the loss of the exact Shapley values.
        """

        # Ensure exact Shapley values are computed and up to date.
        self.get_exact(exact_buffer)

        exact_values = torch.stack(list(self.exact.values())).to(DEVICE)
        approx_values = self.sv(torch.tensor(list(self.exact)).to(DEVICE))

        return namedtuple('LossTuple', [f'{self.__class__.__name__}_exact_loss'])(self.exact_loss_fn(exact_values, approx_values))
    
    def save_explanations(self, run_name):
        """
        Saves dicts of the obs and predicted Shapley values for the buffers.
        """

        if self.train_buffer:
            train_buffer_obs = self.train_buffer.sample(self.train_buffer.size, 'obs', start=0)[0]
            svs = self.sv(train_buffer_obs).cpu().numpy()
            svs_dict = {tuple(obs): sv for obs, sv in zip(train_buffer_obs.cpu().numpy(), svs)}
            
            with open(f"runs/{run_name}/train_svs.pkl", "wb") as f:
                pickle.dump(svs_dict, f)

        if self.val_buffer:
            val_buffer_obs = self.val_buffer.sample(self.val_buffer.size, 'obs', start=0)[0]
            svs = self.sv(val_buffer_obs).cpu().numpy()
            svs_dict = {tuple(obs): sv for obs, sv in zip(val_buffer_obs.cpu().numpy(), svs)}
            
            with open(f"runs/{run_name}/val_svs.pkl", "wb") as f:
                pickle.dump(svs_dict, f)

class PolicyShapley(Shapley):
    """Shapley value model for behaviour explanations."""
    def __init__(self, env, sv_args, char=None, train_buffer=None, val_buffer=None, model=None):
        super().__init__(env, sv_args, char, train_buffer, val_buffer, model)

    def output_shape(self, env):
        return (*env.single_observation_space.shape, env.single_action_space.n)


class ValueShapley(Shapley):
    """Shapley value model for prediction explanations."""
    def __init__(self, env, sv_args, char=None, train_buffer=None, val_buffer=None, model=None):
        super().__init__(env, sv_args, char, train_buffer, val_buffer, model)

    def output_shape(self, env):
        return (*env.single_observation_space.shape, 1)


class PerformanceShapley(Shapley):
    """Shapley value model for outcome explanations."""
    def __init__(self, env, sv_args, char=None, train_buffer=None, val_buffer=None, model=None):
        super().__init__(env, sv_args, char, train_buffer, val_buffer, model)

    def output_shape(self, env):
        return (*env.single_observation_space.shape, 1)

# -------------------------------------------- XDQN and XPPO Shapley --------------------------------------------
        
class MixedShapley(Shapley):
    """
    Wrapper class for multiple Shapley value models.
    Most functions call the respective functions of each Shapley model.
    """

    def __init__(self, env, **kwargs):
        """
        kwargs = {'__class__.__name__': {sv_args:, char:, train_buffer:, val_buffer:, model:}, ...}
        Each value may or may not contain a char, train_buffer, val_buffer, and model.
        """

        # Shapley value models for each explanation
        self.shapleys = {class_name: globals()[class_name](env, **values) for class_name, values in kwargs.items()}

    def add_chars(self, chars):
        for name, char in chars.items():
            self.shapleys[name].add_char(char)

    def load_model(self, model_paths, eval=False, train_buffer=False):
        for shapley, model_path in zip(self.shapleys.values(), model_paths):
            shapley.load_model(model_path, eval, train_buffer)

    def forward(self, obs):
        """
        Forward pass to compute the mixed Shapley value losses,
        summing the losses from each Shapley model.
        """

        sv_losses = {
            name: loss 
            for shapley in self.shapleys.values() 
            for name, loss in shapley.forward(obs)._asdict().items()
            }
        
        return namedtuple('LossTuple', [
            f'MixedShapley_loss',
            *sv_losses.keys(),
            ]
        )(
            sum(sv_losses.values()),
            **sv_losses,
        )
    
    def sv(self, obs):
        return namedtuple(
            'svTuple', list(self.shapleys.keys()) 
            )(
                **{class_name: shapley.sv(obs) for class_name, shapley in self.shapleys.items()}
            )

    def exact_loss(self, exact_buffer=None):
        """
        Computing the exact losses for the characteristic and Shapley values.
        Using the exact characteristic values for computing exact Shapley values.
        """

        # Collect loss names and values
        losses = {}
        for shapley in self.shapleys.values():
            if shapley.args.exact_loss:
                losses |= shapley.exact_loss(exact_buffer)._asdict()

        # Return the named tuple
        return namedtuple('LossTuple', [*losses.keys()])(**losses)
    
# -------------------------------------------- XDQN-Shapley --------------------------------------------
class XDQN_S(DQN_Super, MixedShapley):
    """XDQN-Shapley model combining DQN with Shapley value models."""
    def __init__(self, env, agent_args, **kwargs):
        """
        kwargs = {'__class__.__name__': sv_args, ...}
        We assume the first head is the DQN head, that is what is copied to the target critic.
        Assume at least one Shapley head is given in kwargs.
        """

        # Set input dimensions for the shared layers 
        if not getattr(agent_args, 'cnn', False):
            agent_args.shared_arch['input'][1][0] = np.prod(env.single_observation_space.shape)

        # Set model architectures and output shapes for the heads.
        heads = {'DQN': agent_args.critic_arch}
        heads_output_shapes = [(env.single_action_space.n,)]

        for class_name, sv_args in kwargs.items():
            heads[class_name] = sv_args.model_arch
            heads_output_shapes.append(globals()[class_name].output_shape(globals()[class_name], env))

        # Initialize the model and target critic
        self.model = MultiHeadNN(agent_args.shared_arch, heads, heads_output_shapes).to(DEVICE)
        self.target_critic = SimpleNN('DQN', **agent_args.shared_arch, **agent_args.critic_arch, output_shape=(env.single_action_space.n,)).to(DEVICE)

        # Set each explanation model to the agent model.
        for class_name, sv_args in kwargs.items():
            kwargs[class_name] = {'sv_args': sv_args, 'model': self.model}

        # Initialise the super classes
        DQN_Super.__init__(self, env, agent_args)
        MixedShapley.__init__(self, env, **kwargs)

    def forward(self, obs, actions, rewards, n_obs, terminations):
        """
        Forward pass to compute the XDQN-Shapley losses,
        summing the DQN loss and the losses from each Shapley model.
        """

        dqn_losses = DQN_Super.forward(self, obs, actions, rewards, n_obs, terminations)
        shapley_losses = MixedShapley.forward(self, obs)

        return namedtuple('LossTuple', [
            'XDQN_S_loss',
            *dqn_losses._fields,
            *shapley_losses._fields,
            ]
        )(
            dqn_losses.DQN_loss + shapley_losses.MixedShapley_loss,
            **dqn_losses._asdict(),
            **shapley_losses._asdict(),
        )
    

# -------------------------------------------- XPPO-Shapley --------------------------------------------
class XPPO_S(PPO_Super, MixedShapley):
    """XPPO-Shapley model combining PPO with Shapley value models."""
    def __init__(self, env, agent_args, **kwargs):
        """
        kwargs = {'__class__.__name__': sv_args, ...}
        Assume at least one Shapley head is given in kwargs.
        """

        # Set input dimensions for the critic and the shared layers of the actor
        agent_args.critic_arch['input'][1][0] = np.prod(env.single_observation_space.shape)
        agent_args.shared_arch['input'][1][0] = np.prod(env.single_observation_space.shape)

        # Set model architectures and output shapes for the heads.
        heads = {'PPO': agent_args.actor_arch}
        heads_output_shapes = [(env.single_action_space.n,)]

        for class_name, sv_args in kwargs.items():
            heads[class_name] = sv_args.model_arch
            heads_output_shapes.append(globals()[class_name].output_shape(globals()[class_name], env))

        # Initialize the shared actor model and the critic
        self.model = MultiHeadNN(agent_args.shared_arch, heads, heads_output_shapes).to(DEVICE)
        self.critic = SimpleNN('PPO', **agent_args.critic_arch, output_shape=(1,)).to(DEVICE)

        # Set each explanation model to the agent model.
        for class_name, sv_args in kwargs.items():
            kwargs[class_name] = {'sv_args': sv_args, 'model': self.model}

        # Initialise the super classes
        PPO_Super.__init__(self, env, agent_args)
        MixedShapley.__init__(self, env, **kwargs)

    def forward(self, mb_obs, mb_actions, mb_logprobs, mb_values, mb_advantages, mb_returns):
        """
        Forward pass to compute the XPPO-Shapley losses,
        summing the PPO loss and the losses from each Shapley model.
        """

        ppo_losses = PPO_Super.forward(
            self, 
            mb_obs, 
            mb_actions,
            mb_logprobs, 
            mb_values, 
            mb_advantages, 
            mb_returns
            )
        shapley_losses = MixedShapley.forward(self, mb_obs)

        return namedtuple('LossTuple', [
            'XPPO_S_loss',
            *ppo_losses._fields,
            *shapley_losses._fields,
            ]
        )(
            ppo_losses.PPO_loss + shapley_losses.MixedShapley_loss,
            **ppo_losses._asdict(),
            **shapley_losses._asdict(),
        )