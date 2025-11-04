import random
import numpy as np
from collections import defaultdict
import gymnasium

"""
n^d Gridworld (hypercube) environment.
    - Start state is [0, 0, ..., 0]
    - Goal state is [n-1, n-1, ..., n-1]
    - p * n^d impassible blocks uniformly distributed in the grid.
    - Reward is -1 each step and n * d for reaching the goal.
"""

class Hypercube(gymnasium.Env):
    def __init__(self, n: int, d: int, p: float = 0, seed: int = None):
        super().__init__()

        # Define spaces
        self.observation_space = gymnasium.spaces.MultiDiscrete(np.array([n] * d))
        self.action_space = gymnasium.spaces.Discrete(2 * d)

        # Set seed
        if seed is not None: 
            self.seed(seed)

        # Domain characteristics
        self.start_state = np.zeros(d, dtype=int)
        self.goal_state = np.full(d, n-1, dtype=int)
        self.term_reward = n * d - 1
        
        # Valid states do not include impassible blocks and goal state
        self.states = np.array([
            s for i, s in enumerate(np.array(np.meshgrid(*[np.arange(n)] * d)).T.reshape(-1, d))
            if ((s == 0).all() or not int(p * n ** d) or (i - 1) % (1 // p)) and not (s == n-1).all()
        ])

        # Define actions
        self.actions = {2*i: np.eye(d, dtype=int)[i] for i in range(d)} # +1 along axis
        self.actions.update({2*i+1: -np.eye(d, dtype=int)[i] for i in range(d)}) # -1 along axis
        
        # Define transition dynamics
        self.get_P()

    def seed(self, seed:int):
        """
        Sets seed for environment.
        """

        np.random.seed(seed)
        random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def get_P(self):
        """
        Defines the transition dynamics.
        """

        self.P = defaultdict(lambda: defaultdict(list))

        for state in self.states:
            for action_ind, action in self.actions.items():

                # Converts action int to vector, 'Take step'
                n_state = np.array(state) + action

                # Terminate if goal is reached
                if (n_state == self.goal_state).all():
                    self.P[*state][action_ind].append([1, n_state, self.term_reward, True, False])
                
                # If out of bounds, stay in the same state
                elif not (n_state == self.states).all(axis=1).any():
                    self.P[*state][action_ind].append([1, state, -1, False, False])
                
                # If in bounds, move to the new state
                else:
                    self.P[*state][action_ind].append([1, n_state, -1, False, False])

    def reset(self, seed:int=None, options:dict=None):
        """
        Places the agent in the start state.
        """

        if seed is not None: 
            self.seed(seed)

        self.pos = self.start_state.copy()

        return self.pos.copy(), {}

    def step(self, action):
        """
        Takes a step in the environment
        """

        # Deterministic env so no sampling needed (just take the first).
        self.pos, reward, terminated, truncated = self.P[*self.pos][action][0][1:]

        return self.pos.copy(), reward, terminated, truncated, {}