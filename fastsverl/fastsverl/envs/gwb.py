import copy
import random
import numpy as np
from collections import defaultdict
import gymnasium
from gymnasium import spaces

class GWB(gymnasium.Env):
    """
    2 x 4 grid world where top two squares are a goal.
    Grid has a boarder around the sides.
    (0, 1) is an impassible block.
    Reward is -1 each step and +10 for reaching the goal.
    Initial position is randomly sampled from bottom two squares.
    """

    def __init__(self, seed:int=None):

        super().__init__()

        # 4 height and 2 width
        H = 4; W = 2
        self.grid = np.zeros((H, W))

        # Define the discrete state space
        self.observation_space = spaces.MultiDiscrete(np.array([W, H]))

        # Define the discrete action space
        self.action_space = spaces.Discrete(4)

        # Set seed
        if seed is not None: 
            self.seed(seed)

        # Initial states
        self.start = np.array([[0, 0], [1, 0]])

        self.action_dict = {0: [0, 1],
                            1: [1, 0],
                            2: [0, -1],
                            3: [-1, 0]}
        
        self.states_decode = {ind: state for ind, state in enumerate(np.array([(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)]))}
        self.decode = lambda state_ind: self.states_decode[state_ind]
        self.states_encode = {tuple(state): ind for ind, state in self.states_decode.items()}
        self.encode = lambda state: self.states_encode[tuple(state)]
        
        self.P = defaultdict(lambda: defaultdict(list))

        # Transition for every state
        for state_ind, state in self.states_decode.items():
            for action_ind, action in self.action_dict.items():

                # Converts action int to vector, 'Take step'
                n_state = np.array(state) + action

                # Check if "hit wall"
                if not ((n_state[1] in np.arange(H)) and (n_state[0] in np.arange(W))) or (n_state == [0, 1]).all(): n_state = np.array(state)
                
                if n_state[1] == H-1: # Reached goal
                    reward, done = 9, True 
                    n_state = [0, 0] # Hack to never consider terminal state.
                else: reward, done = -1, False

                # Update transition dictionary
                self.P[state_ind][action_ind].append([1, self.encode(n_state), reward, done])

    def seed(self, seed:int):
        """
        Sets seed for environment.
        """

        np.random.seed(seed)
        random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def reset(self, start_state=None, seed:int=None, options:dict=None):
        """
        Randomly places the agent in one of the bottom two squares.
        Or sets environment to given state.
        """

        if seed is not None: 
            self.seed(seed)

        if start_state is None: self.pos = self.start[np.random.choice(2)]
        else: self.pos = np.array(start_state)

        return self.pos.copy(), {}

    def step(self, action):
        """
        Takes a step in the fully observed environment
        """

        pos_encoded, reward, done = self.P[self.encode(self.pos)][action][0][1:]
        self.pos = np.array(self.decode(pos_encoded))

        return self.pos.copy(), reward, done, False, {}