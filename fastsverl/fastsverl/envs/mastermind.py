import copy
import numpy as np
from collections import defaultdict
import random
import gymnasium
from gymnasium import spaces
from tqdm import tqdm

# To do:
    # Can change step to use P. Need to still have a version of step to make P though.

class Mastermind(gymnasium.Env):
    """
    Environment for the game mastermind.
    """

    def __init__(self, code_size:int=2, num_guesses:int=3, num_pegs:int=2, num_goals:int=None, seed:int=None):

        super().__init__()

        self.code_size = code_size
        self.num_guesses = num_guesses
        if num_goals == None: self.num_goals = num_pegs ** code_size
        else: self.num_goals = num_goals

        # Define the discrete state space. START IS NOT STRICTLY CORRECT.
        self.observation_space = spaces.MultiDiscrete(np.full(self.num_guesses * (self.code_size + 2), num_pegs + 1))
        # self.observation_space = spaces.MultiDiscrete(np.full(self.num_guesses * (self.code_size + 2), num_pegs + 1), start=np.full(self.num_guesses * (self.code_size + 2), -1))

        # Define the discrete action space
        self.action_space = spaces.Discrete(num_pegs ** code_size)

        # Set seed
        if seed is not None: self.seed(seed)

        # Function mapping a peg index to a letter 
        self.peg_to_letter = np.vectorize(lambda peg: chr(peg + 64) if peg > 0 else ' ')

        # All possible guesses. Each guess is a combination of pegs, with the smallest peg being 1 and the largest num_pegs.
        self.index_to_guess = np.array([[(peg // num_pegs ** size) % num_pegs for size in range(code_size)] for peg in range(self.action_space.n)]) + 1

        # Chosen goals from all possible goals
        self.goals = self.index_to_guess[np.random.choice(self.action_space.n, self.action_space.n, replace=False)][:num_goals]
        self.goals_letters = self.peg_to_letter(self.goals)
        
        # Saving the clues for each goal and guess pair, for speed.
        self.clues_dict = CluesDict()

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
        Returns transition dynamics.
        """

        P = defaultdict(lambda: defaultdict(list))

        self.encode_dict, self.decode_dict, self.encoding_counter = {}, {}, 0
        def encode(s):
            if s not in self.encode_dict:
                self.encode_dict[s], self.decode_dict[self.encoding_counter] = self.encoding_counter, np.array(s)
                self.encoding_counter += 1
            return self.encode_dict[s]
        
        self.encode = encode
        self.decode = lambda s: self.decode_dict[s]

        def inner(env, obs, action, P): 

            n_obs, reward, terminated, truncated, _ = env.step(action)

            P[self.encode(tuple(obs))][action].append([1.0, self.encode(tuple(n_obs)), reward, terminated, truncated])

            if terminated: return None
            else:
                for action in range(env.action_space.n):
                    inner(copy.deepcopy(env), n_obs, action, P)

        for goal in tqdm(self.goals, 'Creating P') if False else self.goals:
            obs, _ = self.reset()
            self.goal = goal
            for action in tqdm(range(self.action_space.n) , 'Creating P for goal') if False else range(self.action_space.n):
                inner(copy.deepcopy(self), obs, action, P)

        # Normalise transitions
        P_stoch = {s: [] for s in P}

        for s, values in P.items():
            for _, transitions in values.items():
                trans, counts = np.unique(np.array(transitions)[:, 1:], axis=0, return_counts=True)
                new_transitions = np.hstack((counts[:, None] / counts.sum(), trans)).astype(object)
                new_transitions[:, 1] = new_transitions[:, 1].astype(int)
                new_transitions[:, 3:] = new_transitions[:, 3:].astype(bool)
                P_stoch[s].append(new_transitions)

        self.P = P_stoch

    def reset(self, state=None, seed:int=None, options:dict=None):
        """
        Resets environment to empty grid with new goal.
        """

        if seed is not None: self.seed(seed)

        self.grid = np.full((self.num_guesses, self.code_size + 2), -1, dtype=int)
        self.goal = self.goals[np.random.randint(self.num_goals)]
        self.count = 0 # To track where guess is placed.

        return self.grid.flatten(), {}
    
    def step(self, action):
        """
        Step in the environment, places piece in next available square.
        """

        # Negative reward for each step
        reward = -1
        done = False

        # The number of pegs exactly right and the number in the wrong position
        guess = self.index_to_guess[action]
        num_exact, num_close = self.clues_dict[guess.tobytes(), self.goal.tobytes()]

        self.grid[self.count, 1:-1] = guess
        self.grid[self.count, 0] = num_close
        self.grid[self.count, -1] = num_exact

        if self.count == self.num_guesses - 1: # Finished game with no win
            done = True

        if num_exact == self.code_size: # Won game
            done = True
            reward += self.num_guesses # WHEN RUN AGAIN THIS LINE DOESNT ACTUALLY NEED TO EXIST.

        self.count += 1

        return self.grid.flatten(), reward, done, False, {}
    
    def get_clues(self, guess, goal):
        """
        Based on a guess and the current goal returns:
            - The number of pegs in the exact right position.
            - The number of pegs that are right but in the wrong position.
        """
           
        num_exact = (guess == goal).sum()

        num_close = - num_exact
        for a in guess:
            if a in goal:
                num_close += 1
                goal[(goal == a).argmax()] = 0
        
        return num_exact, num_close
    
    def render(self):
        """
        Renders environment, expensive!
        """

        grid_render = self.grid.copy().astype(object)
        grid_render[:, 1:-1] = self.peg_to_letter(grid_render[:, 1:-1])
        print(np.flipud(grid_render), '\n')

class CluesDict(dict, Mastermind):
    
    def __missing__(self, key):

        guess_bytes, goal_bytes = key
        val = self.get_clues(np.frombuffer(guess_bytes, dtype=np.int_),
                             np.frombuffer(goal_bytes, dtype=np.int_).copy())
        
        self.__setitem__(key, val)
        
        return val