# -*- coding:utf-8 -*-
import math, os, time, sys
import numpy as np
import gym
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.
from collections import defaultdict
from typing import Optional

def _argmax_with_random_tie(q_values: np.ndarray, actions: np.ndarray) -> int:
    """Return an argmax action, breaking ties uniformly at random.

    q_values: shape (n_actions,)
    actions: array of action ids corresponding to q_values indices
    """
    max_v = np.max(q_values)
    candidates = actions[np.where(q_values == max_v)[0]]
    return np.random.choice(candidates)

##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #

class SarsaAgent(object):
    ##### START CODING HERE #####
    def __init__(
        self,
        all_actions: np.ndarray,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        """Initialize the SARSA agent.

        Parameters
        - all_actions: array/list of available actions (e.g., np.arange(n_actions))
        - alpha: learning rate
        - gamma: reward discount factor
        - epsilon: initial epsilon for epsilon-greedy
        - epsilon_min: minimum epsilon
        - epsilon_decay: multiplicative decay applied after each action selection
        """
        self.all_actions = np.asarray(all_actions)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table as a defaultdict to avoid predefining state size.
        # Each unseen state maps to a zero vector of size |A|.
        self.Q = defaultdict(lambda: np.zeros(len(self.all_actions), dtype=np.float32))

    def choose_action(self, observation):
        """Choose an action with epsilon-greedy policy and apply epsilon decay.

        observation: a discrete state id or any hashable key.
        """
        # Ensure state exists in Q
        q_values = self.Q[observation]

        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.all_actions)
        else:
            action = _argmax_with_random_tie(q_values, self.all_actions)
        return action
    
    def learn(self, s, a, r, s_next, a_next=None, done: bool = False):
        """Learn from one SARSA transition.

        Parameters
        - s: current state
        - a: action taken at state s
        - r: reward received
        - s_next: next state observed
        - a_next: next action chosen by current policy (None if terminal)
        - done: whether s_next is terminal

        Returns: TD error (float) for logging/debugging
        """
        q_sa = self.Q[s][a]
        if done or a_next is None:
            target = r
        else:
            target = r + self.gamma * self.Q[s_next][a_next]
        td_error = target - q_sa
        self.Q[s][a] = q_sa + self.alpha * td_error
        return td_error

    ##### END CODING HERE #####


class QLearningAgent(object):
    ##### START CODING HERE #####
    def __init__(
        self,
        all_actions: np.ndarray,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        """Initialize the Q-Learning agent (off-policy).
        
        Parameters
        - all_actions: array/list of available actions (e.g., np.arange(n_actions))
        - alpha: learning rate
        - gamma: reward discount factor
        - epsilon: initial epsilon for epsilon-greedy
        - epsilon_min: minimum epsilon
        - epsilon_decay: multiplicative decay applied after each action selection
        """
        self.all_actions = np.asarray(all_actions)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = defaultdict(lambda: np.zeros(len(self.all_actions), dtype=np.float32))

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm and decay epsilon.
        
        observation: a discrete state id or any hashable key.
        """
        q_values = self.Q[observation]
        if np.random.rand() < self.epsilon:
            action = int(np.random.choice(self.all_actions))
        else:
            action = int(_argmax_with_random_tie(q_values, self.all_actions))
        return action
    
    def learn(self, s, a, r, s_next, done: bool = False):
        """Q-learning update using max_a' Q(s',a')."""
        q_sa = self.Q[s][a]
        if done:
            target = r
        else:
            target = r + self.gamma * np.max(self.Q[s_next])
        td_error = target - q_sa
        self.Q[s][a] = q_sa + self.alpha * td_error
        return td_error

    ##### END CODING HERE #####
    
    

class Dyna_QAgent(object):
    ##### START CODING HERE #####
    def __init__(
        self,
        all_actions: np.ndarray,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        n_planning: int = 10,
    ):
        """Initialize the Dyna-Q agent (model-based + planning)."""
        self.all_actions = np.asarray(all_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_planning = n_planning

        self.Q = defaultdict(lambda: np.zeros(len(self.all_actions), dtype=np.float32))
        # Simple deterministic model: (s,a)->(s',r)
        self.model_next = {}
        self.model_reward = {}

    def choose_action(self, observation):
        q_values = self.Q[observation]
        if np.random.rand() < self.epsilon:
            action = int(np.random.choice(self.all_actions))
        else:
            action = int(_argmax_with_random_tie(q_values, self.all_actions))
        return action
    
    def _q_learning_update(self, s, a, r, s_next, done: bool):
        q_sa = self.Q[s][a]
        target = r if done else r + self.gamma * np.max(self.Q[s_next])
        td_error = target - q_sa
        self.Q[s][a] = q_sa + self.alpha * td_error
        return float(td_error)

    def learn(self, s, a, r, s_next, done: bool = False):
        """Real experience update + planning updates from learned model."""
        # 1) Real update
        td = self._q_learning_update(s, a, r, s_next, done)

        # 2) Model learning
        self.model_next[(s, a)] = s_next
        self.model_reward[(s, a)] = r

        # 3) Planning updates
        if self.model_next:
            keys = list(self.model_next.keys())
            for _ in range(self.n_planning):
                ss, aa = keys[np.random.randint(len(keys))]
                ss_next = self.model_next[(ss, aa)]
                rr = self.model_reward[(ss, aa)]
                # Note: we don't know if the sampled (ss,aa) leads to terminal; assume non-terminal unless it's goal/cliff detected during real runs.
                self._q_learning_update(ss, aa, rr, ss_next, False)
        return td

    ##### END CODING HERE #####
