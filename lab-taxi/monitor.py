import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, alpha=0.01, gamma=1.0, epsilon=1.0):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
    
    def calc_probs(self, state):
        best_action_idx = np.argmax(self.Q.get(state))
        probs = np.full(self.nA, self.epsilon / self.nA)
        probs[best_action_idx] += (1.0 - self.epsilon)
        return probs
    
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        probs = self.calc_probs(state)
        return np.random.choice(self.nA, p=probs)
    
    def expected_sarsa(self, state):
        probs = self.calc_probs(state)
        expected_reward = np.sum(self.Q[state] * probs)
        return expected_reward
        
        
    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        expected_reward = 0 if done else self.expected_sarsa(next_state)
        self.Q[state][action] *= (1 - self.alpha)
        self.Q[state][action] += self.alpha * (reward + self.gamma * expected_reward)