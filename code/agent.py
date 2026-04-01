import numpy as np
import random
from code.config import GRID_SIZE, LEARNING_RATE, DISCOUNT_FACTOR

class QLearningAgent:
    def __init__(self, n_actions=4):
        self.n_actions = n_actions
        self.lr = LEARNING_RATE
        self.gamma = DISCOUNT_FACTOR
        # Q-table: (row, col, action)
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, n_actions))

    def choose_action(self, state, epsilon):
        """Epsilon-greedy strategy"""
        r, c = state
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            # Exploitation: pick action with max Q-value for current state
            return np.argmax(self.q_table[r, c])

    def update(self, state, action, reward, next_state):
        """Bellman equation implementation"""
        r, c = state
        nr, nc = next_state
        
        # Max Q for the next state
        best_next_action = np.argmax(self.q_table[nr, nc])
        td_target = reward + self.gamma * self.q_table[nr, nc, best_next_action]
        td_error = td_target - self.q_table[r, c, action]
        
        # Update Q-value
        self.q_table[r, c, action] += self.lr * td_error
