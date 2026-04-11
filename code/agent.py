import numpy as np
import random
from code.config import (
    GRID_SIZE, GAMMA, ALPHA_START, ALPHA_MIN, ALPHA_DECAY,
    EPSILON_START, EPSILON_MIN, EPSILON_DECAY, DISCOUNT_FACTOR, CONVERGENCE_THRESHOLD
)

class QLearningAgent:
    def __init__(self, n_actions=4):
        self.n_actions = n_actions
        self.alpha = ALPHA_START
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, n_actions))

    def choose_action(self, state, epsilon=None):
        eps = epsilon if epsilon is not None else self.epsilon
        r, c = state
        if random.random() < eps:
            return random.randint(0, self.n_actions - 1)
        else:
            # FIX: Randomly break ties to prevent getting stuck in loops/corners
            q_values = self.q_table[r, c]
            max_q = np.max(q_values)
            # Find all indices that have the maximum value
            actions_with_max_q = np.where(q_values == max_q)[0]
            return random.choice(actions_with_max_q)

    def update_q_value(self, state, action, reward, next_state):
        r, c = state
        nr, nc = next_state
        max_future_q = np.max(self.q_table[nr, nc])
        td_target = reward + self.gamma * max_future_q
        self.q_table[r, c, action] += self.alpha * (td_target - self.q_table[r, c, action])

    def decay_hyperparameters(self):
        self.alpha = max(ALPHA_MIN, self.alpha * ALPHA_DECAY)
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

class ValueIterationSolver:
    def __init__(self, env):
        self.env = env
        self.gamma = DISCOUNT_FACTOR # 0.99 from L47
        self.theta = CONVERGENCE_THRESHOLD
        self.v_table = np.zeros((GRID_SIZE, GRID_SIZE))
        self.policy = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    def compute_v_iteration(self):
        while True:
            delta = 0
            new_v_table = np.copy(self.v_table)
            for state in self.env.get_all_states():
                if state == self.env.goal_pos: continue
                r, c = state
                q_values = []
                for action in range(4):
                    q_a = 0
                    for prob, next_state, reward, is_terminal in self.env.get_transitions(state, action):
                        nr, nc = next_state
                        q_a += prob * (reward + self.gamma * self.v_table[nr, nc])
                    q_values.append(q_a)
                new_v_table[r, c] = max(q_values)
                delta = max(delta, abs(new_v_table[r, c] - self.v_table[r, c]))
            self.v_table = new_v_table
            if delta < self.theta: break

    def extract_policy(self):
        for state in self.env.get_all_states():
            if state == self.env.goal_pos: continue
            r, c = state
            q_values = []
            for action in range(4):
                q_a = 0
                for prob, next_state, reward, is_terminal in self.env.get_transitions(state, action):
                    nr, nc = next_state
                    q_a += prob * (reward + self.gamma * self.v_table[nr, nc])
                q_values.append(q_a)
            self.policy[r, c] = np.argmax(q_values)
        return self.policy
