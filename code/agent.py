import numpy as np
from code.config import GRID_SIZE, DISCOUNT_FACTOR, CONVERGENCE_THRESHOLD

class ValueIterationSolver:
    def __init__(self, env):
        self.env = env
        self.gamma = DISCOUNT_FACTOR
        self.theta = CONVERGENCE_THRESHOLD
        self.v_table = np.zeros((GRID_SIZE, GRID_SIZE))
        self.policy = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    def compute_v_iteration(self):
        """
        Bellman Optimality Equation: 
        V(s) = max_a sum_{s'} P(s'|s,a) [R(s,a,s') + gamma * V(s')]
        """
        iterations = 0
        while True:
            delta = 0
            new_v_table = np.copy(self.v_table)
            
            for state in self.env.get_all_states():
                r, c = state
                if state == self.env.goal_pos:
                    continue
                    
                q_values = []
                for action in self.env.get_valid_actions(state):
                    q_a = 0
                    transitions = self.env.get_transitions(state, action)
                    for prob, next_state, reward, is_terminal in transitions:
                        nr, nc = next_state
                        q_a += prob * (reward + self.gamma * self.v_table[nr, nc])
                    q_values.append(q_a)
                
                new_v_table[r, c] = max(q_values)
                delta = max(delta, abs(new_v_table[r, c] - self.v_table[r, c]))
            
            self.v_table = new_v_table
            iterations += 1
            
            if delta < self.theta:
                break
        
        return iterations

    def extract_policy(self):
        """
        Derive optimal policy from the converged value function.
        """
        for state in self.env.get_all_states():
            r, c = state
            if state == self.env.goal_pos:
                continue
                
            q_values = []
            for action in self.env.get_valid_actions(state):
                q_a = 0
                transitions = self.env.get_transitions(state, action)
                for prob, next_state, reward, is_terminal in transitions:
                    nr, nc = next_state
                    q_a += prob * (reward + self.gamma * self.v_table[nr, nc])
                q_values.append(q_a)
            
            self.policy[r, c] = np.argmax(q_values)
            
        return self.policy
