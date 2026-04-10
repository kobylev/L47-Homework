import numpy as np
from code.config import (
    GRID_SIZE, START_POS, GOAL_POS, OBSTACLES, TRAPS,
    GOAL_REWARD, TRAP_PENALTY, STEP_PENALTY
)

class GridWorld:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.start_pos = START_POS
        self.goal_pos = GOAL_POS
        self.obstacles = set(OBSTACLES)
        self.traps = set(TRAPS)
        self.agent_pos = START_POS

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def get_transitions(self, state, action):
        """
        Returns a list of tuples: (probability, next_state, reward, is_terminal)
        Actions: 0: Up, 1: Down, 2: Left, 3: Right
        """
        if state == self.goal_pos:
            return [(1.0, state, 0, True)] # Already at goal

        r, c = state
        nr, nc = r, c

        if action == 0: # Up
            nr = max(0, r - 1)
        elif action == 1: # Down
            nr = min(self.grid_size - 1, r + 1)
        elif action == 2: # Left
            nc = max(0, c - 1)
        elif action == 3: # Right
            nc = min(self.grid_size - 1, c + 1)

        # Check for static obstacles
        if (nr, nc) in self.obstacles:
            nr, nc = r, c # Hit a wall, stay in place

        next_state = (nr, nc)
        
        # Calculate reward
        reward = STEP_PENALTY
        is_terminal = False

        if next_state == self.goal_pos:
            reward = GOAL_REWARD
            is_terminal = True
        elif next_state in self.traps:
            reward = TRAP_PENALTY
            # PRD doesn't say traps are terminal, so they just penalize.

        return [(1.0, next_state, reward, is_terminal)]

    def get_all_states(self):
        states = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) not in self.obstacles:
                    states.append((r, c))
        return states

    def get_valid_actions(self, state):
        return [0, 1, 2, 3]
