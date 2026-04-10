import numpy as np
from code.config import (
    GRID_SIZE, START_POS, GOAL_POS, OBSTACLES, TRAPS, WIND_ZONES,
    GOAL_REWARD, TRAP_PENALTY, STEP_PENALTY, WIND_ZONE_PENALTY
)

class GridWorld:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.start_pos = START_POS
        self.goal_pos = GOAL_POS
        self.obstacles = set(OBSTACLES)
        self.traps = set(TRAPS)
        self.wind_zones = set(WIND_ZONES)
        self.agent_pos = START_POS

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def get_transitions(self, state, action):
        """
        Returns a list of tuples: (probability, next_state, reward, is_terminal)
        """
        if state == self.goal_pos:
            return [(1.0, state, 0, True)]

        r, c = state
        nr, nc = r, c

        if action == 0: nr = max(0, r - 1)
        elif action == 1: nr = min(self.grid_size - 1, r + 1)
        elif action == 2: nc = max(0, c - 1)
        elif action == 3: nc = min(self.grid_size - 1, c + 1)

        if (nr, nc) in self.obstacles:
            nr, nc = r, c

        next_state = (nr, nc)
        
        # Reward Calculation based on PRD
        reward = STEP_PENALTY # Empty cell: -1
        is_terminal = False

        if next_state == self.goal_pos:
            reward = GOAL_REWARD # Goal: +10
            is_terminal = True
        elif next_state in self.traps:
            reward = TRAP_PENALTY # Trap: -5
        elif next_state in self.wind_zones:
            reward = WIND_ZONE_PENALTY # Wind Zone: -2

        return [(1.0, next_state, reward, is_terminal)]

    def get_all_states(self):
        states = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) not in self.obstacles:
                    states.append((r, c))
        return states

    def toggle_obstacle(self, pos):
        """Adds or removes an obstacle at the given position."""
        if pos == self.start_pos or pos == self.goal_pos:
            return # Don't block start or goal
        
        if pos in self.obstacles:
            self.obstacles.remove(pos)
        else:
            self.obstacles.add(pos)

    def get_valid_actions(self, state):
        return [0, 1, 2, 3]
