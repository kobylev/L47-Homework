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

    def step(self, action):
        """
        Actions:
        0: Up
        1: Down
        2: Left
        3: Right
        """
        r, c = self.agent_pos
        new_r, new_c = r, c

        if action == 0: # Up
            new_r = max(0, r - 1)
        elif action == 1: # Down
            new_r = min(self.grid_size - 1, r + 1)
        elif action == 2: # Left
            new_c = max(0, c - 1)
        elif action == 3: # Right
            new_c = min(self.grid_size - 1, c + 1)

        # Check for obstacles
        if (new_r, new_c) in self.obstacles:
            new_r, new_c = r, c # Hit a wall, stay in place

        self.agent_pos = (new_r, new_c)
        
        # Calculate reward
        done = False
        reward = STEP_PENALTY

        if self.agent_pos == self.goal_pos:
            reward = GOAL_REWARD
            done = True
        elif self.agent_pos in self.traps:
            reward = TRAP_PENALTY
            # Optionally reset or end episode, but here we just penalize
            # done = True 

        return self.agent_pos, reward, done

    def get_valid_actions(self, pos):
        # All actions are valid; hitting walls just keeps agent in place
        return [0, 1, 2, 3]
