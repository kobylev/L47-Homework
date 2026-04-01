import numpy as np
import random
from code.config import (
    GRID_SIZE, START_POS, GOAL_POS, OBSTACLES, TRAPS,
    GOAL_REWARD, TRAP_PENALTY, STEP_PENALTY,
    N_DYNAMIC_OBSTACLES, DYNAMIC_OBSTACLE_MOVE_CHANCE, DYNAMIC_OBSTACLE_PENALTY
)

class GridWorld:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.start_pos = START_POS
        self.goal_pos = GOAL_POS
        self.obstacles = set(OBSTACLES)
        self.traps = set(TRAPS)
        self.agent_pos = START_POS
        self.dynamic_obstacles = []

    def reset(self):
        self.agent_pos = self.start_pos
        self._init_dynamic_obstacles()
        return self.agent_pos

    def _init_dynamic_obstacles(self):
        self.dynamic_obstacles = []
        forbidden = self.obstacles.union(self.traps).union({self.start_pos, self.goal_pos})
        
        while len(self.dynamic_obstacles) < N_DYNAMIC_OBSTACLES:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if pos not in forbidden and pos not in self.dynamic_obstacles:
                self.dynamic_obstacles.append(pos)

    def _move_dynamic_obstacles(self):
        new_positions = []
        forbidden = self.obstacles.union({self.goal_pos}) # Don't block the goal or move into walls
        
        for r, c in self.dynamic_obstacles:
            if random.random() < DYNAMIC_OBSTACLE_MOVE_CHANCE:
                # Try moving in a random direction
                action = random.randint(0, 3)
                nr, nc = r, c
                if action == 0: nr = max(0, r - 1)
                elif action == 1: nr = min(self.grid_size - 1, r + 1)
                elif action == 2: nc = max(0, c - 1)
                elif action == 3: nc = min(self.grid_size - 1, c + 1)
                
                if (nr, nc) not in forbidden:
                    new_positions.append((nr, nc))
                else:
                    new_positions.append((r, c))
            else:
                new_positions.append((r, c))
        self.dynamic_obstacles = new_positions

    def step(self, action):
        """
        Actions:
        0: Up
        1: Down
        2: Left
        3: Right
        """
        # 1. Environment moves
        self._move_dynamic_obstacles()

        # 2. Agent moves
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

        # Check for static obstacles
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
        elif self.agent_pos in self.dynamic_obstacles:
            reward = DYNAMIC_OBSTACLE_PENALTY
            # Optionally don't end episode, but penalize collision
            # self.agent_pos = (r, c) # Could also bounce back

        return self.agent_pos, reward, done

    def get_valid_actions(self, pos):
        # All actions are valid; hitting walls just keeps agent in place
        return [0, 1, 2, 3]
