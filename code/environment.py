import numpy as np
import random
from code.config import (
    GRID_SIZE, START_POS, GOAL_POS, STEP_PENALTY, GOAL_REWARD,
    CELL_TYPES, WALL_PROBABILITY
)

class GridWorld:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.start_pos = START_POS
        self.goal_pos = GOAL_POS
        
        self.static_walls = set()
        self.cell_types = {} # (r, c) -> type_name
        
        self.q_pos = START_POS
        self.vi_pos = START_POS
        self.agent_pos = START_POS
        
        # Compatibility
        self.dynamic_obstacles = set()
        self.dynamic_bonuses = set()
        self.static_traps = set()
        self.static_wind_zones = set()

    def randomize_layout(self):
        """Generates a new probabilistic grid configuration every step."""
        self.static_walls = set()
        self.cell_types = {}
        
        # Define types and their relative probabilities
        types = list(CELL_TYPES.keys())
        probs = [CELL_TYPES[t][1] for t in types]
        
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                pos = (r, c)
                if pos == self.start_pos or pos == self.goal_pos:
                    self.cell_types[pos] = "EMPTY"
                    continue
                
                # Check for active agents (SAFE ZONE)
                if pos in {self.q_pos, self.vi_pos, self.agent_pos}:
                    self.cell_types[pos] = "EMPTY"
                    continue
                
                # Roll for WALL first
                if random.random() < WALL_PROBABILITY:
                    self.static_walls.add(pos)
                    self.cell_types[pos] = "WALL"
                else:
                    # Roll for probabilistic cell type
                    self.cell_types[pos] = random.choices(types, weights=probs)[0]

    def reset_competition(self):
        self.q_pos = self.start_pos
        self.vi_pos = self.start_pos
        self.randomize_layout() 
        return self.q_pos, self.vi_pos

    def reset(self):
        self.agent_pos = self.start_pos
        self.randomize_layout()
        return self.agent_pos

    def _move_agent(self, current_pos, action):
        if action == -1: return current_pos
        r, c = current_pos
        nr, nc = r, c
        if action == 0: nr = max(0, r - 1)
        elif action == 1: nr = min(self.grid_size - 1, r + 1)
        elif action == 2: nc = max(0, c - 1)
        elif action == 3: nc = min(self.grid_size - 1, c + 1)
        
        if (nr, nc) in self.static_walls: return current_pos
        return (nr, nc)

    def step_dual(self, action_q, action_vi):
        self.q_pos = self._move_agent(self.q_pos, action_q)
        self.vi_pos = self._move_agent(self.vi_pos, action_vi)
        self.randomize_layout() # Board re-rolls AFTER moves
        return self._get_result(self.q_pos), self._get_result(self.vi_pos)

    def _get_result(self, pos):
        done = False
        if pos == self.goal_pos:
            return pos, GOAL_REWARD, True
        
        type_name = self.cell_types.get(pos, "EMPTY")
        cell_reward = CELL_TYPES.get(type_name, (0, 0))[0]
        
        reward = STEP_PENALTY + cell_reward
        return pos, reward, done

    def step(self, action):
        self.agent_pos = self._move_agent(self.agent_pos, action)
        self.randomize_layout()
        next_pos, reward, done = self._get_result(self.agent_pos)
        self.agent_pos = next_pos
        return next_pos, reward, done

    def get_all_states(self):
        return [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if (r, c) not in self.static_walls]

    def get_transitions(self, state, action):
        r, c = state
        nr, nc = r, c
        if action == 0: nr = max(0, r - 1)
        elif action == 1: nr = min(self.grid_size - 1, r + 1)
        elif action == 2: nc = max(0, c - 1)
        elif action == 3: nc = min(self.grid_size - 1, c + 1)
        if (nr, nc) in self.static_walls: nr, nc = r, c
        
        next_state = (nr, nc)
        is_terminal = (next_state == self.goal_pos)
        
        # Calculate Reward based on current grid state
        # (Since Bellman re-calculates every step, we use the visible current grid)
        type_name = self.cell_types.get(next_state, "EMPTY")
        cell_reward = CELL_TYPES.get(type_name, (0, 0))[0] if not is_terminal else GOAL_REWARD
        
        reward = STEP_PENALTY + cell_reward if not is_terminal else GOAL_REWARD
        return [(1.0, next_state, reward, is_terminal)]
