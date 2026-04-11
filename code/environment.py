import numpy as np
import random
from code.config import (
    GRID_SIZE, START_POS, GOAL_POS, STEP_PENALTY,
    GOAL_REWARD_RANGE, TRAP_PENALTY_RANGE, WIND_PENALTY_RANGE,
    WALL_COUNT_RANGE, TRAP_COUNT_RANGE, WIND_COUNT_RANGE
)

class GridWorld:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.start_pos = START_POS
        self.goal_pos = GOAL_POS
        
        # Session-specific layout
        self.static_walls = set()
        self.static_traps = set()
        self.static_wind_zones = set()
        
        # Session-specific rewards
        self.current_goal_reward = 100
        self.current_trap_penalty = -50
        self.current_wind_penalty = -5
        
        # Agent positions
        self.q_pos = START_POS
        self.vi_pos = START_POS
        self.agent_pos = START_POS
        
        # Compatibility placeholders
        self.dynamic_obstacles = set()
        self.dynamic_bonuses = set()

    def randomize_layout(self):
        """Force-regenerates the entire city structure and physics."""
        # 1. Reset all structures
        self.static_walls = set()
        self.static_traps = set()
        self.static_wind_zones = set()
        
        # 2. Randomize rewards for this specific session
        self.current_goal_reward = random.randint(*GOAL_REWARD_RANGE)
        self.current_trap_penalty = random.randint(*TRAP_PENALTY_RANGE)
        self.current_wind_penalty = random.randint(*WIND_PENALTY_RANGE)
        
        # 3. Randomize Coordinates
        all_coords = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]
        forbidden = {self.start_pos, self.goal_pos}
        available = [c for c in all_coords if c not in forbidden]
        
        # Re-shuffle to ensure true randomness per call
        random.shuffle(available)
        
        # Sample Buildings
        n_walls = random.randint(*WALL_COUNT_RANGE)
        self.static_walls = set(available[:n_walls])
        available = available[n_walls:]
        
        # Sample Traps
        n_traps = random.randint(*TRAP_COUNT_RANGE)
        self.static_traps = set(available[:n_traps])
        available = available[n_traps:]
        
        # Sample Wind Zones
        n_wind = random.randint(*WIND_COUNT_RANGE)
        self.static_wind_zones = set(available[:n_wind])

    def reset_competition(self):
        """Resets agent positions AND triggers a fresh map randomization."""
        self.q_pos = self.start_pos
        self.vi_pos = self.start_pos
        self.randomize_layout() 
        return self.q_pos, self.vi_pos

    def reset(self):
        """Standard reset for training."""
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
        """Intra-episode step: Environment stays static."""
        self.q_pos = self._move_agent(self.q_pos, action_q)
        self.vi_pos = self._move_agent(self.vi_pos, action_vi)
        return self._get_result(self.q_pos), self._get_result(self.vi_pos)

    def _get_result(self, pos):
        reward = STEP_PENALTY
        done = False
        if pos == self.goal_pos:
            reward = self.current_goal_reward
            done = True
        elif pos in self.static_traps: reward = self.current_trap_penalty
        elif pos in self.static_wind_zones: reward = self.current_wind_penalty
        return pos, reward, done

    def step(self, action):
        self.agent_pos = self._move_agent(self.agent_pos, action)
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
        reward = STEP_PENALTY
        is_terminal = False
        if next_state == self.goal_pos:
            reward = self.current_goal_reward
            is_terminal = True
        elif next_state in self.static_traps: reward = self.current_trap_penalty
        elif next_state in self.static_wind_zones: reward = self.current_wind_penalty
        return [(1.0, next_state, reward, is_terminal)]
