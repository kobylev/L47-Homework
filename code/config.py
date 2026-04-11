import numpy as np

# Grid Configuration
GRID_SIZE = 12
START_POS = (0, 0)
GOAL_POS = (11, 11)

# Randomized Reward Ranges (Per Episode)
GOAL_REWARD_RANGE = (50, 150)
TRAP_PENALTY_RANGE = (-100, -30)
WIND_PENALTY_RANGE = (-10, -2)
STEP_PENALTY = -1

# Randomized Element Counts
WALL_COUNT_RANGE = (15, 25)
TRAP_COUNT_RANGE = (5, 10)
WIND_COUNT_RANGE = (10, 15)

# Dynamic Elements Count (Legacy support if needed, but PRD says canvas is static during episode)
DYNAMIC_OBSTACLE_COUNT = 0 
DYNAMIC_BONUS_COUNT = 0

# Q-Learning Hyperparameters
GAMMA = 0.9
ALPHA_START = 0.6
ALPHA_MIN = 0.01
ALPHA_DECAY = 0.9998  # Much slower decay

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9992 # Much slower decay (Epsilon ~0.1 after 3000 eps)

# Bellman Agent Hyperparameters
DISCOUNT_FACTOR = 0.99
CONVERGENCE_THRESHOLD = 1e-6

# Training
N_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 200

# GUI Configuration
CELL_SIZE = 50
SIDEBAR_WIDTH = 300
WINDOW_WIDTH = (GRID_SIZE * CELL_SIZE) + SIDEBAR_WIDTH
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE
FPS = 60
