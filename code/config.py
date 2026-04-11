import numpy as np

# Grid Configuration
GRID_SIZE = 12
START_POS = (0, 0)
GOAL_POS = (11, 11)

# Probabilistic Element Configuration
# (Value, Probability)
CELL_TYPES = {
    "EMPTY":       (0,   0.60),  # 60% Chance
    "WIND":        (-2,  0.15),  # 15% Chance
    "TRAP":        (-3,  0.10),  # 10% Chance
    "ROADBLOCK":   (-5,  0.05),  # 5% Chance (Penalty only, not a wall)
    "BRIDGE":      (1,   0.07),  # 7% Chance
    "SHORTCUT":    (2,   0.03),  # 3% Chance
}

# Fixed Obstacles (Walls)
WALL_PROBABILITY = 0.10 # 10% of cells will be impassable walls

# Rewards
GOAL_REWARD = 100
STEP_PENALTY = -1

# Q-Learning Hyperparameters
GAMMA = 0.9
ALPHA_START = 0.1
ALPHA_MIN = 0.01
ALPHA_DECAY = 0.9995
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9992

# Bellman Agent Hyperparameters
DISCOUNT_FACTOR = 0.99
CONVERGENCE_THRESHOLD = 1e-6

# Training
N_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 200

# GUI
CELL_SIZE = 50
SIDEBAR_WIDTH = 300
WINDOW_WIDTH = (GRID_SIZE * CELL_SIZE) + SIDEBAR_WIDTH
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE
FPS = 60
