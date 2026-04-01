import numpy as np

# Grid Configuration
GRID_SIZE = 12
START_POS = (0, 0)
GOAL_POS = (11, 11)

# Obstacles (Walls)
OBSTACLES = [
    (5, i) for i in range(2, 10)
] + [
    (i, 5) for i in range(2, 6)
]

# Trap Zones
TRAPS = [
    (2, 2), (8, 8), (3, 9), (9, 3)
]

# Dynamic Obstacles
N_DYNAMIC_OBSTACLES = 5
DYNAMIC_OBSTACLE_MOVE_CHANCE = 0.3 # 30% chance to move each step
DYNAMIC_OBSTACLE_PENALTY = -20

# Rewards
GOAL_REWARD = 100
TRAP_PENALTY = -50
STEP_PENALTY = -1

# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

# Training
N_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 500
