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

# Rewards
GOAL_REWARD = 100
TRAP_PENALTY = -50
STEP_PENALTY = -1

# Hyperparameters
DISCOUNT_FACTOR = 0.99
CONVERGENCE_THRESHOLD = 1e-6
