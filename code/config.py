import numpy as np

# Grid Configuration
GRID_SIZE = 12
START_POS = (0, 0)
GOAL_POS = (11, 11)

# Obstacles (Walls) - More complex layout
OBSTACLES = [
    (5, i) for i in range(2, 10)
] + [
    (i, 5) for i in range(2, 6)
] + [
    (2, 7), (2, 8), (3, 7), (4, 7), # Top right cluster
    (9, 2), (9, 3), (8, 2), (7, 2), # Bottom left cluster
    (0, 10), (1, 10), (10, 0), (10, 1) # Corners
]

# Trap Zones
TRAPS = [
    (2, 2), (8, 8), (3, 9), (9, 3)
]

# Rewards
GOAL_REWARD = 10
TRAP_PENALTY = -5
WIND_ZONE_PENALTY = -2
STEP_PENALTY = -1

# Wind Zones (New feature from PRD)
WIND_ZONES = [
    (1, 1), (1, 2), (2, 1), (2, 2),
    (8, 9), (8, 10), (9, 9), (9, 10)
]

# Hyperparameters
DISCOUNT_FACTOR = 0.99
CONVERGENCE_THRESHOLD = 1e-6

# GUI Configuration
CELL_SIZE = 50
SIDEBAR_WIDTH = 300
WINDOW_WIDTH = (GRID_SIZE * CELL_SIZE) + SIDEBAR_WIDTH
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE
FPS = 60
