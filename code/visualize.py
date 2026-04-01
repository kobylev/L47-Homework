import matplotlib.pyplot as plt
import numpy as np
import os
from code.config import GRID_SIZE, START_POS, GOAL_POS, OBSTACLES, TRAPS

def plot_rewards(episode_rewards, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_agent_path(path, save_path=None):
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    
    # Fill grid for visualization
    # 0: Empty, 1: Obstacle, 2: Trap, 3: Goal, 4: Start
    for r, c in OBSTACLES:
        grid[r, c] = 1
    for r, c in TRAPS:
        grid[r, c] = 2
    grid[GOAL_POS] = 3
    grid[START_POS] = 4

    plt.figure(figsize=(8, 8))
    # Custom colormap for better visualization
    # 0-empty (white), 1-wall (black), 2-trap (red), 3-goal (green), 4-start (blue)
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'red', 'green', 'blue'])
    plt.imshow(grid, cmap=cmap)
    
    # Extract path coordinates
    rows, cols = zip(*path)
    plt.plot(cols, rows, marker='o', color='gold', markersize=3, linewidth=1, label='Path')
    
    plt.title('Agent Path Analysis')
    plt.xticks(range(GRID_SIZE))
    plt.yticks(range(GRID_SIZE))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def ensure_assets_dir():
    assets_dir = os.path.join(os.getcwd(), 'assets')
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
    return assets_dir
