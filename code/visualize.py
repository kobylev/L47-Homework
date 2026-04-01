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

def plot_agent_path(path, dynamic_obstacles=None, save_path=None):
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    
    # Fill grid for visualization
    # 0: Empty, 1: Obstacle, 2: Trap, 3: Goal, 4: Start, 5: Dynamic
    for r, c in OBSTACLES:
        grid[r, c] = 1
    for r, c in TRAPS:
        grid[r, c] = 2
    grid[GOAL_POS] = 3
    grid[START_POS] = 4
    
    if dynamic_obstacles:
        for r, c in dynamic_obstacles:
            grid[r, c] = 5

    plt.figure(figsize=(10, 10))
    # Custom colormap: white, black, red, green, blue, purple
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'red', 'green', 'blue', 'purple'])
    plt.imshow(grid, cmap=cmap)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='black', label='Static Wall'),
        Patch(facecolor='red', label='Trap Zone'),
        Patch(facecolor='green', label='Goal'),
        Patch(facecolor='blue', label='Start'),
        Patch(facecolor='purple', label='Dynamic Obstacle'),
        plt.Line2D([0], [0], color='gold', marker='o', label='Learned Path')
    ]
    
    # Extract path coordinates
    rows, cols = zip(*path)
    plt.plot(cols, rows, marker='o', color='gold', markersize=3, linewidth=1.5, alpha=0.8)
    
    plt.title('Agent Path Analysis (with Dynamic Obstacles)')
    plt.xticks(range(GRID_SIZE))
    plt.yticks(range(GRID_SIZE))
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1))
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def ensure_assets_dir():
    assets_dir = os.path.join(os.getcwd(), 'assets')
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
    return assets_dir
