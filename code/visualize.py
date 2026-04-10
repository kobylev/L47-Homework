import matplotlib.pyplot as plt
import numpy as np
import os
from code.config import GRID_SIZE, START_POS, GOAL_POS, OBSTACLES, TRAPS

def ensure_assets_dir():
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets')
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
    return assets_dir

def plot_value_heatmap(v_table, save_path=None):
    """
    Visualize the value function as a heatmap.
    """
    plt.figure(figsize=(10, 8))
    
    # Mask obstacles for better visualization
    masked_v = np.copy(v_table)
    for r, c in OBSTACLES:
        masked_v[r, c] = np.nan
        
    plt.imshow(masked_v, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Value (V)')
    
    # Annotate values
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if (r, c) not in OBSTACLES:
                plt.text(c, r, f'{v_table[r, c]:.1f}', ha='center', va='center', color='white', fontsize=8)
            else:
                plt.text(c, r, 'X', ha='center', va='center', color='red', fontweight='bold')

    plt.title('Value Function Heatmap (Converged)')
    plt.grid(visible=False)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_agent_path(path, save_path=None):
    """
    Visualize the final path from start to goal.
    """
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    
    # Mark special zones
    for r, c in OBSTACLES: grid[r, c] = -1
    for r, c in TRAPS: grid[r, c] = -0.5
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot grid
    ax.imshow(grid, cmap='Greys', origin='upper')
    
    # Draw path
    if path:
        path_rs, path_cs = zip(*path)
        ax.plot(path_cs, path_rs, marker='o', color='blue', linewidth=2, label='Agent Path')
        
    # Mark start and goal
    ax.scatter(START_POS[1], START_POS[0], color='green', s=200, label='Start', marker='*')
    ax.scatter(GOAL_POS[1], GOAL_POS[0], color='red', s=200, label='Goal', marker='X')
    
    # Add trap labels
    for r, c in TRAPS:
        ax.text(c, r, 'T', ha='center', va='center', color='white', fontweight='bold')
        
    ax.set_title("Optimal Path (Value Iteration)")
    ax.legend()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
