import matplotlib.pyplot as plt
import numpy as np
import os
import time
from code.config import GRID_SIZE, START_POS, GOAL_POS, OBSTACLES, TRAPS
from code.benchmark_astar import a_star_search
from code.main import train, evaluate
from code.visualize import ensure_assets_dir

def plot_comparison_metrics(astar_time, q_train_time, astar_len, q_len):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Execution Time Comparison (Log Scale)
    categories = ['A* Search', 'Q-Learning\n(Training)']
    times = [astar_time * 1000, q_train_time * 1000] # ms
    
    ax1.bar(categories, times, color=['#3498db', '#e74c3c'], alpha=0.8)
    ax1.set_yscale('log')
    ax1.set_title('Execution / Training Latency (ms)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time (ms) - Log Scale')
    for i, v in enumerate(times):
        ax1.text(i, v, f"{v:.2f}ms", ha='center', va='bottom', fontweight='bold')

    # 2. Path Optimality
    path_lengths = [astar_len, q_len]
    ax2.bar(categories, path_lengths, color=['#2ecc71', '#f1c40f'], alpha=0.8)
    ax2.set_title('Path Optimality (Total Steps)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Steps')
    ax2.set_ylim(0, max(path_lengths) + 5)
    for i, v in enumerate(path_lengths):
        ax2.text(i, v, f"{v} steps", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    assets_dir = ensure_assets_dir()
    plt.savefig(os.path.join(assets_dir, 'benchmark_comparison.png'), dpi=300)
    plt.close()

def plot_dual_paths(astar_path, q_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    def draw_grid(ax, path, title, color):
        grid = np.zeros((GRID_SIZE, GRID_SIZE))
        for r, c in OBSTACLES: grid[r, c] = 1
        for r, c in TRAPS: grid[r, c] = 2
        grid[GOAL_POS] = 3
        grid[START_POS] = 4
        
        cmap = plt.cm.colors.ListedColormap(['white', 'black', 'red', 'green', 'blue'])
        ax.imshow(grid, cmap=cmap)
        
        rows, cols = zip(*path)
        ax.plot(cols, rows, marker='o', color=color, markersize=4, linewidth=2, label='Path')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(range(GRID_SIZE))
        ax.set_yticks(range(GRID_SIZE))
        ax.grid(True, linestyle='--', alpha=0.3)

    draw_grid(ax1, astar_path, "A* Search (Geometric Optimum)", "#3498db")
    draw_grid(ax2, q_path, "Q-Learning (Learned Robustness)", "#f1c40f")
    
    plt.tight_layout()
    assets_dir = ensure_assets_dir()
    plt.savefig(os.path.join(assets_dir, 'path_comparison.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Running benchmarks...")
    
    # Benchmark A*
    astar_path, astar_len, astar_time = a_star_search(START_POS, GOAL_POS, set(OBSTACLES))
    
    # Benchmark Q-Learning
    start_train = time.perf_counter()
    agent, env, _ = train()
    end_train = time.perf_counter()
    q_train_time = end_train - start_train
    
    q_path, _, _ = evaluate(agent, env)
    q_len = len(q_path) - 1 # steps
    
    # Generate Plots
    plot_comparison_metrics(astar_time, q_train_time, astar_len, q_len)
    plot_dual_paths(astar_path, q_path)
    
    print("Benchmark visualizations saved to assets/benchmark_comparison.png and assets/path_comparison.png")
