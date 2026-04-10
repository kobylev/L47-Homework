import matplotlib.pyplot as plt
import numpy as np
import os
import time
from code.config import GRID_SIZE, START_POS, GOAL_POS, OBSTACLES, TRAPS
from code.benchmark_astar import a_star_search
from code.main import run_value_iteration, generate_optimal_path
from code.visualize import ensure_assets_dir

def plot_comparison_metrics(astar_time, vi_time, astar_len, vi_len):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Execution Time Comparison
    categories = ['A* Search', 'Value Iteration']
    times = [astar_time * 1000, vi_time * 1000] # ms
    
    ax1.bar(categories, times, color=['#3498db', '#e74c3c'], alpha=0.8)
    ax1.set_title('Execution Latency (ms)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time (ms)')
    for i, v in enumerate(times):
        ax1.text(i, v, f"{v:.2f}ms", ha='center', va='bottom', fontweight='bold')

    # 2. Path Optimality
    path_lengths = [astar_len, vi_len]
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

def plot_dual_paths(astar_path, vi_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    def draw_grid(ax, path, title, color):
        grid = np.zeros((GRID_SIZE, GRID_SIZE))
        for r, c in OBSTACLES: grid[r, c] = 1
        for r, c in TRAPS: grid[r, c] = 2
        
        # Color definitions for visualization
        # 0: Empty, 1: Obstacle, 2: Trap
        cmap = plt.cm.colors.ListedColormap(['white', 'black', 'red'])
        ax.imshow(grid, cmap=cmap)
        
        if path:
            rows, cols = zip(*path)
            ax.plot(cols, rows, marker='o', color=color, markersize=4, linewidth=2, label='Path')
            
        # Mark start and goal
        ax.scatter(START_POS[1], START_POS[0], color='green', s=100, marker='*', label='Start')
        ax.scatter(GOAL_POS[1], GOAL_POS[0], color='red', s=100, marker='X', label='Goal')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(range(GRID_SIZE))
        ax.set_yticks(range(GRID_SIZE))
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()

    draw_grid(ax1, astar_path, "A* Search (Geometric Optimum)", "#3498db")
    draw_grid(ax2, vi_path, "Value Iteration (Dynamic Optimum)", "#f1c40f")
    
    plt.tight_layout()
    assets_dir = ensure_assets_dir()
    plt.savefig(os.path.join(assets_dir, 'path_comparison.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Running benchmarks...")
    
    # Benchmark A*
    astar_path, astar_len, astar_time = a_star_search(START_POS, GOAL_POS, set(OBSTACLES))
    
    # Benchmark Value Iteration
    start_vi = time.perf_counter()
    solver, env = run_value_iteration()
    end_vi = time.perf_counter()
    vi_time = end_vi - start_vi
    
    vi_path, _ = generate_optimal_path(solver, env)
    vi_len = len(vi_path) - 1 # steps
    
    # Generate Plots
    plot_comparison_metrics(astar_time, vi_time, astar_len, vi_len)
    plot_dual_paths(astar_path, vi_path)
    
    print("Benchmark visualizations saved to assets/benchmark_comparison.png and assets/path_comparison.png")
