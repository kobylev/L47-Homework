import numpy as np
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.config import START_POS, GOAL_POS
from code.environment import GridWorld
from code.agent import ValueIterationSolver
from code.visualize import plot_agent_path, plot_value_heatmap, ensure_assets_dir

def run_value_iteration():
    env = GridWorld()
    solver = ValueIterationSolver(env)
    
    print("Starting Value Iteration...")
    start_time = time.time()
    iterations = solver.compute_v_iteration()
    end_time = time.time()
    
    print(f"Converged in {iterations} iterations.")
    print(f"Time taken: {(end_time - start_time) * 1000:.2f}ms")
    
    # Extract Policy
    policy = solver.extract_policy()
    print("Optimal policy extracted.")
    
    return solver, env

def generate_optimal_path(solver, env):
    """
    Follow the optimal policy from start to goal.
    """
    state = START_POS
    path = [state]
    total_reward = 0
    max_steps = 200
    steps = 0
    
    while state != GOAL_POS and steps < max_steps:
        r, c = state
        action = solver.policy[r, c]
        
        # In value iteration, transitions are known, but we take the first (most likely) one
        transitions = env.get_transitions(state, action)
        _, next_state, reward, _ = transitions[0]
        
        state = next_state
        path.append(state)
        total_reward += reward
        steps += 1
        
    return path, total_reward

if __name__ == "__main__":
    solver, env = run_value_iteration()
    
    # Generate and display path
    path, total_reward = generate_optimal_path(solver, env)
    print(f"Path Length: {len(path)}")
    print(f"Total Cumulative Reward: {total_reward}")
    
    # Save visualizations
    assets_dir = ensure_assets_dir()
    plot_value_heatmap(solver.v_table, save_path=os.path.join(assets_dir, 'value_heatmap.png'))
    plot_agent_path(path, save_path=os.path.join(assets_dir, 'optimal_path.png'))
    
    print(f"Visualizations saved to {assets_dir}/")
