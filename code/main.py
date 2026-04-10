import numpy as np
import os
import sys
import time
import pygame

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.config import START_POS, GOAL_POS, FPS, CELL_SIZE, GRID_SIZE
from code.environment import GridWorld
from code.agent import ValueIterationSolver
from code.gui import SmartCityGUI

def simulate_with_gui():
    env = GridWorld()
    solver = ValueIterationSolver(env)
    gui = SmartCityGUI()
    
    # 1. Compute Initial Optimal Policy
    print("Computing Initial Optimal Policy...")
    solver.compute_v_iteration()
    solver.extract_policy()
    
    total_episodes = 0
    goal_count = 0
    reward_history = []
    
    running = True
    while running:
        state = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        total_episodes += 1
        status = "NAVIGATING"
        
        while not done and running:
            # Handle Events (Mouse Clicks)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if x < GRID_SIZE * CELL_SIZE:
                        grid_c = x // CELL_SIZE
                        grid_r = y // CELL_SIZE
                        env.toggle_obstacle((grid_r, grid_c))
                        solver.compute_v_iteration()
                        solver.extract_policy()
            
            if not running: break

            # Follow optimal policy
            r, c = state
            action = solver.policy[r, c]
            transitions = env.get_transitions(state, action)
            _, next_state, reward, done = transitions[0]
            
            state = next_state
            env.agent_pos = state 
            episode_reward += reward
            steps += 1
            
            if done:
                if state == GOAL_POS:
                    status = "SUCCESS!"
                    goal_count += 1
                else:
                    status = "FAILED"
            
            # Metrics for Dashboard
            metrics = {
                "Flight Status": status,
                "Current Episode": total_episodes,
                "Steps Taken": steps,
                "Episode Reward": episode_reward,
                "Success Rate": f"{(goal_count/total_episodes)*100:.1f}%",
                "Total Success": goal_count
            }
            
            gui.update(env, metrics, reward_history)
            time.sleep(0.05) 

        if not running: break
        
        reward_history.append(episode_reward)
        if len(reward_history) > 100:
            reward_history.pop(0)

        # Brief pause to show SUCCESS status before reset
        time.sleep(0.8)

    gui.close()

if __name__ == "__main__":
    simulate_with_gui()
