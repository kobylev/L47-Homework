import numpy as np
import os
import sys
import time
import pygame
import csv

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.config import (
    N_EPISODES, MAX_STEPS_PER_EPISODE, CELL_SIZE, GRID_SIZE, HISTORY_LOG_PATH
)
from code.environment import GridWorld
from code.agent import QLearningAgent
from code.gui import SmartCityGUI

def save_to_csv(data):
    file_exists = os.path.isfile(HISTORY_LOG_PATH)
    with open(HISTORY_LOG_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Episode", "Reward", "Alpha", "Epsilon"])
        writer.writerow(data)

def train_and_render():
    env = GridWorld()
    agent = QLearningAgent()
    gui = SmartCityGUI()
    
    reward_history = []
    success_count = 0
    
    # Ensure assets directory exists
    os.makedirs(os.path.dirname(HISTORY_LOG_PATH), exist_ok=True)
    
    print(f"Starting Q-Learning in Dynamic Environment ({N_EPISODES} episodes)...")
    
    running = True
    for episode in range(1, N_EPISODES + 1):
        if not running: break
        
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < MAX_STEPS_PER_EPISODE:
            # 1. Handle Events (Mouse Clicks)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if x < GRID_SIZE * CELL_SIZE:
                        grid_c = x // CELL_SIZE
                        grid_r = y // CELL_SIZE
                        env.toggle_static_element((grid_r, grid_c))
            
            if not running: break
            
            # 2. Agent chooses and executes action
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            # 3. Agent learns
            agent.update_q_value(state, action, reward, next_state)
            
            # 4. Update metrics and render
            episode_reward += reward
            steps += 1
            state = next_state
            
            metrics = {
                "Episode": episode,
                "Alpha (LR)": f"{agent.alpha:.4f}",
                "Epsilon": f"{agent.epsilon:.4f}",
                "Steps": steps,
                "Current Reward": f"{episode_reward:.1f}",
                "Success Rate": f"{(success_count/episode)*100:.1f}%" if episode > 0 else "0%"
            }
            
            gui.update(env, metrics, reward_history)
            
        if not running: break
            
        if done and state == env.goal_pos:
            success_count += 1
            
        reward_history.append(episode_reward)
        
        # Save sample point to CSV
        save_to_csv([episode, episode_reward, agent.alpha, agent.epsilon])
        
        # 5. Decay hyperparameters
        agent.decay_hyperparameters()
        
        if episode % 100 == 0:
            print(f"Ep {episode}/{N_EPISODES} | Alpha: {agent.alpha:.3f} | Epsilon: {agent.epsilon:.3f} | Success Rate: {(success_count/episode)*100:.1f}%")

    print("Training complete.")
    gui.close()

if __name__ == "__main__":
    train_and_render()
