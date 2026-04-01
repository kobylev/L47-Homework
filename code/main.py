import numpy as np
import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.config import (
    N_EPISODES, MAX_STEPS_PER_EPISODE, EPSILON_START,
    EPSILON_DECAY, EPSILON_MIN
)
from code.environment import GridWorld
from code.agent import QLearningAgent
from code.visualize import plot_rewards, plot_agent_path, ensure_assets_dir

def train():
    env = GridWorld()
    agent = QLearningAgent()
    epsilon = EPSILON_START
    all_episode_rewards = []
    
    print(f"Starting training for {N_EPISODES} episodes...")
    
    for episode in range(N_EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < MAX_STEPS_PER_EPISODE:
            action = agent.choose_action(state, epsilon)
            next_state, reward, done = env.step(action)
            
            agent.update(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            steps += 1
            
        all_episode_rewards.append(total_reward)
        
        # Decay epsilon
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{N_EPISODES} | Total Reward: {total_reward:.2f} | Epsilon: {epsilon:.4f}")
            
    print("Training complete.")
    return agent, env, all_episode_rewards

def evaluate(agent, env):
    """Evaluate final policy without exploration"""
    state = env.reset()
    path = [state]
    total_reward = 0
    done = False
    steps = 0
    
    while not done and steps < MAX_STEPS_PER_EPISODE:
        action = agent.choose_action(state, epsilon=0) # Greedy
        next_state, reward, done = env.step(action)
        state = next_state
        path.append(state)
        total_reward += reward
        steps += 1
        
    return path, total_reward, env.dynamic_obstacles

if __name__ == "__main__":
    agent, env, rewards = train()
    
    # Evaluate final agent
    path, final_reward, dyn_obs = evaluate(agent, env)
    print(f"Final Path Length: {len(path)}")
    print(f"Final Reward: {final_reward}")
    
    # Save visualizations
    assets_dir = ensure_assets_dir()
    plot_rewards(rewards, save_path=os.path.join(assets_dir, 'training_rewards.png'))
    plot_agent_path(path, dynamic_obstacles=dyn_obs, save_path=os.path.join(assets_dir, 'final_path.png'))
    
    print(f"Visualizations saved to {assets_dir}/")
