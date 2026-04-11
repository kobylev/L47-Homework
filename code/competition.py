import numpy as np
import os
import sys
import pygame
import csv
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.config import N_EPISODES, MAX_STEPS_PER_EPISODE, CELL_SIZE, GRID_SIZE
from code.environment import GridWorld
from code.agent import QLearningAgent, ValueIterationSolver
from code.gui import SmartCityGUI

# Export Paths
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets')
COMPETITION_LOG_PATH = os.path.join(ASSETS_DIR, 'competition_metrics.csv')
COMPETITION_GRAPH_PATH = os.path.join(ASSETS_DIR, 'competition_reward_graph.png')
COMPETITION_SCREENSHOT_PATH = os.path.join(ASSETS_DIR, 'competition_dashboard.png')

def save_competition_csv(data):
    file_exists = os.path.isfile(COMPETITION_LOG_PATH)
    with open(COMPETITION_LOG_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Episode", "Q_Reward", "Q_Steps", "Q_GoalRate", "VI_Reward", "VI_Steps", "VI_GoalRate"])
        writer.writerow(data)

def export_competition_graph(history_q, history_vi):
    plt.figure(figsize=(10, 5))
    plt.plot(history_q, color='blue', alpha=0.7, label='Q-Learning Agent', linewidth=1.5)
    plt.plot(history_vi, color='purple', alpha=0.7, label='Bellman Agent (VI)', linewidth=1.5)
    plt.title('Drone Competition: Reward History')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.savefig(COMPETITION_GRAPH_PATH)
    plt.close()

def train_q_agent_headless(env, agent):
    """Performs full training in the console for maximum stability and speed."""
    print("="*50)
    print(f"PHASE 1: HEADLESS TRAINING ({N_EPISODES} Episodes)")
    print("="*50)
    
    for ep in range(1, N_EPISODES + 1):
        state = env.reset() # Randomizes layout per episode
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < MAX_STEPS_PER_EPISODE:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            steps += 1
            
        agent.decay_hyperparameters()
        
        if ep % 500 == 0 or ep == 1:
            print(f"[Ep {ep:4d}] | Alpha: {agent.alpha:.4f} | Epsilon: {agent.epsilon:.4f} | Reward: {total_reward:7.1f}")

    print("\n[✔] Q-Agent Training Complete.\n" + "="*50)

def run_competition():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    env = GridWorld()
    q_agent = QLearningAgent()
    
    # --- 1. HEADLESS TRAINING ---
    train_q_agent_headless(env, q_agent)
    
    # --- 2. GUI COMPETITION ---
    print("PHASE 2: GUI COMPETITION (50 Episodes)")
    gui = SmartCityGUI()
    history_q, history_vi = [], []
    stats_q, stats_vi = {"goals": 0}, {"goals": 0}
    
    running = True
    for ep in range(1, 51):
        if not running: break
        
        env.reset_competition()
        vi_solver = ValueIterationSolver(env)
        vi_solver.compute_v_iteration()
        vi_solver.extract_policy()
        
        done_q, done_vi = False, False
        reward_q, reward_vi = 0, 0
        steps_q, steps_vi = 0, 0
        
        while (not done_q or not done_vi) and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
            
            # Action Selection
            act_q = q_agent.choose_action(env.q_pos, epsilon=0.05) if not done_q else -1
            act_vi = vi_solver.policy[env.vi_pos[0], env.vi_pos[1]] if not done_vi else -1
            
            (q_res, vi_res) = env.step_dual(act_q, act_vi)
            
            if not done_q:
                reward_q += q_res[1]; steps_q += 1
                if q_res[2]: 
                    done_q = True
                    if q_res[0] == env.goal_pos: stats_q["goals"] += 1
                elif steps_q >= MAX_STEPS_PER_EPISODE: done_q = True
                
            if not done_vi:
                reward_vi += vi_res[1]; steps_vi += 1
                if vi_res[2]: 
                    done_vi = True
                    if vi_res[0] == env.goal_pos: stats_vi["goals"] += 1
                elif steps_vi >= MAX_STEPS_PER_EPISODE: done_vi = True
            
            # Live GUI Update
            q_metrics = {"Total Reward": f"{reward_q:.1f}", "Steps": steps_q, 
                         "Goal Rate": f"{(stats_q['goals']/ep)*100:.1f}%", "Epsilon": f"{q_agent.epsilon:.2f}"}
            vi_metrics = {"Total Reward": f"{reward_vi:.1f}", "Steps": steps_vi, 
                          "Goal Rate": f"{(stats_vi['goals']/ep)*100:.1f}%", "Epsilon": "0.00"}
            
            gui.update_competition(env, ep, q_metrics, vi_metrics, history_q, history_vi)
            
        history_q.append(reward_q)
        history_vi.append(reward_vi)
        save_competition_csv([ep, reward_q, steps_q, round((stats_q['goals']/ep)*100, 1), 
                              reward_vi, steps_vi, round((stats_vi['goals']/ep)*100, 1)])

    pygame.image.save(gui.screen, COMPETITION_SCREENSHOT_PATH)
    export_competition_graph(history_q, history_vi)
    gui.close()

if __name__ == "__main__":
    run_competition()
