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
    """Generates a high-quality comparative line chart for the README."""
    plt.figure(figsize=(12, 6))
    plt.style.use('ggplot')
    
    plt.plot(history_q, color='blue', label='Q-Learning (Model-Free)', linewidth=2, alpha=0.8)
    plt.plot(history_vi, color='purple', label='Bellman Agent (Model-Based)', linewidth=2, alpha=0.8)
    
    # Calculate Moving Averages for trend analysis
    if len(history_q) > 5:
        ma_q = np.convolve(history_q, np.ones(5)/5, mode='valid')
        ma_vi = np.convolve(history_vi, np.ones(5)/5, mode='valid')
        plt.plot(range(4, len(history_q)), ma_q, color='darkblue', linestyle='--', label='Q-Agent Trend')
        plt.plot(range(4, len(history_vi)), ma_vi, color='indigo', linestyle='--', label='Bellman Trend')

    plt.title('Performance Comparison: High-Volatility Smart City', fontsize=14)
    plt.xlabel('Competition Episode', fontsize=12)
    plt.ylabel('Cumulative Reward (Per Step Rewards)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig(COMPETITION_GRAPH_PATH, dpi=300)
    plt.close()
    print(f"[✔] Comparative graph saved to: {COMPETITION_GRAPH_PATH}")

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

def run_competition(headless_mode=False):
    os.makedirs(ASSETS_DIR, exist_ok=True)
    env = GridWorld()
    q_agent = QLearningAgent()
    
    # --- 1. HEADLESS TRAINING ---
    train_q_agent_headless(env, q_agent)
    
    # --- 2. GUI/HEADLESS COMPETITION ---
    print(f"PHASE 2: COMPETITION ({'HEADLESS' if headless_mode else 'GUI'} MODE)")
    
    gui = None
    if not headless_mode:
        import pygame
        pygame.init() # Initialize ONLY if needed
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
            if not headless_mode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: running = False
            
            # Action Selection
            act_q = q_agent.choose_action(env.q_pos, epsilon=0.05) if not done_q else -1
            
            # Bellman dynamic step recalculation
            if not done_vi:
                vi_solver = ValueIterationSolver(env)
                vi_solver.compute_v_iteration()
                vi_solver.extract_policy()
                act_vi = vi_solver.policy[env.vi_pos[0], env.vi_pos[1]]
            else:
                act_vi = -1
            
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
            if not headless_mode:
                q_metrics = {"Total Reward": f"{reward_q:.1f}", "Steps": steps_q, 
                             "Goal Rate": f"{(stats_q['goals']/ep)*100:.1f}%", "Epsilon": f"{q_agent.epsilon:.2f}"}
                vi_metrics = {"Total Reward": f"{reward_vi:.1f}", "Steps": steps_vi, 
                              "Goal Rate": f"{(stats_vi['goals']/ep)*100:.1f}%", "Epsilon": "0.00"}
                gui.update_competition(env, ep, q_metrics, vi_metrics, history_q, history_vi)
            
        history_q.append(reward_q)
        history_vi.append(reward_vi)
        save_competition_csv([ep, reward_q, steps_q, round((stats_q['goals']/ep)*100, 1), 
                              reward_vi, steps_vi, round((stats_vi['goals']/ep)*100, 1)])
        
        if headless_mode:
            print(f"Episode {ep:2d} | Q: {reward_q:6.1f} | B: {reward_vi:6.1f} | Goal Rate: {(stats_q['goals']/ep)*100:.1f}%")

    if not headless_mode:
        import pygame
        pygame.image.save(gui.screen, COMPETITION_SCREENSHOT_PATH)
        gui.close()
    
    export_competition_graph(history_q, history_vi)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Drone Competition CLI")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    args = parser.parse_args()
    
    run_competition(headless_mode=args.headless)
