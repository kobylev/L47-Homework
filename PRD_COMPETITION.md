# Product Requirements Document (PRD): Drone Competition Module

## 1. Project Overview
The objective is to create a head-to-head comparison module between two distinct AI paradigms: **Model-Based Planning (Value Iteration)** and **Model-Free Reinforcement Learning (Q-Learning)**. Both agents will compete in the same "Smart City" dynamic environment to determine which approach is more robust under high volatility.

## 2. Competition Mechanics & Fair Play
*   **Synchronized Environment:** Both agents must face the **exact same environmental state** at every step. This means for a given episode $E$ at step $T$, the positions of dynamic obstacles and bonuses must be identical for both the Bellman Agent and the Q-Learning Agent.
*   **Agent Initialization:**
    *   **Bellman Agent:** Uses the Bellman Optimality Equation. In this dynamic environment, it will perform a "Real-Time Planning" approach (one-step look-ahead based on the current visible grid).
    *   **Q-Learning Agent:** Uses the learned Q-Table from the L48 training phase.
*   **Episode Constraints:** Both agents start at `(0, 0)` and attempt to reach `(11, 11)` within a fixed `MAX_STEPS` limit.

## 3. Comparative Metrics (The Scoreboard)
The module must track and display the following metrics for both agents:
1.  **Total Reward:** Cumulative reward earned across all episodes.
2.  **Average Steps:** The mean number of steps taken to reach the goal (efficiency).
3.  **Goal Rate (Success %):** The percentage of episodes where the agent successfully reached the delivery goal without timing out.

## 4. GUI & Visualization Requirements
*   **Dual-Drone Display:** Render both drones on the grid simultaneously.
    *   **Blue Drone:** Q-Learning Agent.
    *   **Purple Drone:** Bellman Agent (Value Iteration).
*   **Comparison Dashboard:** A modified right-side panel showing two columns of data (one for each agent) for real-time comparison of Reward, Steps, and Goal Rate.
*   **Competition Graph:** A dual-line graph showing the "Reward History" of both agents overlayed on the same axes.

## 5. Technical Requirements
*   **Module Decoupling:** The competition logic should reside in a new file (e.g., `code/competition.py`) to avoid breaking the standalone training modes.
*   **State Consistency:** The environment must provide a `get_state()` and `set_state()` or a shared random seed to ensure both agents experience the same "random" obstacles at every step.
