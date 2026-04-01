# Reinforcement Learning: Tabular Q-Learning Grid World

A modular Python simulation of a Reinforcement Learning (RL) agent using the **Tabular Q-Learning** algorithm to navigate a 12x12 grid world.

## The Core Idea
The goal is to train an agent to find the most efficient path from a starting position (0,0) to a goal (11,11) in a grid containing obstacles (walls) and traps.

- **Agent**: Learns through interaction with the environment using the Bellman Equation.
- **Environment**: A 12x12 grid with static obstacles and penalty zones.
- **Reward Structure**: 
    - Goal: +100
    - Trap: -50
    - Step Penalty: -1 (encourages efficiency)
- **Algorithm**: Tabular Q-Learning with Epsilon-Greedy exploration.

## Project Structure
```text
C:\Ai_Expert\L47-Homework\
├── assets/                 # Generated visualizations (plots, paths)
├── code/
│   ├── config.py           # Hyperparameters and grid settings
│   ├── environment.py      # GridWorld class (MDP logic)
│   ├── agent.py            # QLearningAgent class (Bellman Eq)
│   ├── visualize.py        # Matplotlib plotting functions
│   ├── train.py            # (Reserved for expanded training loops)
│   └── main.py             # Entry point (orchestration)
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

## Data Flow / Architecture
The system follows a standard RL feedback loop:
1. **Agent** observes current **State** (row, col).
2. **Agent** selects an **Action** (Up, Down, Left, Right) using $\epsilon$-greedy strategy.
3. **Environment** processes action, returns **Reward** and **Next State**.
4. **Agent** updates its **Q-Table** using the Bellman Equation:
   $Q(s, a) \leftarrow Q(s, a) + \alpha [R + \gamma \max_{a'} Q(s', a') - Q(s, a)]$
5. Repeat until convergence.

## Results
*Plots are generated in the `assets/` directory.*
- **Training Rewards**: Shows convergence as the agent learns to avoid traps and find the goal faster.
- **Final Path**: A visualization of the grid with obstacles (black), traps (red), and the agent's learned path (gold).

## Honest Assessment
- **What worked**: The agent successfully learned to navigate around the complex wall structure and avoided all traps within 1000 episodes.
- **Challenges**: Early in training, the high exploration rate ($\epsilon$) leads to very low rewards. The `MAX_STEPS_PER_EPISODE` limit is crucial to prevent the agent from wandering indefinitely in the early random phase.
- **Why Tabular?**: For a 12x12 grid, the state space (144 states) is small enough for a table. For larger or continuous spaces, Deep Q-Learning (DQN) would be necessary.

## What Needs to Be Done (Next Steps)
| Task | Description |
|------|-------------|
| Dynamic Obstacles | Introduce moving obstacles to test temporal learning. |
| Stochastic Env | Add a probability that the agent moves in an unintended direction. |
| DQN Implementation | Transition from a table to a Neural Network for larger grids. |

## Setup & Usage

### 1. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Run Simulation
```bash
python -m code.main
```

## Dataset
This project uses a synthetic environment generated in `environment.py` based on standard RL benchmarking principles.
