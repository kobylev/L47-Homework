# GridWorld Navigation: A Q-Learning Reinforcement Learning Study

## 🎯 Project Overview
This project implements a **Q-Learning agent** designed to navigate a complex $12 \times 12$ grid environment. The agent must find the optimal path from a starting position $(0, 0)$ to a goal $(11, 11)$ while avoiding static obstacles and high-penalty trap zones. 

The core objective is to demonstrate the efficacy of **Temporal Difference (TD)** learning in discrete state-action spaces, balancing exploration (discovering the environment) and exploitation (using learned knowledge to maximize rewards).

---

## 🏗️ Project Structure
```text
C:\Ai_Expert\L47-Homework\
├── README.md               # Detailed project documentation and analysis
├── requirements.txt        # Python dependencies (numpy, matplotlib)
├── .gitignore              # Git exclusion rules
├── assets\                 # Visual performance analysis & path results
│   ├── final_path.png      # Visualization of the agent's learned trajectory
│   └── training_rewards.png # Cumulative reward trends over training episodes
└── code\                   # Source implementation
    ├── __init__.py         # Package initialization
    ├── agent.py            # Q-Learning Agent implementation (Bellman updates)
    ├── config.py           # Environment parameters, rewards, & hyperparameters
    ├── environment.py      # GridWorld physics and reward logic
    ├── main.py             # Training loop and evaluation execution
    └── visualize.py        # Matplotlib-based plotting utilities
```

---

## 🧠 The Core Idea: Q-Learning & The Bellman Equation

The agent's "brain" is a **Q-Table**, a matrix of size $S \times A$ (where $S$ is the number of states and $A$ is the number of possible actions). Each entry $Q(s, a)$ represents the expected cumulative reward for taking action $a$ in state $s$ and following the optimal policy thereafter.

### 📐 Mathematical Expansion

The learning process is governed by the **Q-Learning update rule**, a specific instance of Temporal Difference learning:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right]$$

Where:
*   **$\alpha$ (Learning Rate):** Set to `0.1`, it determines to what extent newly acquired information overrides old information.
*   **$\gamma$ (Discount Factor):** Set to `0.9`, it controls the importance of future rewards. A value of $0.9$ ensures the agent values long-term success over immediate "step" rewards.
*   **$\text{TD Target} = r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a)$:** The current estimation of the state-action value.
*   **$\text{TD Error} = \text{TD Target} - Q(s_t, a_t)$:** The difference between the estimated future reward and the current Q-value.

### 🕹️ Strategy: $\epsilon$-Greedy Exploration
To solve the **Exploration-Exploitation Dilemma**, we employ an exponential $\epsilon$-decay strategy:
$$\epsilon_{t+1} = \max(\epsilon_{min}, \epsilon_t \times \text{decay})$$
*   **Initial $\epsilon$:** $1.0$ (Pure exploration)
*   **Decay Rate:** $0.995$
*   **Min $\epsilon$:** $0.01$ (Residual exploration to ensure robustness)

---

## 🌍 Environment Dynamics

The $12 \times 12$ GridWorld is not a simple walk. It incorporates:
*   **Static Obstacles (Walls):** Horizontal and vertical blocks that force the agent to find non-linear paths.
*   **Trap Zones:** High-penalty cells $(-50)$ that simulate "dangerous" areas to be avoided.
*   **Step Penalty:** A minor penalty $(-1)$ per movement to incentivize the discovery of the *shortest* path, not just *any* path.

| Feature | Coordinate / Value |
| :--- | :--- |
| **Grid Size** | $12 \times 12$ |
| **Start State** | $(0, 0)$ |
| **Goal State** | $(11, 11)$ |
| **Goal Reward** | $+100$ |
| **Trap Penalty** | $-50$ |
| **Step Penalty** | $-1$ |

---

## 📊 Results & Performance Analysis

### 1. Training Convergence
The following chart illustrates the agent's learning progress over 1,000 episodes. 

![Training Rewards](assets/training_rewards.png)

**Technical Breakdown:**
*   **Initial Volatility (Episodes 0-250):** The high variance in rewards is a direct result of the $\epsilon$-greedy strategy where $\epsilon \approx 1.0$. The agent is primarily "sampling" the environment, frequently encountering **Trap Zones $(-50)$** and accumulating high **Step Penalties**.
*   **The "Eureka" Moment (Episodes 250-500):** As the Q-values for the goal $(11, 11)$ propagate backwards via the Bellman update, we see a sharp logarithmic growth. This indicates the agent has discovered a stable sequence of actions that reach the goal.
*   **Saturation (Episodes 600+):** The curve flattens as the agent reaches the theoretical maximum reward for the shortest path $(\approx +85)$. The remaining small fluctuations are due to the residual exploration rate $(\epsilon_{min} = 0.01)$.

### 2. Path Optimization & Obstacle Avoidance
This visualization shows the final "Greedy" trajectory $(\epsilon = 0)$ mapped across the $12 \times 12$ grid.

![Final Path](assets/final_path.png)

**Strategic Analysis:**
*   **Efficient Manifolds:** The agent does not simply move diagonally. It has learned to navigate around the **L-shaped wall structure** (centered at row 5 and column 5) using the most efficient Manhattan distance path.
*   **Risk Mitigation:** Notice how the path maintains a "buffer zone" from the Trap coordinates $(2, 2)$ and $(8, 8)$. Even though the agent *could* pass closer, the learned Q-values for those transitions are significantly lower, guiding the agent toward safer, high-value state-action pairs.
*   **Shortest Path Verification:** The path length is approximately 22 steps, which is the mathematical minimum given the obstacle constraints $(11+11 = 22)$.

---

## 🛠️ Setup & Usage

### Prerequisites
*   Python 3.8+
*   `pip install -r requirements.txt`

### Execution
To train the agent and generate the analysis visualizations:
```bash
python code/main.py
```

---

## 🧐 Honest Assessment & Technical Insights

### ✅ Strengths
*   **Provable Convergence:** In a finite MDP like this GridWorld, Q-learning is guaranteed to converge to the optimal policy given enough exploration.
*   **Efficiency:** The discrete state space allows for rapid training ($1000$ episodes in seconds) without the need for complex neural networks (Deep Q-Learning).

### ⚠️ Limitations & Potential Improvements
*   **State Space Scalability:** A Q-Table grows exponentially with the number of features. For continuous environments (e.g., robotic arms), this approach would suffer from the "Curse of Dimensionality."
*   **Static Environment:** The current agent learns a fixed map. If obstacles moved, the agent would require **Deep Q-Networks (DQN)** or online re-learning.
*   **Next Step:** Implementation of **SARSA** (State-Action-Reward-State-Action) to compare "On-Policy" vs "Off-Policy" behavior, specifically looking at how SARSA's safer approach might avoid traps more conservatively during training.

---

## 🚀 Future Roadmap
- [ ] Implement **DQN** for continuous state space navigation.
- [ ] Add **Dynamic Obstacles** to test policy robustness.
- [ ] Compare performance against **A* Search** (Classical AI vs RL).
