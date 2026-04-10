# Product Requirements Document (PRD): Drone Navigation via Value Iteration

## 1. Project Overview
The goal of this project is to implement an autonomous drone navigation system that computes an optimal path across a 2D grid. The system uses the **Bellman Equation** and the **Value Iteration** algorithm to solve the Markov Decision Process (MDP) for a static environment with known transition probabilities and rewards.

## 2. Environment Specifications (Static Canvas)
*   **Grid Structure:** A fixed 12x12 grid.
*   **Starting Point:** Fixed at coordinate `(0, 0)`.
*   **Goal Position:** Fixed at coordinate `(11, 11)`.
*   **Static Obstacles (Walls):** Fixed coordinates that are impassable. If the drone attempts to move into an obstacle, it remains in its current cell.
*   **Traps & Penalties:** Specific fixed cells that incur a negative reward (penalty) to discourage entry.
*   **Transition Model:** Known transition probabilities for each action (e.g., deterministic movement where an action leads to the intended state with 100% probability, or a defined noise factor).

## 3. Functional Requirements
*   **Reward Function:**
    *   **Goal:** +100 reward upon reaching `(11, 11)`.
    *   **Trap:** -50 penalty for entering a trap cell.
    *   **Step Penalty:** -1 penalty for every movement to incentivize path efficiency.
*   **Algorithm (Value Iteration):**
    *   Initialize a Value Table $V(s)$ for all states.
    *   Iteratively update $V(s)$ using the Bellman Optimality Equation:
        $V(s) \leftarrow \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V(s')]$
    *   Continue iteration until the maximum change in the value function falls below a convergence threshold $\theta$.
*   **Policy Extraction:** 
    *   Once $V(s)$ converges, derive the optimal policy $\pi^*(s)$ by selecting the action that maximizes the expected value of the next state.
*   **Navigation:**
    *   Execute the derived policy from the start position to the goal.

## 4. Technical Constraints
*   **Language:** Python 3.x.
*   **Dependencies:** NumPy for matrix operations, Matplotlib for path and reward visualization.
*   **Performance:** The algorithm must converge in sub-second time for the 12x12 grid.

## 5. Success Criteria
1.  The Value Iteration algorithm correctly converges to a stable value function.
2.  The extracted policy generates a path that avoids all static obstacles.
3.  The path taken is mathematically optimal (shortest path while avoiding penalties).
4.  Visualization confirms the drone's trajectory from start to goal.
