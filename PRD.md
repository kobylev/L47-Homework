# Product Requirements Document (PRD): Smart City Drone Delivery

## 1. Project Overview
The goal of this project is to implement an autonomous drone navigation system that computes an optimal path across a 12x12 "Smart City" grid. The system uses the **Bellman Equation** and the **Value Iteration** algorithm to solve the Markov Decision Process (MDP) for an environment with known transition probabilities and rewards, featuring a real-time interactive GUI.

## 2. Environment Specifications (Smart City Canvas)
*   **Grid Structure:** A fixed 12x12 grid.
*   **Starting Point:** Fixed at coordinate `(0, 0)`.
*   **Goal Position:** Fixed at coordinate `(11, 11)`.
*   **Cell Types & Physics:**v
    *   **Empty Cell:** Standard navigable space. **Reward: -1** (Step penalty).
    *   **Building (Obstacle):** Impassable wall. The drone remains in its current cell if it attempts to enter.
    *   **Trap:** Dangerous area. **Reward: -5**.
    *   **Wind Zone:** Areas of turbulence. **Reward: -2**.
    *   **Goal:** The delivery destination. **Reward: +10**.
*   **Interactive Toggling:** Users can click on grid cells during simulation to add or remove buildings (obstacles). The system must re-calculate the optimal policy immediately.

## 3. Functional Requirements
*   **Algorithm (Value Iteration):**
    *   Initialize a Value Table $V(s)$ for all states.
    *   Iteratively update $V(s)$ using the Bellman Optimality Equation:
        $V(s) \leftarrow \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V(s')]$
    *   Continue iteration until convergence ($\Delta V < 10^{-6}$).
*   **GUI & Dashboard:**
    *   **Grid View:** Visual representation of the city with color-coded cells (Blue for Wind, Red for Traps, Grey for Buildings).
    *   **Metrics Panel:** Real-time display of Flight Status, Current Episode, Steps Taken, Episode Reward, Success Rate, and Total Success.
    *   **Live Analytics:** A dynamic line graph showing "Reward History" for the last 100 episodes.

## 4. Technical Constraints
*   **Language:** Python 3.10+.
*   **GUI Framework:** Pygame.
*   **Performance:** Sub-second convergence for value iteration, 60 FPS GUI refresh.

## 5. Success Criteria
1.  The drone dynamically reroutes when obstacles are toggled by the user.
2.  The "Reward History" graph updates in real-time as episodes complete.
3.  The dashboard accurately reflects the agent's telemetry and success metrics.
4.  The value iteration algorithm correctly accounts for wind zones and traps.
