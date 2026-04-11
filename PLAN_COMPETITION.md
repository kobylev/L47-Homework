# Architecture & Execution Plan: Drone Competition Module

## 1. System Architecture
The competition module follows a **Shared-State Controller** pattern.

*   **Environment Sync (`environment.py`):** Refactored to handle two independent agent positions while locking the "Environment State" (dynamic obstacles/bonuses) per turn.
*   **Dual-Agent Engine (`competition.py`):** Orchestrates the simulation. It handles the sequential execution of actions within a single environmental time-step.
*   **Visual Comparison (`gui.py`):** A specialized dashboard view that splits metrics into two comparative columns.

## 2. Micro-Level Todo List

### Phase 1: Environment Parity
- [ ] **1.1** Update `GridWorld.__init__` to include `self.q_pos` and `self.vi_pos`.
- [ ] **1.2** Implement `GridWorld.reset_competition()` to set both positions to `START_POS`.
- [ ] **1.3** Implement `GridWorld.step_dual(action_q, action_vi)`:
    - [ ] Calculate new `q_pos`.
    - [ ] Calculate new `vi_pos`.
    - [ ] Regenerate dynamic obstacles/bonuses **ONCE**.
    - [ ] Calculate rewards/done for both agents based on the **same** new positions.
    - [ ] Return `((q_next, q_reward, q_done), (vi_next, vi_reward, vi_done))`.

### Phase 2: Agent Integration
- [ ] **2.1** Integrate L47 `ValueIterationSolver` into the L48 project.
- [ ] **2.2** Ensure `ValueIterationSolver` can compute optimal actions for the *current* grid snapshot (one-step look-ahead).
- [ ] **2.3** Load the pre-trained `q_table` from the L48 training history.

### Phase 3: GUI Evolution
- [ ] **3.1** Define `COLOR_DRONE_Q = (0, 0, 255)` (Blue) and `COLOR_DRONE_VI = (128, 0, 128)` (Purple).
- [ ] **3.2** Modify `draw_grid()` to accept two agent positions.
- [ ] **3.3** Implement `draw_comparison_dashboard()` with a 2-column layout.
- [ ] **3.4** Update `draw_graph()` to support two lines (Red and Purple).

### Phase 4: Competition Orchestration
- [ ] **4.1** Create `code/competition.py`.
- [ ] **4.2** Define `N_COMP_EPISODES = 100`.
- [ ] **4.3** Implement the loop that alternates actions for both drones.
- [ ] **4.4** Store cumulative stats: `stats = {"Q": {"reward": 0, "steps": 0, "goals": 0}, "VI": {...}}`.
- [ ] **4.5** Save final comparison plot to `assets/competition_results.png`.

## 3. Strict Validation (Bits and Bytes)
- **PRD vs Todo Parity Check:**
    - "Exact same dynamic canvas" -> Covered by Task 1.3 (Regenerate ONCE per dual-step).
    - "Track Total Reward, Steps, Goal Rate" -> Covered by Task 4.4.
    - "Dual-Drone Display" -> Covered by Task 3.1 & 3.2.
    - "Comparison Graph" -> Covered by Task 3.4.
    - "Clean integration" -> Covered by Phase 4 (Separate module).

**Validation Result:** 100% of PRD requirements are mapped to micro-tasks.
