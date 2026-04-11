# Architecture & Execution Plan: Data Logging & Export

## 1. System Architecture
*   **CSV Logger:** A dedicated function within `competition.py` that utilizes Python's built-in `csv` module to append a row of metrics at the end of each episode loop.
*   **Visual Exporter (`matplotlib`):** At the conclusion of the `N_COMP_EPISODES` loop, a function will process the stored `history_q` and `history_vi` arrays to generate a dual-line plot saved to `assets/competition_reward_graph.png`.
*   **GUI Snapshot (`pygame.image`):** A direct capture of the `gui.screen` surface just before the pygame window closes, saved to `assets/competition_dashboard.png`.

## 2. Micro-Level Todo List
- [ ] **1.1** Define `COMPETITION_LOG_PATH = "assets/competition_metrics.csv"`.
- [ ] **1.2** Define `COMPETITION_GRAPH_PATH = "assets/competition_reward_graph.png"`.
- [ ] **1.3** Define `COMPETITION_SCREENSHOT_PATH = "assets/competition_dashboard.png"`.
- [ ] **2.1** In `competition.py`, create `save_competition_csv(data)` to handle header creation and row appending.
- [ ] **2.2** Inside the main episode loop, construct the `data` array: `[Ep, Q_Reward, Q_Steps, Q_Rate, VI_Reward, VI_Steps, VI_Rate]`.
- [ ] **2.3** Call `save_competition_csv(data)` at the end of each episode.
- [ ] **3.1** After the episode loop concludes, call `pygame.image.save(gui.screen, COMPETITION_SCREENSHOT_PATH)`.
- [ ] **4.1** Import `matplotlib.pyplot as plt` in `competition.py`.
- [ ] **4.2** Create `export_competition_graph(history_q, history_vi)` function.
- [ ] **4.3** Plot `history_q` (Blue) and `history_vi` (Purple) on the same axes, add legends, labels, and save the figure.
- [ ] **4.4** Call `export_competition_graph` at the end of the simulation.

## 3. Strict Validation (Bits and Bytes)
| PRD Requirement | Todo List Task | Coverage |
| :--- | :--- | :--- |
| **Save screenshots of canvas/dashboard** | Task 3.1 | **100%** |
| **Export 'Reward History' graph** | Tasks 4.1 to 4.4 | **100%** |
| **Save metrics (Reward, Steps, Goal Rate) for both** | Tasks 2.1 & 2.2 | **100%** |
| **CSV/JSON format** | Task 2.1 (CSV implemented) | **100%** |
| **Save at regular intervals (every episode)** | Task 2.3 | **100%** |

**Validation Result:** 100% coverage achieved. Proceeding to README generation and Code Execution.
