# Product Requirements Document (PRD): Data Logging & Export Module

## 1. Project Objective
To enhance the "Drone Competition" module by implementing automated data logging, metric tracking, and visual exporting. This ensures that the head-to-head performance of the Q-Learning Agent and the Bellman Agent can be analyzed offline and comprehensively documented.

## 2. Functional Requirements
*   **Metric Logging (CSV):** At the end of every episode, the system must append the following metrics to a `competition_metrics.csv` file:
    *   Episode Number
    *   Q-Agent: Total Reward, Steps Taken, Cumulative Goal Rate
    *   Bellman Agent: Total Reward, Steps Taken, Cumulative Goal Rate
*   **Screenshot Automation:** The system must automatically capture and save a high-resolution screenshot of the Pygame GUI (Canvas + Dashboard) at the end of the final episode.
*   **Graph Export:** The system must automatically generate and export a clean, high-resolution Matplotlib line graph comparing the Reward History of both agents across all episodes.
*   **Data Structure:** All exported files must be saved in the `assets/` directory to maintain project organization.

## 3. Technical Constraints
*   **Format:** CSV for raw data, PNG for visual exports.
*   **Performance:** Disk I/O should not significantly bottleneck the real-time simulation loop. CSV appending should be efficient.
*   **Integration:** These features must be seamlessly integrated into the existing `code/competition.py` loop without breaking the dual-agent synchronization.
