import pygame
import numpy as np
from code.config import (
    GRID_SIZE, CELL_SIZE, SIDEBAR_WIDTH, WINDOW_WIDTH, WINDOW_HEIGHT,
    START_POS, GOAL_POS
)

# Colors
COLOR_EMPTY = (255, 255, 255)
COLOR_GRID = (200, 200, 200)
COLOR_GOAL = (0, 255, 0)

# Cell Type Colors
COLOR_MAP = {
    "EMPTY":       (255, 255, 255),
    "WALL":        (50, 50, 50),
    "WIND":        (173, 216, 230), # Light Blue
    "TRAP":        (200, 100, 100), # Dull Red
    "ROADBLOCK":   (255, 0, 0),     # Bright Red (Extreme Penalty)
    "BRIDGE":      (144, 238, 144), # Light Green (Bonus)
    "SHORTCUT":    (255, 215, 0),   # Gold (High Bonus)
}

COLOR_DRONE_Q = (0, 0, 255)
COLOR_DRONE_VI = (128, 0, 128)
COLOR_TEXT = (0, 0, 0)
COLOR_DASHBOARD = (240, 240, 240)

class SmartCityGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Drone Competition: Probabilistic Dynamic Smart City")
        self.font = pygame.font.SysFont("Arial", 16)
        self.header_font = pygame.font.SysFont("Arial", 20, bold=True)
        self.clock = pygame.time.Clock()

    def draw_grid(self, env, dual=False):
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                pos = (r, c)
                rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                
                if pos == GOAL_POS:
                    color = COLOR_GOAL
                else:
                    type_name = env.cell_types.get(pos, "EMPTY")
                    color = COLOR_MAP.get(type_name, COLOR_EMPTY)
                
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, COLOR_GRID, rect, 1)

        if dual:
            # Q-Agent
            qr, qc = env.q_pos
            q_center = (qc * CELL_SIZE + CELL_SIZE // 2, qr * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.circle(self.screen, COLOR_DRONE_Q, q_center, CELL_SIZE // 3)
            
            # VI-Agent
            vr, vc = env.vi_pos
            v_center = (vc * CELL_SIZE + CELL_SIZE // 2, vr * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.circle(self.screen, COLOR_DRONE_VI, v_center, CELL_SIZE // 4)
        else:
            r, c = env.agent_pos
            center = (c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.circle(self.screen, COLOR_DRONE_Q, center, CELL_SIZE // 3)

    def draw_comparison_dashboard(self, episode, q_stats, vi_stats, history_q, history_vi):
        dash_rect = pygame.Rect(GRID_SIZE * CELL_SIZE, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_DASHBOARD, dash_rect)
        y = 20
        self.screen.blit(self.header_font.render(f"EPISODE: {episode}", True, COLOR_TEXT), (GRID_SIZE * CELL_SIZE + 20, y))
        
        y += 40
        col1_x = GRID_SIZE * CELL_SIZE + 20
        col2_x = GRID_SIZE * CELL_SIZE + 160
        self.screen.blit(self.header_font.render("Q-AGENT", True, COLOR_DRONE_Q), (col1_x, y))
        self.screen.blit(self.header_font.render("BELLMAN", True, COLOR_DRONE_VI), (col2_x, y))
        
        y += 30
        labels = ["Total Reward", "Steps", "Goal Rate", "Epsilon"]
        for label in labels:
            self.screen.blit(self.font.render(label, True, COLOR_TEXT), (col1_x, y))
            y += 20
            q_val = self.font.render(str(q_stats.get(label, "0.0")), True, COLOR_DRONE_Q)
            vi_val = self.font.render(str(vi_stats.get(label, "0.0")), True, COLOR_DRONE_VI)
            self.screen.blit(q_val, (col1_x + 10, y))
            self.screen.blit(vi_val, (col2_x + 10, y))
            y += 25
            
        self.draw_dual_graph(history_q, history_vi, y + 20)

    def draw_dual_graph(self, hq, hvi, start_y):
        graph_width, graph_height = SIDEBAR_WIDTH - 40, 100
        graph_x, graph_y = GRID_SIZE * CELL_SIZE + 20, start_y + 25
        pygame.draw.rect(self.screen, (255, 255, 255), (graph_x, graph_y, graph_width, graph_height))
        pygame.draw.rect(self.screen, (0, 0, 0), (graph_x, graph_y, graph_width, graph_height), 1)
        if len(hq) < 2 or len(hvi) < 2: return
        all_vals = hq[-50:] + hvi[-50:]
        max_h, min_h = max(all_vals), min(all_vals)
        range_h = max_h - min_h if max_h != min_h else 1
        def get_p(hist, color):
            pts = []
            for i, val in enumerate(hist[-50:]):
                x = graph_x + (i / (len(hist[-50:]) - 1)) * graph_width
                y = graph_y + graph_height - ((val - min_h) / range_h) * graph_height
                pts.append((x, y))
            pygame.draw.lines(self.screen, color, False, pts, 2)
        get_p(hq, COLOR_DRONE_Q)
        get_p(hvi, COLOR_DRONE_VI)

    def update_competition(self, env, episode, q_stats, vi_stats, hq, hvi):
        self.screen.fill(COLOR_EMPTY)
        self.draw_grid(env, dual=True)
        self.draw_comparison_dashboard(episode, q_stats, vi_stats, hq, hvi)
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()
