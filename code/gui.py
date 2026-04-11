import pygame
import numpy as np
from code.config import (
    GRID_SIZE, CELL_SIZE, SIDEBAR_WIDTH, WINDOW_WIDTH, WINDOW_HEIGHT,
    START_POS, GOAL_POS
)

# Colors
COLOR_EMPTY = (255, 255, 255)
COLOR_GRID = (200, 200, 200)
COLOR_DYNOBSTACLE = (255, 0, 0)     # Bright Red for Dynamic Obstacles
COLOR_DYNBONUS = (255, 215, 0)       # Gold for Bonuses
COLOR_GOAL = (0, 255, 0)             # Green

# Static Colors (User Placed)
COLOR_WALL = (50, 50, 50)            # Dark Grey
COLOR_TRAP = (200, 100, 100)          # Dull Red
COLOR_WIND = (173, 216, 230)          # Light Blue

COLOR_DRONE_Q = (0, 0, 255)          # Blue
COLOR_DRONE_VI = (128, 0, 128)        # Purple
COLOR_TEXT = (0, 0, 0)
COLOR_DASHBOARD = (240, 240, 240)

class SmartCityGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Drone Competition: Q-Learning vs Bellman Agent")
        self.font = pygame.font.SysFont("Arial", 16)
        self.header_font = pygame.font.SysFont("Arial", 20, bold=True)
        self.clock = pygame.time.Clock()

    def draw_grid(self, env, dual=False):
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                color = COLOR_EMPTY
                if (r, c) == GOAL_POS: color = COLOR_GOAL
                elif (r, c) in env.static_walls: color = COLOR_WALL
                elif (r, c) in env.static_traps: color = COLOR_TRAP
                elif (r, c) in env.static_wind_zones: color = COLOR_WIND
                elif (r, c) in env.dynamic_obstacles: color = COLOR_DYNOBSTACLE
                elif (r, c) in env.dynamic_bonuses: color = COLOR_DYNBONUS
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, COLOR_GRID, rect, 1)

        if dual:
            # Draw Q-Agent
            qr, qc = env.q_pos
            q_center = (qc * CELL_SIZE + CELL_SIZE // 2, qr * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.circle(self.screen, COLOR_DRONE_Q, q_center, CELL_SIZE // 3)
            q_label = self.font.render("Q", True, (255, 255, 255))
            self.screen.blit(q_label, (q_center[0]-5, q_center[1]-8))
            
            # Draw VI-Agent
            vr, vc = env.vi_pos
            v_center = (vc * CELL_SIZE + CELL_SIZE // 2, vr * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.circle(self.screen, COLOR_DRONE_VI, v_center, CELL_SIZE // 4)
            v_label = self.font.render("B", True, (255, 255, 255))
            self.screen.blit(v_label, (v_center[0]-5, v_center[1]-8))
        else:
            r, c = env.agent_pos
            center = (c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.circle(self.screen, COLOR_DRONE_Q, center, CELL_SIZE // 3)

    def draw_comparison_dashboard(self, episode, q_stats, vi_stats, history_q, history_vi):
        dash_rect = pygame.Rect(GRID_SIZE * CELL_SIZE, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_DASHBOARD, dash_rect)
        
        y = 20
        title = self.header_font.render(f"EPISODE: {episode}", True, COLOR_TEXT)
        self.screen.blit(title, (GRID_SIZE * CELL_SIZE + 20, y))
        
        y += 40
        # Q Column
        col1_x = GRID_SIZE * CELL_SIZE + 20
        self.screen.blit(self.header_font.render("Q-AGENT (Blue)", True, COLOR_DRONE_Q), (col1_x, y))
        # VI Column
        col2_x = GRID_SIZE * CELL_SIZE + 160
        self.screen.blit(self.header_font.render("BELLMAN (Purp)", True, COLOR_DRONE_VI), (col2_x, y))
        
        y += 30
        labels = ["Total Reward", "Steps", "Goal Rate", "Epsilon"]
        for label in labels:
            self.screen.blit(self.font.render(label, True, COLOR_TEXT), (col1_x, y))
            y += 20
            q_val = self.font.render(str(q_stats.get(label, "N/A")), True, COLOR_DRONE_Q)
            vi_val = self.font.render(str(vi_stats.get(label, "N/A")), True, COLOR_DRONE_VI)
            self.screen.blit(q_val, (col1_x + 10, y))
            self.screen.blit(vi_val, (col2_x + 10, y))
            y += 25
            
        self.draw_dual_graph(history_q, history_vi, y + 20)

    def draw_dual_graph(self, hq, hvi, start_y):
        graph_width = SIDEBAR_WIDTH - 40
        graph_height = 120
        graph_x = GRID_SIZE * CELL_SIZE + 20
        graph_y = start_y + 25
        pygame.draw.rect(self.screen, (255, 255, 255), (graph_x, graph_y, graph_width, graph_height))
        pygame.draw.rect(self.screen, (0, 0, 0), (graph_x, graph_y, graph_width, graph_height), 1)
        self.screen.blit(self.font.render("Reward Comparison", True, COLOR_TEXT), (graph_x, start_y))

        if len(hq) < 2 or len(hvi) < 2: return
        
        # Calculate scales
        all_vals = hq[-100:] + hvi[-100:]
        max_h = max(all_vals) if max(all_vals) != min(all_vals) else all_vals[0] + 1
        min_h = min(all_vals)
        range_h = max_h - min_h if max_h != min_h else 1
        
        def get_points(history, color):
            points = []
            slice_h = history[-100:]
            for i, val in enumerate(slice_h):
                x = graph_x + (i / (len(slice_h) - 1)) * graph_width
                y = graph_y + graph_height - ((val - min_h) / range_h) * graph_height
                points.append((x, y))
            pygame.draw.lines(self.screen, color, False, points, 2)

        get_points(hq, COLOR_DRONE_Q)
        get_points(hvi, COLOR_DRONE_VI)

    def update_competition(self, env, episode, q_stats, vi_stats, hq, hvi):
        self.screen.fill(COLOR_EMPTY)
        self.draw_grid(env, dual=True)
        self.draw_comparison_dashboard(episode, q_stats, vi_stats, hq, hvi)
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()
