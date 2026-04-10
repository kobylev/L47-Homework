import pygame
import numpy as np
from code.config import (
    GRID_SIZE, CELL_SIZE, SIDEBAR_WIDTH, WINDOW_WIDTH, WINDOW_HEIGHT,
    OBSTACLES, TRAPS, WIND_ZONES, START_POS, GOAL_POS
)

# Colors
COLOR_EMPTY = (255, 255, 255)
COLOR_GRID = (200, 200, 200)
COLOR_BUILDING = (50, 50, 50) # Dark Grey
COLOR_TRAP = (255, 100, 100)   # Red
COLOR_WIND = (173, 216, 230)   # Light Blue
COLOR_GOAL = (144, 238, 144)   # Light Green
COLOR_DRONE = (0, 0, 255)      # Blue
COLOR_TEXT = (0, 0, 0)
COLOR_DASHBOARD = (240, 240, 240)

class SmartCityGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Smart City Drone Delivery Dashboard")
        self.font = pygame.font.SysFont("Arial", 20)
        self.header_font = pygame.font.SysFont("Arial", 24, bold=True)
        self.clock = pygame.time.Clock()

    def draw_grid(self, env):
        # Draw base grid
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                
                # Determine cell color
                color = COLOR_EMPTY
                if (r, c) in env.obstacles: color = COLOR_BUILDING
                elif (r, c) in TRAPS: color = COLOR_TRAP
                elif (r, c) in WIND_ZONES: color = COLOR_WIND
                elif (r, c) == GOAL_POS: color = COLOR_GOAL
                
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, COLOR_GRID, rect, 1)

        # Draw Drone
        r, c = env.agent_pos
        center = (c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2)
        pygame.draw.circle(self.screen, COLOR_DRONE, center, CELL_SIZE // 3)

    def draw_dashboard(self, metrics, reward_history):
        dash_rect = pygame.Rect(GRID_SIZE * CELL_SIZE, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_DASHBOARD, dash_rect)
        
        # Render Metrics
        y_offset = 20
        title = self.header_font.render("DASHBOARD", True, COLOR_TEXT)
        self.screen.blit(title, (GRID_SIZE * CELL_SIZE + 20, y_offset))
        
        y_offset += 50
        for key, value in metrics.items():
            text = self.font.render(f"{key}: {value}", True, COLOR_TEXT)
            self.screen.blit(text, (GRID_SIZE * CELL_SIZE + 20, y_offset))
            y_offset += 30

        # Draw Reward Graph
        self.draw_graph(reward_history, y_offset + 20)

    def draw_graph(self, history, start_y):
        graph_width = SIDEBAR_WIDTH - 40
        graph_height = 150
        graph_x = GRID_SIZE * CELL_SIZE + 20
        graph_y = start_y + 30
        
        # Graph Background
        pygame.draw.rect(self.screen, (255, 255, 255), (graph_x, graph_y, graph_width, graph_height))
        pygame.draw.rect(self.screen, (0, 0, 0), (graph_x, graph_y, graph_width, graph_height), 1)
        
        title = self.font.render("Reward History (last 100)", True, COLOR_TEXT)
        self.screen.blit(title, (graph_x, start_y))

        if len(history) < 1:
            return

        # Plot points
        if len(history) == 1:
            points = [(graph_x, graph_y + graph_height // 2), (graph_x + graph_width, graph_y + graph_height // 2)]
        else:
            max_h = max(history) if max(history) != min(history) else history[0] + 1
            min_h = min(history)
            range_h = max_h - min_h if max_h != min_h else 1
            
            points = []
            slice_history = history[-100:]
            for i, val in enumerate(slice_history):
                x = graph_x + (i / (len(slice_history) - 1)) * graph_width
                y = graph_y + graph_height - ((val - min_h) / range_h) * graph_height
                points.append((x, y))
            
        pygame.draw.lines(self.screen, (255, 0, 0), False, points, 2)

    def update(self, env, metrics, reward_history):
        self.screen.fill((255, 255, 255))
        self.draw_grid(env)
        self.draw_dashboard(metrics, reward_history)
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()
