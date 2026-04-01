import heapq
import time
from code.config import GRID_SIZE, START_POS, GOAL_POS, OBSTACLES

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(start, goal, obstacles):
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    start_time = time.perf_counter()
    
    while frontier:
        current = heapq.heappop(frontier)[1]
        
        if current == goal:
            break
            
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dr, current[1] + dc)
            
            if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE:
                if neighbor in obstacles:
                    continue
                    
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(goal, neighbor)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current
    
    end_time = time.perf_counter()
    
    # Reconstruct path
    path = []
    curr = goal
    if goal not in came_from:
        return [], 0, (end_time - start_time)
        
    while curr is not None:
        path.append(curr)
        curr = came_from[curr]
    path.reverse()
    
    return path, cost_so_far[goal], (end_time - start_time)

if __name__ == "__main__":
    from code.config import OBSTACLES
    path, cost, duration = a_star_search(START_POS, GOAL_POS, set(OBSTACLES))
    print(f"A* Path Length: {cost}")
    print(f"A* Execution Time: {duration*1000:.4f}ms")
