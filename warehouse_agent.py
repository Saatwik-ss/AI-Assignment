from collections import deque

warehouse_map = """
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
"""

print("Warehouse Layout:")
print(warehouse_map)

lines = warehouse_map.strip().split('\n')
grid = [list(line) for line in lines]

start_pos = None
goal_pos = None

for i in range(len(grid)):
    for j in range(len(grid[i])):
        if grid[i][j] == 'S':
            start_pos = (i, j)
        elif grid[i][j] == 'G':
            goal_pos = (i, j)

print(f"Start: {start_pos}")
print(f"Goal: {goal_pos}")
print(f"Grid: {len(grid)} x {len(grid[0])}\n")

class WarehouseAgent:
    def __init__(self, map_str):
        lines = map_str.strip().split('\n')
        self.grid = [list(line) for line in lines]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        
        self.start = None
        self.goal = None
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j] == 'S':
                    self.start = (i, j)
                elif self.grid[i][j] == 'G':
                    self.goal = (i, j)
        
        self.states_explored = 0
    
    def is_valid_move(self, pos):
        row, col = pos
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return False
        return self.grid[row][col] != '#'
    
    def get_neighbors(self, pos):
        row, col = pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neighbors = []
        for dr, dc in directions:
            new_pos = (row + dr, col + dc)
            if self.is_valid_move(new_pos):
                neighbors.append(new_pos)
        return neighbors
    
    def bfs_search(self):
        if not self.start or not self.goal:
            return None, 0
        
        self.states_explored = 0
        queue = deque([self.start])
        visited = {self.start}
        parent = {self.start: None}
        
        while queue:
            current = queue.popleft()
            self.states_explored += 1
            
            if current == self.goal:
                path = []
                node = current
                while node is not None:
                    path.append(node)
                    node = parent[node]
                return path[::-1], len(path) - 1
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
        
        return None, 0
    
    def solve(self):
        path, length = self.bfs_search()
        return {
            'path': path,
            'length': length,
            'states_explored': self.states_explored,
            'found': path is not None
        }
    
    def show_result(self, result):
        print(f"Solution found: {result['found']}")
        if result['found']:
            print(f"Path length: {result['length']}")
            print(f"States explored: {result['states_explored']}")
            if len(result['path']) <= 10:
                print(f"Path: {result['path']}")
        return result

agent = WarehouseAgent(warehouse_map)
result = agent.solve()
agent.show_result(result)

# Validate path
if result['found']:
    path = result['path']
    print(f"\nValidation:")
    print(f"  Start: {path[0]}")
    print(f"  Goal: {path[-1]}")
    
    all_valid = all(agent.is_valid_move(pos) for pos in path)
    print(f"  All positions valid: {all_valid}")
    
    adjacent = True
    for i in range(len(path)-1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        if abs(r1-r2) + abs(c1-c2) != 1:
            adjacent = False
            break
    print(f"  All moves adjacent: {adjacent}")

# Test 1: Trivial (adjacent goal)
print("\n" + "="*60)
print("Test 1: Goal Adjacent")
print("="*60)

trivial_map = """
#####
#SG##
#####
"""

agent1 = WarehouseAgent(trivial_map)
result1 = agent1.solve()
agent1.show_result(result1)
print(f"Expected length 1: {result1['length'] == 1}")

# Test 2: Impossible
print("\n" + "="*60)
print("Test 2: Unreachable Goal")
print("="*60)

impossible_map = """
#######
#S....#
###.###
#...#G#
#######
"""

agent2 = WarehouseAgent(impossible_map)
result2 = agent2.solve()
agent2.show_result(result2)
print(f"Expected no solution: {not result2['found']}")

# Test 3: Multiple paths
print("\n" + "="*60)
print("Test 3: Multiple Paths")
print("="*60)

multi_map = """
#########
#S.....G#
#.#####.#
#.......#
#########
"""

agent3 = WarehouseAgent(multi_map)
result3 = agent3.solve()
agent3.show_result(result3)

if result3['found']:
    manhattan = abs(agent3.goal[0]-agent3.start[0]) + abs(agent3.goal[1]-agent3.start[1])
    print(f"Manhattan distance: {manhattan}")
    print(f"Actual path length: {result3['length']}")
    print(f"BFS finds shortest path: {result3['length'] >= manhattan}")
