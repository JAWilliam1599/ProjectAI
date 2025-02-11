from collections import deque
import multiprocessing
import concurrent.futures
import threading
import heapq
from queue import PriorityQueue
import numpy as np
from scipy.optimize import linear_sum_assignment

data = None
### Data--------------------------------------------------------------------------------------------
class Initialized_data:
    """
    This function is used to initialize data that is not changed during bfs, dfs, UCS, A*\n
    This includes walls, goal_state, total nodes generated, and stone_weights
    """
    def __init__(self, walls, goal_state, stone_weights = None):
        """
        Walls is a list of list of row and column [[row, column], [row, column],...] representing the walls in the game\n
        Goal state is a list of list of row and column [[row, column], [row, column],...]
        """
        self.walls = set(map(tuple, walls))
        self.goal_state = goal_state
        self.stone_weights = stone_weights
        self.node_count = 0

### Manager-----------------------------------------------------------------------------------------
class Manager_Algorithm:
    """
    This class is used to manage the BFS, DFS, UCS, and A* algorithms\n
    This includes the shared visited list, the shared stop event, and the pool for multiprocessing
    """
    def __init__(self, data):
        """
        Initialize the Manager\n
        :param data: Initialized_data object
        """
        self.shared_stop_event = threading.Event()
        self.data = data
        self.manager = multiprocessing.Manager()
        self.shared_visited_list = self.manager.list()
        self.pool = None
        self.poolsize = 16
        self.isTimeOut = False
        self.timer = None

    def run_bfs(self, player_position, boxes):
        """
        This method is used for BFS\n
        :param player_position: Position of the player
        :param boxes: Position of the boxes
        :return: the goal state and the number of nodes explored
        """
        game_state = BFS_GameState(player_position, boxes)
        self.pool = multiprocessing.Pool(processes=self.poolsize)

        # Run BFS using ThreadPoolExecutor to handle asynchronous execution
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(game_state.bfs, self.data, self.shared_visited_list, self.shared_stop_event, self.pool, self.poolsize)
            goal_state, node_counter = future.result()  # Wait for BFS to complete
            print(self.isTimeOut)
            if (self.isTimeOut): return 1, node_counter
            return goal_state, node_counter

    def run_dfs(self, player_position, boxes):
        """
        This method is used for BFS\n
        :param player_position: Position of the player
        :param boxes: Position of the boxes
        :return: the goal state and the number of nodes explored
        """
        game_state = DFS_GameState(player_position, boxes)
        self.pool = multiprocessing.Pool(processes=self.poolsize)

        # Run DFS using ThreadPoolExecutor to handle asynchronous execution
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(game_state.dfs, self.data, self.shared_visited_list, self.shared_stop_event, self.pool, self.poolsize)
            goal_state, node_counter = future.result()  # Wait for BFS to complete
            if (self.isTimeOut): return 1, node_counter
            return goal_state, node_counter
            
    def run_ucs(self, player_position, boxes):
        # Initialize game state
        game_state = UCS_GameState(player_position, boxes)

        # Run UCS using ThreadPoolExecutor to handle asynchronous execution
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(UCS_GameState.ucs, game_state, self.data.goal_state, self.data, self.shared_stop_event)
            goal_state, node_counter = future.result() # Wait for UCS to complete
            if (self.isTimeOut): return 1, node_counter
            return goal_state, node_counter


    def run_astar(self, player_position, boxes):
        """
        This method is used for A* search
        :param player_position: Position of the player
        :param boxes: Position of the boxes
        :param stone_weights: List of stone weights
        :return: the solution path and number of nodes explored from A* search
        """
        # Initialize A*
        astar_game_state = AStar(player_position, boxes, self.data)

        # Run A* search using ThreadPoolExecutor to handle asynchronous execution
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(astar_game_state.search, self.shared_stop_event)
            solution_path, nodes_explored = future.result()  # Wait for A* to complete
            if (self.isTimeOut): return 1, nodes_explored
            return solution_path, nodes_explored
    
    def stop(self):
        """
        This function is called to stop all the algorithms
        """
        self.shared_stop_event.set()  # Signal all GameStates to stop

    def timeOut(self):
        """
        # This function is called to stop all the algorithms after a timeout
        """
        print("Time Out: ")
        self.isTimeOut = True
        self.stop()


### BFS--------------------------------------------------------------------------------------------
class BFS_GameState:
    """
    This is the class implement the game state using BFS\n
    BFS doesn't support weights
    """
    def __init__(self, player_pos, boxes, string_move=""):
        """
        Initialize the game state with player's position and string move\n
        Player's position is a list of row and column [row, column]\n
        String move is a string representation of the move (e.g., "RURU")\n
        """
        self.player_pos = player_pos
        self.boxes = boxes
        self.string_move = string_move  # Store the string representation of the move (e.g., "RURU")
        
    def get_neighbors(self, data):
        """
        Get the possible next states after the player moves\n
        Returns a list of BFS_GameState objects
        """
        row, col = self.player_pos
        directions = [(-1,0, "u"), (1,0, "d"), (0,-1, "l"), (0,1, "r")] # Up, down, left, right
        neighbors = []

        # Check up, down, left, right
        for r, c, character in directions:
            # Check if the next position is valid
            boxes = self.boxes.copy()
            result = action(row, col, boxes, data, r, c)
            if (result==0): continue

            if (result == 2):
                character = character.upper()

            # Create a new state
            new_state = BFS_GameState([row + r, col + c], boxes, self.string_move + character)
            neighbors.append(new_state)

        return neighbors
    
    def bfs(self, data, visited, shared_stop_event, pool, pool_size=1):
        """
        Breadth-first search algorithm to find the shortest path to the goal state
        """
        queue = deque([self])

        # Use the shared list
        visited.append((self.player_pos, tuple(self.boxes)))

        while not shared_stop_event.is_set() and queue:
            batch_size = min(len(queue), pool_size)
            current_states = [queue.popleft() for _ in range(batch_size)]

            # Check if in the batch, there is a goal state or not
            for state in current_states:
                if is_goal_state(state, data.goal_state): 
                    print("done")
                    queue.clear()
                    return state, data.node_count
            
            results = pool.starmap(self.generate_state, [(state, data) for state in current_states])

            for neighbors in results:
                for neighbor in neighbors:
                    if neighbor is None: continue

                    # Only add the neighbor to the visited list if it hasn't been added yet
                    visited_list = [neighbor.player_pos, neighbor.boxes]
                    if visited_list not in visited:
                        visited.append(visited_list)
                        queue.append(neighbor)
                        data.node_count += 1
        return None, data.node_count

    def generate_state(self, state, data):
        """
        This function is only for multiprocessing
        """
        return state.get_neighbors(data)

### DFS--------------------------------------------------------------------------------------------
class DFS_GameState:
    """
    This is the class implement the game state using BFS\n
    BFS doesn't support weights
    """
    def __init__(self, player_pos, boxes, string_move=""):
        """
        Initialize the game state with player's position and string move\n
        Player's position is a list of row and column [row, column]\n
        String move is a string representation of the move (e.g., "RURU")\n
        """
        self.player_pos = player_pos
        self.boxes = boxes
        self.string_move = string_move  # Store the string representation of the move (e.g., "RURU")
        
    def get_neighbors(self, data):
        """
        Get the possible next states after the player moves\n
        Returns a list of DFS_GameState objects
        """
        row, col = self.player_pos
        directions = [(-1,0, "u"), (1,0, "d"), (0,-1, "l"), (0,1, "r")] # Up, down, left, right
        neighbors = []

        # Check up, down, left, right
        for r, c, character in directions:
            # Check if the next position is valid
            boxes = self.boxes.copy()
            result = action(row, col, boxes, data, r, c)
            if (result==0): continue

            if (result == 2):
                character = character.upper()

            # Create a new state
            new_state = BFS_GameState([row + r, col + c], boxes, self.string_move + character)
            neighbors.append(new_state)

        return neighbors
    
    def dfs(self, data, visited, shared_stop_event, pool, pool_size=1):
        """
        Depth-first search algorithm to find the shortest path to the goal state
        """
        queue = deque([self])

        # Use the shared list
        visited.append((self.player_pos, tuple(self.boxes)))

        while not shared_stop_event.is_set() and queue:
            batch_size = min(len(queue), pool_size)
            current_states = [queue.pop() for _ in range(batch_size)]

            # Check if in the batch, there is a goal state or not
            for state in current_states:
                if is_goal_state(state, data.goal_state): 
                    print("done")
                    queue.clear()
                    return state, data.node_count
            
            results = pool.starmap(self.generate_state, [(state, data) for state in current_states])

            for neighbors in results:
                for neighbor in neighbors:
                    if neighbor is None: continue

                    # Only add the neighbor to the visited list if it hasn't been added yet
                    visited_list = [neighbor.player_pos, neighbor.boxes]
                    if visited_list not in visited:
                        visited.append(visited_list)
                        queue.append(neighbor)
                        data.node_count += 1
        return None, data.node_count
    
    def generate_state(self, state, data):
        """
        This function is only for multiprocessing
        """
        return state.get_neighbors(data)

### UCS--------------------------------------------------------------------------------------------

class UCS_GameState:
    """
    This class implements the game state using Uniform Cost Search (UCS).
    UCS considers the cost to reach each node, expanding the least-cost path first.
    """
    def __init__(self, player_pos, boxes, string_move="", g_cost=0, parent=None):
        """
        Initialize the game state with player's position, boxes, g-cost, and parent reference.
        """
        self.player_pos = player_pos
        self.boxes = boxes
        self.string_move = string_move  # Move string (e.g., "RURU")
        self.g_cost = g_cost
        self.parent = parent

    def __lt__(self, other):
        """
        Less-than method for comparison, needed for PriorityQueue.
        """
        return self.g_cost < other.g_cost

    @staticmethod
    def action(row, col, boxes, data, r, c):
        """
        Implement the action logic to move the player and possibly push boxes.
        Return if a box was moved and the weight of the moved box (or None if none moved).
        """
        new_row, new_col = row + r, col + c
        if (new_row, new_col) in data.walls:
            return False, None

        if [new_row, new_col] in boxes:
            new_box_row, new_box_col = new_row + r, new_col + c
            if (new_box_row, new_box_col) in data.walls or [new_box_row, new_box_col] in boxes:
                return False, None

            # Get the index of the box and its weight from stone_weights
            box_index = boxes.index([new_row, new_col])
            box_weight = data.stone_weights[box_index]

            # Move the box and update the box positions
            boxes[box_index] = [new_box_row, new_box_col]

            # Return True (box moved) and the weight of the box
            return True, box_weight

        return True, None

    def get_neighbors(self, data):
        """
        Get possible next states after the player moves and pushes boxes.
        Returns a list of UCS_GameState objects.
        """
        directions = [(-1, 0, "u"), (1, 0, "d"), (0, -1, "l"), (0, 1, "r")]
        row, col = self.player_pos
        neighbors = []

        for r, c, move_direction in directions:
            # Calculate new player position
            new_row, new_col = row + r, col + c

            # Check if the next position is a wall
            if (new_row, new_col) in data.walls:  # Skip if it's a wall
                continue

            # Create a copy of boxes and attempt to move in the direction
            boxes = self.boxes.copy()
            box_moved, box_weight = UCS_GameState.action(row, col, boxes, data, r, c)
            
            if box_moved:
                new_player_pos = [new_row, new_col]

                # Capitalize move direction if a box was moved
                if boxes != self.boxes:
                    move_direction = move_direction.upper()

                # Calculate the move cost: base cost of 1 plus weight if a box is moved
                move_cost = 1
                if box_weight is not None:
                    move_cost += box_weight  # Add box weight to the move cost

                # Update the cumulative cost, considering both the player and box movement
                new_g_cost = self.g_cost + move_cost

                # Create the move string
                new_string_move = self.string_move + move_direction

                # Create the new state and add to neighbors
                new_state = UCS_GameState(new_player_pos, boxes, new_string_move, new_g_cost, self)
                neighbors.append(new_state)

        return neighbors

    @staticmethod
    def ucs(initial_state, goal_state, data, shared_stop_event):
        priority_queue = []
        heapq.heappush(priority_queue, (initial_state.g_cost, initial_state))
        visited = {}

        while priority_queue and not shared_stop_event.is_set():
            current_cost, current_state = heapq.heappop(priority_queue)

            if is_goal_state(current_state, goal_state):
                shared_stop_event.set()
                return current_state, data.node_count
            
            # Track the state by its position and boxes, and only expand if we find a lower cost
            state_key = (tuple(current_state.player_pos), tuple(map(tuple, current_state.boxes)))

            if state_key in visited and visited[state_key] <= current_cost:
                continue

            # Mark this state with the current cost
            visited[state_key] = current_cost

            # Expand neighbors
            neighbors = current_state.get_neighbors(data)
            data.node_count += len(neighbors)
            for neighbor in neighbors:
                neighbor_key = (tuple(neighbor.player_pos), tuple(map(tuple, neighbor.boxes)))
                if neighbor_key not in visited or visited[neighbor_key] > neighbor.g_cost:
                    heapq.heappush(priority_queue, (neighbor.g_cost, neighbor))
                    
        return None, data.node_count



### A*---------------------------------------------------------------------------------------------

class AStar_GameState:
    def __init__(self, player_pos, boxes, string_move="", g_cost=0, data=None):
        """
        Initialize the game state with player's position, boxes, g-cost and data of the map.
        """
        self.player_pos = player_pos
        self.boxes = boxes
        self.string_move = string_move  # Move string (e.g., "RURU")
        self.data = data  # Reference to Initialized_data instance
        
        # Cost to reach this state from the start state
        self.g_cost = g_cost 

        # Heuristic cost
        self.h_cost = self.calculate_heuristic(self, self.data.goal_state)

        # Total cost (f_cost = g_cost + h_cost)
        self.f_cost = self.g_cost + self.h_cost
        

    def is_goal_state(self, goal_state):
        """
        Check if all the boxes are on the goal positions.
        """
        return all(box in goal_state for box in self.boxes)

    def get_neighbors(self, data, closed_set):
        """
        Get possible next states after the player moves and pushes boxes.
        Returns a list of AStar_GameState objects.
        """
        row, col = self.player_pos
        directions = [(-1, 0, "u"), (1, 0, "d"), (0, -1, "l"), (0, 1, "r")]  # Up, down, left, right
        neighbors = []

        # Check up, down, left, right
        for r, c, move_direction in directions:
            # Calculate new player position
            new_player_pos = (row + r, col + c)

            # Perform the action to determine if the move is valid and update boxes if needed
            boxes = self.boxes.copy()
            result = action(row, col, boxes, data, r, c)
            if result == 0: 
                continue  # Skip if move is invalid

            # If a box was moved, capitalize the move direction
            if result == 2:
                move_direction = move_direction.upper()

            # Check if the new state (player position + boxes) has been explored
            new_state_tuple = (
                tuple(new_player_pos),
                tuple(tuple(box) for box in boxes)
            )

            if new_state_tuple not in closed_set:
                # Calculate the g-cost (path cost so far) and the heuristic h-cost
                move_cost = self.calculate_move_cost(boxes, data)
                new_g_cost = self.g_cost + move_cost
                new_string_move = self.string_move + move_direction

                # Create a new state with updated costs and moves
                new_state = AStar_GameState(new_player_pos, boxes, new_string_move, new_g_cost, data)

                # Add the new state to the neighbors list
                neighbors.append((new_state.f_cost, new_state))

        
        return neighbors

    def calculate_move_cost(self, boxes, data):
        """
        Calculate the cost of the current move based on the weight of the box being pushed.
        """
        move_cost = 1  # Base cost of moving (without pushing a box)

        # Check if any box has been moved
        for i, box in enumerate(boxes):
            if box != self.boxes[i]:
                # Add the weight of the moved box to the move cost
                move_cost += data.stone_weights[i] if i < len(data.stone_weights) else 1  # Default weight is 1

        return move_cost

    

    def calculate_heuristic(self, state, goal_state):
        """
        Calculate an admissible and consistent heuristic for the Sokoban puzzle.
        This heuristic considers:
        - The weighted distances between each box and each goal.
        - The minimum distance between the player and each unsolved box.
        """
        distance_matrix = []

        # Compute weighted Manhattan distances between each box and each goal
        for i, box in enumerate(state.boxes):
            row = []
            for goal in goal_state:
                # Calculate Manhattan distance and apply weight
                manhattan_distance = abs(box[0] - goal[0]) + abs(box[1] - goal[1])
                weight = self.data.stone_weights[i] if i < len(self.data.stone_weights) else 1  # Default weight is 1
                weighted_distance = manhattan_distance * weight
                row.append(weighted_distance)
            distance_matrix.append(row)

        # Use the Hungarian algorithm for optimal box-goal assignments
        box_indices, goal_indices = linear_sum_assignment(distance_matrix)
        total_weighted_distance = sum(distance_matrix[box][goal] for box, goal in zip(box_indices, goal_indices))

        # Calculate minimum distance between player and each unsolved box
        unsolved_boxes = [box for box in state.boxes if box not in goal_state]
        player_box_distances = [
            abs(state.player_pos[0] - box[0]) + abs(state.player_pos[1] - box[1])
            for box in unsolved_boxes
        ]
        min_player_box_distance = min(player_box_distances) if player_box_distances else 0

        return total_weighted_distance + min_player_box_distance

    def __lt__(self, other):
        """
        Compare two game states for priority queue sorting.
        States with lower f_cost are given higher priority.
        """
        return self.f_cost < other.f_cost

class AStar:
    def __init__(self, start_player_pos, start_boxes, data):
        self.start_player_pos = start_player_pos
        self.start_boxes = start_boxes
        self.data = data
        self.open_list = []  # Min-heap priority queue
        self.open_dict = {}  # Dictionary to track open states by their (player_pos, boxes) key
        self.closed_set = set()  # Explored states
        self.node_count = 0

        # Initialize the start node (initial state)
        start_node = AStar_GameState(start_player_pos, start_boxes, "", 0, data)
        heapq.heappush(self.open_list, (start_node.f_cost, start_node))
        self.open_dict[(tuple(start_node.player_pos), tuple(tuple(box) for box in start_node.boxes))] = start_node

    def search(self, shared_stop_event):
        while self.open_list and not shared_stop_event.is_set():
            f_cost, current_state = heapq.heappop(self.open_list)
            key = (tuple(current_state.player_pos), tuple(tuple(box) for box in current_state.boxes))

            # If the popped state is no longer in open_dict, skip it
            if key not in self.open_dict or self.open_dict[key].f_cost != f_cost:
                continue
            del self.open_dict[key]

            if is_goal_state(current_state, self.data.goal_state):
                print(f"Solution path: {current_state.string_move}")
                shared_stop_event.set()
                return current_state, self.node_count

            self.closed_set.add(key)
            neighbors = current_state.get_neighbors(self.data, self.closed_set)
            self.node_count += len(neighbors)

            for _, neighbor in neighbors:
                neighbor_key = (tuple(neighbor.player_pos), tuple(tuple(box) for box in neighbor.boxes))

                if neighbor_key in self.closed_set:
                    continue

                if neighbor_key not in self.open_dict or neighbor.f_cost < self.open_dict[neighbor_key].f_cost:
                    heapq.heappush(self.open_list, (neighbor.f_cost, neighbor))
                    self.open_dict[neighbor_key] = neighbor

        print("No solution found.")
        return None, self.node_count

### Action-----------------------------------------------------------------------------------------

def is_goal_state(self, goal_state):
    """
    Check if all the boxes are on the goal positions.
    """
    return all(box in goal_state for box in self.boxes)

def action(row, col, boxes, data, move_ud, move_lr):
    """
    Move the player to the next position
    Returns:
      2 if the move and box are both successful (box moved)
      1 if the player moved without moving a box
      0 if the move is not successful (blocked by wall or other conditions)
    """
    walls = data.walls
    new_player_row = row + move_ud
    new_player_col = col + move_lr

    # Check if the next player position is a wall
    if (new_player_row, new_player_col) in walls:
        return 0

    # Check if the player is moving into a box
    # improve
    if [new_player_row, new_player_col] in boxes:
        # Calculate the new box position after the move
        new_box_row = new_player_row + move_ud
        new_box_col = new_player_col + move_lr
        
        # Check if the new box position is a wall or another box
        if (new_box_row, new_box_col) in walls or [new_box_row, new_box_col] in boxes:
            return 0  # Box cannot be moved here, blocked by wall or another box
        
        # Check if the box is trapped
        if isTrapped(new_box_row, new_box_col, move_ud, move_lr, walls, data.goal_state):
            return 0  # Box is trapped, cannot move it

        box_index = boxes.index([new_player_row, new_player_col])
        boxes[box_index] = [new_box_row, new_box_col]
        return 2

    # If no box is moved, the player can move freely
    return 1

def isTrapped(box_row, box_col, move_ud, move_lr, walls, goal_state):
    """
    Check if the new box position is trapped by the player
    """
    #Move up, down
    if [box_row, box_col] in goal_state: return False
    if (move_ud != 0):
        if (box_row + move_ud, box_col) in walls and (box_row, box_col + 1) in walls: return True
        if (box_row + move_ud, box_col) in walls and (box_row, box_col - 1) in walls: return True
        return False
        
    #Move left, right
    if (box_row, box_col + move_lr) in walls and (box_row + 1, box_col) in walls: return True
    if (box_row, box_col + move_lr) in walls and (box_row - 1, box_col) in walls: return True
