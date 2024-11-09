from collections import deque
from multiprocessing import Manager, Pool
import concurrent.futures
import threading
import heapq
from queue import PriorityQueue
import numpy as np
import scipy.optimize   # Requires SciPy library
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
        self.walls = walls
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
        self.manager = Manager()
        self.shared_visited_list = self.manager.list()
        self.pool = None

    def run_bfs(self, player_position, boxes):
        """
        This method is used for BFS\n
        :param player_position: Position of the player
        :param boxes: Position of the boxes
        :return: the goal state and the number of nodes explored
        """
        game_state = BFS_GameState(player_position, boxes)
        self.pool = Pool(processes=16)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(game_state.bfs, self.data, self.shared_visited_list, self.shared_stop_event, self.pool)
            goal_state, node_counter = future.result()  # Wait for BFS to complete
            return goal_state, node_counter

    def run_dfs(self, player_position, boxes):
        """
        This method is used for BFS\n
        :param player_position: Position of the player
        :param boxes: Position of the boxes
        :return: the goal state and the number of nodes explored
        """
        game_state = DFS_GameState(player_position, boxes)
        self.pool = Pool(processes=16)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(game_state.dfs, self.data, self.shared_visited_list, self.shared_stop_event, self.pool)
            goal_state, node_counter = future.result()  # Wait for BFS to complete
            return goal_state, node_counter
            
    def run_ucs(self, player_position, boxes):
        # Initialize game state
        game_state = UCS_GameState(player_position, boxes)

        # Pass `data` to the UCS function
        goal_state, node_counter = UCS_GameState.ucs(game_state, self.data.goal_state, self.data)
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
        astar_game_state = AStar(player_position, boxes, self.data, 6)

        # Run A* search using ThreadPoolExecutor to handle asynchronous execution
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(astar_game_state.search, self.shared_stop_event)
            solution_path, nodes_explored = future.result()  # Wait for A* to complete
            return solution_path, nodes_explored
    
    def stop(self):
        """
        This function is called to stop all the algorithms
        """
        self.shared_stop_event.set()  # Signal all GameStates to stop

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

    def is_goal_state(self, goal_state):
        # Check if all the boxes are on the goal
        return goal_state == self.boxes # Is the goal state is all the boxes is on the goal
        
    
    def get_neighbors(self, data):
        """
        Get the possible next states after the player moves\n
        Returns a list of BFS_GameState objects
        """
        row, col = self.player_pos
        directions = [(-1,0), (1,0), (0,-1), (0,1)] # Up, down, left, right
        neighbors = []

        # Check up, down, left, right
        for r, c in directions:
            # Check if the next position is valid
            boxes = self.boxes.copy()
            result = action(row, col, boxes, data, r, c)
            if (result):
                # Create a new player position
                new_row, new_col = row + r, col + c
                new_players_position = [new_row, new_col]

                # Update the string representation of the move
                move_direction = ""
                if r == -1: move_direction = "u"
                elif r == 1: move_direction = "d"
                elif c == -1: move_direction = "l"
                elif c == 1: move_direction = "r"

                # Check if a box is moved
                if boxes != self.boxes:
                    move_direction = move_direction.upper()  # Capitalize the move direction when a box is moved

                new_string_move = self.string_move + move_direction

                # Create a new state
                new_state = BFS_GameState(new_players_position, boxes, new_string_move)
                neighbors.append(new_state)
            else: continue

        return neighbors
    
    def bfs(self, data, visited, shared_stop_event, pool):
        """
        Breadth-first search algorithm to find the shortest path to the goal state
        """
        queue = deque([self])

        # Use the shared list
        visited.append((self.player_pos, tuple(self.boxes)))

        while not shared_stop_event.is_set() and queue:
            batch_size = min(len(queue), 16)
            current_states = [queue.popleft() for _ in range(batch_size)]
            
            results = pool.starmap(self.generate_state, [(state, data) for state in current_states])

            for neighbors in results:
                for neighbor in neighbors:
                    if neighbor is None: continue

                    if neighbor.is_goal_state(data.goal_state):
                        print("done")
                        queue.clear()
                        return neighbor, data.node_count

                    # Only add the neighbor to the visited list if it hasn't been added yet
                    visited_list = [neighbor.player_pos, neighbor.boxes]
                    if visited_list not in visited:
                        visited.append(visited_list)
                        queue.append(neighbor)
                        data.node_count += 1
                        print(neighbor.string_move)
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

    def is_goal_state(self, goal_state):
        # Check if all the boxes are on the goal
        return goal_state == self.boxes # Is the goal state is all the boxes is on the goal
        
    
    def get_neighbors(self, data):
        """
        Get the possible next states after the player moves\n
        Returns a list of BFS_GameState objects
        """
        row, col = self.player_pos
        directions = [(-1,0), (1,0), (0,-1), (0,1)] # Up, down, left, right
        neighbors = []

        # Check up, down, left, right
        for r, c in directions:
            # Check if the next position is valid
            boxes = self.boxes.copy()
            result = action(row, col, boxes, data, r, c)
            if (result):
                # Create a new player position
                new_row, new_col = row + r, col + c
                new_players_position = [new_row, new_col]

                # Update the string representation of the move
                move_direction = ""
                if r == -1: move_direction = "u"
                elif r == 1: move_direction = "d"
                elif c == -1: move_direction = "l"
                elif c == 1: move_direction = "r"

                # Check if a box is moved
                if boxes != self.boxes:
                    move_direction = move_direction.upper()  # Capitalize the move direction when a box is moved

                new_string_move = self.string_move + move_direction

                # Create a new state
                new_state = DFS_GameState(new_players_position, boxes, new_string_move)
                data.node_count += 1 # Count all the nodes that being generated
                neighbors.append(new_state)
            else: continue

        return neighbors
    
    def dfs(self, data, visited, shared_stop_event, pool):
        """
        Depth-first search algorithm to find the shortest path to the goal state
        """
        queue = deque([self])

        # Use the shared list
        visited.append((self.player_pos, tuple(self.boxes)))

        while not shared_stop_event.is_set() and queue:
            batch_size = min(len(queue), 16)
            current_states = [queue.pop() for _ in range(batch_size)]

            # Check if in the batch, there is a goal state or not
            for state in current_states:
                if state.is_goal_state(data.goal_state): 
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
                        print(neighbor.string_move)
        return None, data.node_count
    
    def generate_state(self, state, data):
        """
        This function is only for multiprocessing
        """
        return state.get_neighbors(data)

### UCS--------------------------------------------------------------------------------------------
import heapq

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

    def is_goal_state(self, goal_state):
        """
        Check if all the boxes are on the goal positions.
        """
        return self.boxes == goal_state

    @staticmethod
    def action(row, col, boxes, data, r, c):
        """
        Implement the action logic to move the player and possibly push boxes.
        Return if a box was moved and the weight of the moved box (or None if none moved).
        """
        new_row, new_col = row + r, col + c
        if [new_row, new_col] in data.walls:
            return False, None

        if [new_row, new_col] in boxes:
            new_box_row, new_box_col = new_row + r, new_col + c
            if [new_box_row, new_box_col] in data.walls or [new_box_row, new_box_col] in boxes:
                return False, None

            # Get the index of the box and its weight from stone_weights
            box_index = boxes.index([new_row, new_col])
            box_weight = data.stone_weights[box_index]

            # Move the box and update the box positions
            boxes.remove([new_row, new_col])
            boxes.append([new_box_row, new_box_col])

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
            if [new_row, new_col] in data.walls:  # Skip if it's a wall
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
    def ucs(initial_state, goal_state, data):
        priority_queue = []
        heapq.heappush(priority_queue, (initial_state.g_cost, initial_state))
        visited = {}

        while priority_queue:
            current_cost, current_state = heapq.heappop(priority_queue)
            
            # Track the state by its position and boxes, and only expand if we find a lower cost
            state_key = (tuple(current_state.player_pos), tuple(map(tuple, current_state.boxes)))
            if state_key in visited and visited[state_key] <= current_cost:
                continue

            # Mark this state with the current cost
            visited[state_key] = current_cost

            if current_state.is_goal_state(goal_state):
                return current_state, data.node_count

            # Expand neighbors
            for neighbor in current_state.get_neighbors(data):
                neighbor_key = (tuple(neighbor.player_pos), tuple(map(tuple, neighbor.boxes)))
                if neighbor_key not in visited or visited[neighbor_key] > neighbor.g_cost:
                    heapq.heappush(priority_queue, (neighbor.g_cost, neighbor))

        return None, data.node_count


    

### A*---------------------------------------------------------------------------------------------

class AStar_GameState:
    def __init__(self, player_pos, boxes, string_move="", g_cost=0, parent=None, data=None):
        """
        Initialize the game state with player's position, boxes, g-cost, and parent reference.
        """
        self.player_pos = player_pos
        self.boxes = boxes
        self.string_move = string_move  # Move string (e.g., "RURU")
        self.g_cost = g_cost  # Cost to reach this state
        self.parent = parent  # Parent node for path reconstruction
        self.data = data  # Reference to Initialized_data instance
        
        # Calculate heuristic cost to the goal state (h_cost)
        self.h_cost = self.calculate_heuristic(self, self.data.goal_state)

        # Calculate the total cost (f_cost = g_cost + h_cost)
        self.f_cost = self.g_cost + self.h_cost
        
        # Move counter for debugging
        self.move_count = 0  # Start with 0 moves

    def is_goal_state(self, goal_state):
        """
        Check if all the boxes are on the goal positions.
        """
        return all(box in goal_state for box in self.boxes)

    def get_neighbors(self, data):
        """
        Get possible next states after the player moves and pushes boxes.
        Returns a list of AStar_GameState objects.
        """
        row, col = self.player_pos
        directions = [(-1, 0, "u"), (1, 0, "d"), (0, -1, "l"), (0, 1, "r")]  # Up, down, left, right
        neighbors = []

        # Check up, down, left, right
        for r, c, move_direction in directions:
            boxes = self.boxes.copy()

            # Perform the action to determine if the move is valid and update boxes if needed
            result = action(row, col, boxes, data, r, c)
            if result:
                # Create a new player position
                new_row, new_col = row + r, col + c
                new_player_pos = [new_row, new_col]

                # If a box was moved, capitalize the move direction
                if boxes != self.boxes:
                    move_direction = move_direction.upper()

                # Calculate the g-cost (path cost so far) and the heuristic h-cost
                move_cost = self.calculate_move_cost(boxes, data)
                new_g_cost = self.g_cost + move_cost
                new_string_move = self.string_move + move_direction

                # Create a new state with updated costs and moves
                new_state = AStar_GameState(new_player_pos, boxes, new_string_move, new_g_cost, self, data)

                # Debugging output for neighbor generation
                print(f"Generated move {move_direction} to position {new_player_pos}, cost: {new_g_cost}, "
                    f"f_cost: {new_state.f_cost}")

                # Add the new state to the neighbors list
                neighbors.append((new_state.f_cost, new_state))

            # If the move was invalid, skip to the next direction
            else:
                #print(f"Skipping move {move_direction} from {(row, col)} due to invalid action.")
                continue

        print(f"Generated {len(neighbors)} neighbors for player position {self.player_pos}")
        return neighbors


    def calculate_move_cost(self, boxes, data):
        """
        Calculate the cost of the current move based on the weight of the box being pushed.
        """
        move_cost = 1  # Base cost of moving (without pushing a box)

        if boxes != self.boxes:  # If a box was moved
            # Find which box was moved and add its weight to the cost
            for i, box in enumerate(boxes):
                # Ensure we don't go out of range by checking if there's a weight for this box
                if i < len(data.stone_weights):
                    move_cost += data.stone_weights[i]  # Add the weight of the moved box
                else:
                    # If there is no stone weight for this box, print a warning and set a default weight
                    print(f"Warning: No stone weight for Box {i} at position {box}. Using default weight 1.")
                    move_cost += 1  # Default weight if none is provided

        return move_cost    

    
    def calculate_heuristic(self, state, goal_state): 
        """
        Calculate a heuristic for the Sokoban puzzle that minimizes the total weighted distance between boxes and goals.
        This method uses an assignment approach to find the optimal box-goal pairing.
        """
        num_boxes = len(state.boxes)
        num_goals = len(goal_state)
        distance_matrix = []

        print("Calculating heuristic...")
        # Calculate the weighted Manhattan distances between each box and each goal
        for i, box in enumerate(state.boxes):
            row = []
            for goal in goal_state:
                manhattan_distance = abs(box[0] - goal[0]) + abs(box[1] - goal[1])
                weight = self.data.stone_weights[i] if i < len(self.data.stone_weights) else 1  # Default weight is 1
                weighted_distance = manhattan_distance * weight
                row.append(weighted_distance)
                print(f"Box {i} at {box} to Goal {goal} -> Manhattan Distance: {manhattan_distance}, "
                    f"Weight: {weight}, Weighted Distance: {weighted_distance}")
            distance_matrix.append(row)

        # Use the Hungarian algorithm to find the minimum-cost assignment of boxes to goals
        box_indices, goal_indices = scipy.optimize.linear_sum_assignment(distance_matrix)
        total_weighted_distance = sum(distance_matrix[box][goal] for box, goal in zip(box_indices, goal_indices))

        print("Optimal box-goal assignments:")
        for box, goal in zip(box_indices, goal_indices):
            print(f"Box {box} assigned to Goal {goal} with cost: {distance_matrix[box][goal]}")

        # Calculate the minimum distance between the worker and any box for the heuristic
        robot_box_distances = [
            abs(state.player_pos[0] - box[0]) + abs(state.player_pos[1] - box[1])
            for box in state.boxes
        ]
        min_robot_box_distance = min(robot_box_distances) if robot_box_distances else 0

        print(f"Minimum distance between robot and nearest box: {min_robot_box_distance}")
        print(f"Total weighted distance: {total_weighted_distance}, Total heuristic: {total_weighted_distance + min_robot_box_distance}")

        # Return the total cost as the heuristic
        return total_weighted_distance + min_robot_box_distance

    def __lt__(self, other):
        """
        Compare two game states for priority queue sorting.
        States with lower f_cost are given higher priority.
        """
        return self.f_cost < other.f_cost


class AStar:
    def __init__(self, start_player_pos, start_boxes, data, max_moves=None):
        self.start_player_pos = start_player_pos
        self.start_boxes = start_boxes
        self.data = data  # Reference to Initialized_data instance
        self.open_list = []  # Min-heap priority queue
        self.closed_set = set()  # Explored states
        self.node_count = 0  # Number of nodes explored
        self.max_moves = max_moves  # Maximum number of moves for debugging

        # Initialize the start node (initial state)
        start_node = AStar_GameState(start_player_pos, start_boxes, "", 0, None, data)
        heapq.heappush(self.open_list, (start_node.f_cost, start_node))
        print(f"Start point: Player Position = {start_node.player_pos}, "
                  f"Boxes = {start_node.boxes}, Goals = {data.goal_state}, f_cost = {start_node.f_cost}")
    def search(self, shared_stop_event):
        """
        Perform A* search to find the optimal path.
        Stops after `max_moves` moves for debugging if set.
        """
        move_counter = 0  # Initialize the move counter

        while self.open_list and not shared_stop_event.is_set():
            # Check if the move counter has reached the max_moves limit
            if self.max_moves is not None and move_counter >= self.max_moves:
                print(f"Debugging stop: Reached max moves limit of {self.max_moves}")
                return None, self.node_count

            # Get the state with the lowest f-cost
            f_cost, current_state = heapq.heappop(self.open_list)

            print(f"Exploring state: Player Position = {current_state.player_pos}, "
                f"Boxes = {current_state.boxes}, f_cost = {current_state.f_cost}")

            # If we reach the goal state, return the solution (string_move) and node count
            if current_state.is_goal_state(self.data.goal_state):
                print("Goal reached!")
                print(f"Solution path: {current_state.string_move}")
                return current_state, self.node_count  # Return both the solution and node count

            # Add current state to the closed set (convert player_pos and boxes to tuples)
            self.closed_set.add((
                tuple(current_state.player_pos),
                tuple(tuple(box) for box in current_state.boxes)  # Convert each box position to a tuple
            ))

            # Generate neighbors and explore them, passing closed_set to avoid redundant states
            neighbors = current_state.get_neighbors(self.data)
            self.node_count += len(neighbors) # qq j đây, mỗi cái được tạo thì là bằng 0, + neighbor thì cùng lắm là 4

            for _, neighbor in neighbors:
                # Check if this neighbor has not been visited before
                neighbor_boxes_tuple = tuple(tuple(box) for box in neighbor.boxes)  # Convert boxes to tuple of tuples

                if (tuple(neighbor.player_pos), neighbor_boxes_tuple) not in self.closed_set:
                    heapq.heappush(self.open_list, (
                        neighbor.f_cost,  # Use the precomputed f_cost (g_cost + h_cost)
                        neighbor
                    ))

            move_counter += 1  # Increment the move counter

        print("No solution found within the move limit.")
        return None, self.node_count  # No solution found




### Action-----------------------------------------------------------------------------------------

def action(row, col, boxes, data, move_ud, move_lr):
    """
    Move the player to the next position\n
    Returns 2 if the move and box is successful\n
    Returns 1 if the only move is successful\n
    Returns 0 if the move is not successful\n
    """
    walls = data.walls
    new_player_row = row + move_ud
    new_player_col = col + move_lr
    # Check if the next position is a wall
    if [new_player_row, new_player_col] in walls:
        return False

    # Check if the next position is a box
    for i, [box_row, box_col] in enumerate(boxes):
        if [box_row, box_col] == [new_player_row, new_player_col]:
            # Calculate the position of the box after the move
            new_box_row = box_row + move_ud
            new_box_col = box_col + move_lr
            
            # Check if the new box position is a wall
            if [new_box_row, new_box_col] in walls or [new_box_row, new_box_col] in boxes:
                return False
            
            # Move the box
            boxes[i] = [new_box_row, new_box_col]
            return True
   
    # Move the player
    return True

