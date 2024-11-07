from collections import deque
from multiprocessing import Manager, Pool
import concurrent.futures
import threading
import heapq
import math
import itertools

data = None
### Data--------------------------------------------------------------------------------------------
class Initialized_data:
    """
    This function is used to initialize data that is not changed during bfs, dfs, UCS, A*\n
    This includes walls, goal_state
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


    def BFS(self, player_position, boxes):
        """
        This function is used for BFS\n
        :param player_position: Position of the player
        :param boxes: Position of the boxes
        :return: First BFS GameState object
        """
        # Initialize BFS
        first_BFS = BFS_GameState(player_position, boxes)
        return first_BFS.bfs(self)
    
    def DFS(self, player_position, boxes):
        first_DFS = DFS_GameState(player_position, boxes)
        return first_DFS.dfs(self)
    
    def AStar(self, player_position, boxes):
        """
        This function is used for A* search.
        :param player_position: Position of the player
        :param boxes: Position of the boxes
        :param stone_weights: List of stone weights
        :return: Solution path and node count from A* search
        """
        astar_solver = AStar(player_position, boxes, self.stone_weights, self)
        return astar_solver.search()

### Manager-----------------------------------------------------------------------------------------
class Manager_Algorithm:
    def __init__(self, data):
        """
        Initialize the Manager\n
        :param data: Initialized_data object
        """
        self.shared_stop_event = threading.Event()
        self.data = data
        self.manager = Manager()
        self.shared_visited_list = self.manager.list()

    def run_bfs(self, player_position, boxes):
        """
        This method is used for BFS\n
        :param player_position: Position of the player
        :param boxes: Position of the boxes
        :return: the goal state and the number of nodes explored
        """
        game_state = BFS_GameState(player_position, boxes, self.shared_visited_list)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(game_state.bfs, self.data, self.shared_stop_event)
            goal_state, node_counter = future.result()  # Wait for BFS to complete
            return goal_state, node_counter

    def run_astar(self, player_position, boxes, stone_weights):
        """
        This method is used for A* search
        :param player_position: Position of the player
        :param boxes: Position of the boxes
        :param stone_weights: List of stone weights
        :return: the solution path and number of nodes explored from A* search
        """
        # Initialize A* GameState for A* search
        astar_game_state = AStar_GameState(player_position, boxes, stone_weights, self.data.walls, self.data.goal_state)

        # Run A* search using ThreadPoolExecutor to handle asynchronous execution
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(astar_game_state.search)
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
    def __init__(self, player_pos, boxes, visited, string_move=""):
        """
        Initialize the game state with player's position and string move\n
        Player's position is a list of row and column [row, column]\n
        String move is a string representation of the move (e.g., "RURU")\n
        """
        self.player_pos = player_pos
        self.boxes = boxes
        self.string_move = string_move  # Store the string representation of the move (e.g., "RURU")
        self.visited = visited

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
                data.node_count += 1 # Count all the nodes that being generated
                neighbors.append(new_state)
            else: continue

        return neighbors
    
    def bfs(self, data, shared_stop_event):
        """
        Breadth-first search algorithm to find the shortest path to the goal state
        """
        queue = deque([self])

        # Use the shared list
        self.visited.append((self.player_pos, tuple(self.boxes)))

        while not shared_stop_event.is_set() and queue:
            batch_size = min(len(queue), 16)
            current_states = [queue.popleft() for _ in range(batch_size)]

            # Check if in the batch, there is a goal state or not
            for state in current_states:
                if state.is_goal_state(data.goal_state): 
                    print("done")
                    queue.clear()
                    print(data.node_count)
                    return state, data.node_count
            
            with Pool(processes=16) as pool:
                results = pool.starmap(self.generate_state, [(state, data) for state in current_states])

            for neighbors in results:
                for neighbor in neighbors:
                    if neighbor is None: continue

                    # Only add the neighbor to the visited list if it hasn't been added yet
                    visited_list = [neighbor.player_pos, neighbor.boxes]
                    if visited_list not in self.visited:
                        self.visited.append(visited_list)
                        queue.append(neighbor)
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
    
    def dfs(self, data):
        """
        Breadth-first search algorithm to find the shortest path to the goal state
        """
        with Manager() as manager:
            queue = deque([self])
            visited = manager.list() # Create a visited set
            visited.append([self.player_pos, self.boxes])

            with Pool(processes=16) as pool:
                while queue:
                    batch_size = min(len(queue), 16)
                    current_states = [queue.pop() for _ in range(batch_size)]

                    # Check if in the batch, there is a goal state or not
                    for state in current_states:
                        if state.is_goal_state(data.goal_state): 
                            print("done")
                            queue.clear()
                            print(data.node_count)
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
                                print(neighbor.string_move)
        return None, data.node_count
    
    def generate_state(self, state, data):
        """
        This function is only for multiprocessing
        """
        return state.get_neighbors(data)


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

    def is_goal_state(self, goal_state):
        """
        Check if all the boxes are on the goal positions.
        """
        return self.boxes == goal_state

    def get_neighbors(self, data):
        """
        Get possible next states after the player moves and pushes boxes.
        Returns a list of AStar_GameState objects.
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

            # Try to move in the direction
            boxes = self.boxes.copy()
            result = action(row, col, boxes, data, r, c)
            if result:
                new_player_pos = [new_row, new_col]

                # If a box is moved, capitalize the move direction
                if boxes != self.boxes:
                    move_direction = move_direction.upper()

                # Calculate the g-cost (the new cost incurred for this move)
                move_cost = self.calculate_move_cost(boxes, data)
                new_g_cost = self.g_cost + move_cost

                # Create the new string of moves
                new_string_move = self.string_move + move_direction

                # Create a new state
                new_state = AStar_GameState(new_player_pos, boxes, new_string_move, new_g_cost, self, data)

                # Calculate the heuristic (h-cost)
                h_cost = new_state.calculate_heuristic(new_state, data.goal_state)

                # Calculate the f-cost
                f_cost = new_g_cost + h_cost

                # Add the new state to the neighbors list
                neighbors.append((f_cost, new_state))

        return neighbors

    def calculate_move_cost(self, boxes, data):
        """
        Calculate the cost of the current move based on the weight of the box being pushed.
        """
        move_cost = 1  # Base cost of moving (without pushing a box)
        
        # Debugging: Print the number of boxes and the length of stone_weights
        print(f"Number of boxes: {len(boxes)}, Length of stone_weights: {len(data.stone_weights)}")

        if boxes != self.boxes:  # If a box was moved
            # Find which box was moved and add its weight to the cost
            for i, box in enumerate(boxes):
                # Ensure we don't go out of range by checking if there's a weight for this box
                if i < len(data.stone_weights):
                    print(f"Box {i} moved: {box}, adding weight {data.stone_weights[i]} to cost.")
                    move_cost += data.stone_weights[i]  # Add the weight of the moved box
                else:
                    # If there is no stone weight for this box, print a warning and set a default weight
                    print(f"Warning: No stone weight for Box {i} at position {box}. Using default weight 1.")
                    move_cost += 1  # Default weight if none is provided

        return move_cost


    def calculate_heuristic(self, state, goal_state):
        """
        Calculate the heuristic using the Manhattan distance for each box.
        """
        heuristic = 0
        for i, box in enumerate(state.boxes):
            goal_pos = goal_state[i]  # The goal position for the i-th box
            heuristic += abs(box[0] - goal_pos[0]) + abs(box[1] - goal_pos[1])
        return heuristic
    
    def __lt__(self, other):
        """
        Compare two game states for priority queue sorting.
        States with lower f_cost are given higher priority.
        """
        return self.f_cost < other.f_cost

class AStar:
    def __init__(self, start_player_pos, start_boxes, stone_weights, data):
        self.start_player_pos = start_player_pos
        self.start_boxes = start_boxes
        self.data = data  # Reference to Initialized_data instance
        self.open_list = []  # Min-heap priority queue
        self.closed_set = set()  # Explored states
        self.node_count = 0  # Number of nodes explored

        # Initialize the start node (initial state)
        start_node = AStar_GameState(start_player_pos, start_boxes, "", 0, None, data)
        heapq.heappush(self.open_list, (0, start_node))

    def search(self):
        """
        Perform A* search to find the optimal path.
        """
        while self.open_list:
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

            # Generate neighbors and explore them
            neighbors = current_state.get_neighbors(self.data)
            self.node_count += len(neighbors)

            for _, neighbor in neighbors:
                # If this neighbor has not been visited before, add it to the open list
                neighbor_boxes_tuple = tuple(tuple(box) for box in neighbor.boxes)  # Convert boxes to tuple of tuples

                if (tuple(neighbor.player_pos), neighbor_boxes_tuple) not in self.closed_set:
                    heapq.heappush(self.open_list, (
                        neighbor.g_cost + neighbor.calculate_heuristic(neighbor, self.data.goal_state), 
                        neighbor
                    ))

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

