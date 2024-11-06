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
    def __init__(self, walls, goal_state):
        """
        Walls is a list of list of row and column [[row, column], [row, column],...] representing the walls in the game\n
        Goal state is a list of list of row and column [[row, column], [row, column],...]
        """
        self.walls = walls
        self.goal_state = goal_state
        self.node_count = 0


    def _initialize_stone_positions_to_weights(self):
        """
        Creates a dictionary that maps each box position to its corresponding weight.
        :return: Dictionary mapping box positions to weights.
        """
        if len(self.boxes) != len(self.stone_weights):
            raise ValueError("The number of boxes must match the number of stone weights.")
        
        # Create the mapping from box position to weight
        return {self.boxes[i]: self.stone_weights[i] for i in range(len(self.boxes))}


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
    
    def AStar(self, player_position, boxes, stone_weights):
        """
        This function is used for A* search.
        :param player_position: Position of the player
        :param boxes: Position of the boxes
        :param stone_weights: List of stone weights
        :return: Solution path and node count from A* search
        """
        boxes = [tuple(box) if isinstance(box, list) else box for box in boxes]

        stone_positions_to_weights = {box : weight for box, weight in zip(boxes, stone_weights)}
        astar_solver = AStar(self, player_position, boxes, stone_positions_to_weights)
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
    def __init__(self, player_position, boxes, stone_positions_to_weights, walls, goal_state, parent=None, string_move=""):
        """
        Initialize the game state with player's position, boxes, stone weights, and the move string
        """
        self.player_position = player_position
        self.boxes = boxes  # List of box positions
        self.stone_positions_to_weights = stone_positions_to_weights  # Mapping of box positions to stone weights
        self.walls = walls
        self.goal_state = goal_state
        self.parent = parent  # For path reconstruction
        self.string_move = string_move  # The string representation of the move sequence (e.g., "RURU")
        self.g = 0  # Path cost
        self.h = self.heuristic()  # Heuristic
        self.f = self.g + self.h  # Total cost

    def heuristic(self):
        """
        Heuristic function for A*.
        In this case, we will calculate the sum of Manhattan distances between each box and its closest goal.
        The distance is weighted by the stone weight.
        """
        total_distance = 0
        for box in self.boxes:
            weight = self.stone_positions_to_weights.get(box, 1)  # Default weight of 1 if the box has no weight
            min_distance = float('inf')
            for goal in self.goal_state:
                distance = abs(box[0] - goal[0]) + abs(box[1] - goal[1])  # Manhattan distance
                min_distance = min(min_distance, distance)
            total_distance += min_distance * weight  # Weighted distance
        return total_distance

    def is_goal(self):
        """
        Check if all boxes are on the goal locations.
        """
        return all(box in self.goal_state for box in self.boxes)

    def get_neighbors(self):
        """
        Generate possible next states by moving or pushing stones.
        """
        neighbors = []
        row, col = self.player_position
        directions = [(-1, 0, "u"), (1, 0, "d"), (0, -1, "l"), (0, 1, "r")]  # up, down, left, right
        
        # Check the possible directions for movement
        for r, c, move in directions:
            # Check if the next position is valid
            new_row, new_col = row + r, col + c
            if (new_row, new_col) in self.walls:  # If it's a wall, skip it
                continue

            # Check if the new position is a box and push it
            if (new_row, new_col) in self.boxes:
                new_box_index = self.boxes.index((new_row, new_col))
                new_box_position = (self.boxes[new_box_index][0] + r, self.boxes[new_box_index][1] + c)
                if new_box_position not in self.walls and new_box_position not in self.boxes:
                    # Create a new list of boxes by pushing the box
                    new_boxes = self.boxes[:]
                    new_boxes[new_box_index] = new_box_position
                    # Update the string move to include the new move (capitalize the move when pushing a box)
                    new_string_move = self.string_move + move.upper()
                    neighbors.append(AStar_GameState((new_row, new_col), new_boxes, self.stone_positions_to_weights, self.walls, self.goal_state, self, new_string_move))

            # Otherwise, just move the player without pushing a box
            else:
                new_string_move = self.string_move + move  # Lowercase for player-only movement
                neighbors.append(AStar_GameState((new_row, new_col), self.boxes, self.stone_positions_to_weights, self.walls, self.goal_state, self, new_string_move))

        return neighbors

    def __lt__(self, other):
        # For priority queue: compare based on total cost f = g + h
        return self.f < other.f

    def __repr__(self):
        return f"Player: {self.player_position}, Boxes: {self.boxes}, f: {self.f}, Moves: {self.string_move}"

class AStar:
    def __init__(self, data, start_player_position, start_boxes, stone_positions_to_weights):
        self.data = data
        self.start_player_position = start_player_position
        self.start_boxes = start_boxes
        self.stone_positions_to_weights = stone_positions_to_weights
        self.visited = set()
        self.goal_state = data.goal_state
        self.walls = data.walls

    def search(self):
        """
        Perform A* search to solve the Sokoban puzzle.
        """
        # Initialize the starting state
        start_state = AStar_GameState(self.start_player_position, self.start_boxes, self.stone_positions_to_weights, self.walls, self.goal_state, string_move="")
        
        # Priority queue for A* search
        open_list = []
        heapq.heappush(open_list, start_state)
        self.visited.add(tuple(start_state.boxes))

        while open_list:
            current_state = heapq.heappop(open_list)
            if current_state.is_goal():
                # Return both the solution (string_move) and node count
                return current_state, len(self.visited)

            for neighbor in current_state.get_neighbors():
                if tuple(neighbor.boxes) not in self.visited:
                    self.visited.add(tuple(neighbor.boxes))
                    heapq.heappush(open_list, neighbor)
                    print(neighbor.string_move)

        return None, len(self.visited)  # If no solution




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

