from collections import deque
from multiprocessing import Manager, Pool
import concurrent.futures
import threading

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

