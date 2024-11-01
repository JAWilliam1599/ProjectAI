# BFS
from collections import deque
from multiprocessing import Manager, Pool

class BFS_GameState:
    """
    This is the class implement the game state using BFS\n
    BFS doesn't support weights
    """
    def __init__(self, player_pos, boxes, goal_state, walls, path=None, string_move="", node_count=0):
        """
        Initialize the game state with player's position, boxes' positions, and goal state\n
        Walls is a list of list of row and column [[row, column], [row, column],...] representing the walls in the game\n
        Player's position is a list of row and column [row, column]\n
        Boxes' positions is a list of list of row and column [[row, column], [row, column],...]\n
        Goal state is a list of list of row and column [[row, column], [row, column],...]\n
        """
        self.player_pos = player_pos
        self.walls = walls # Make this global
        self.boxes = boxes # Make this as a key value of dictionary as not frequently moved
        self.goal_state = goal_state # Make this global also
        self.string_move = string_move  # Store the string representation of the move (e.g., "RURU")
        self.path = path if path else [] # Remove this as we can use string_move, or reconstruct the path for further optimization
        self.node_counter = node_count # Make this global to count the total number of node

    def is_goal_state(self):
        return self.goal_state == self.boxes  # Is the goal state is all the boxes is on the goal
        
    
    def get_neighbors(self):
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
            result = action(row, col, boxes, self.walls, r, c)
            if (result):
                # Create a new player position
                new_row, new_col = row + r, col + c
                new_players_position = [new_row, new_col]

                # Create a new path by appending the current state to the path
                new_path = self.path + [self]

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
                new_state = BFS_GameState(new_players_position, boxes, self.goal_state, self.walls, new_path, new_string_move, self.node_counter+1)
                neighbors.append(new_state)
            else: continue

        return neighbors
    
    def bfs(self):
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
                    current_states = [queue.popleft() for _ in range(batch_size)]

                    # Check if in the batch, there is a goal state or not
                    for state in current_states:
                        if state.is_goal_state(): 
                            print("done")
                            return state

                    results = pool.map(self.generate_state, current_states)

                    for neighbors in results:
                        for neighbor in neighbors:
                            if neighbor is None: continue

                            # Only add the neighbor to the visited list if it hasn't been added yet
                            visited_list = [neighbor.player_pos, neighbor.boxes]
                            if visited_list not in visited:
                                visited.append(visited_list)
                                queue.append(neighbor)
                                print(neighbor.string_move)
        return None
    
    def generate_state(self, state):
        """
        This function is only for multiprocessing
        """
        return state.get_neighbors()

def action(row, col, boxes, walls, move_ud, move_lr):
    """
    Move the player to the next position\n
    Returns 2 if the move and box is successful\n
    Returns 1 if the only move is successful\n
    Returns 0 if the move is not successful\n
    """
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

