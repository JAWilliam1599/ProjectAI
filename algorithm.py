# BFS
from collections import *

class BFS_GameState:
    """
    This is the class implement the game state using BFS\n
    BFS doesn't support weights
    """
    def __init__(self, player_pos, boxes, goal_state):
        """
        Initialize the game state with player's position, boxes' positions, and goal state\n
        Player's position is a list of row and column [row, column]\n
        Boxes' positions is a list of list of row and column [[row, column], [row, column],...]\n
        Goal state is a list of list of row and column [[row, column], [row, column],...]\n
        """
        self.player_pos = player_pos
        self.boxes = boxes
        self.goal_state = goal_state
        self.path = [] # To track the path made

    def is_goal_state(self):
        return set(self.goal_state) == set(self.boxes) # Is the goal state is all the boxes is on the goal
    
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
            new_row, new_col = row + r, col + c

            # Check if the position is valid