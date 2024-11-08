# Import module  
from multiprocessing import Manager
from tkinter import *
import os
import time
import threading
import algorithm as a
import concurrent.futures
import time
import tracemalloc
  
"""
Note:
  + Resources
  - The image file is inside assets.
  - #222a5c is the blue background
  - #38002c is the red background of the button

  + Functions
  - Called reset_state before drawing or else, the widget inside the frame will not be destroyed

  + Patterns
  - Singleton pattern is used to manage the window instance, So that only one window is open at a time.
"""
class SingletonMeta(type):
  _instances = {}

  def __call__(cls, *args, **kwargs):
      if cls not in cls._instances:
          cls._instances[cls] = super().__call__(*args, **kwargs)
      return cls._instances[cls]

chosenAlgo = 0
chosenLevel = 1
filenames = []
# Iterate over all files in the folder
for filename in os.listdir("Testcase"):
    # Check if it's a file (not a directory)
    if os.path.isfile(os.path.join("Testcase", filename)):
        filenames.append(filename)
numberOfFiles = len(filenames)

class tkinterApp(Tk):
  # __init__ function for class tkinterApp 
  def __init__(self, *args, **kwargs): 

      # __init__ function for class Tk
      Tk.__init__(self, *args, **kwargs)
      self.title("Ares's Adventure")
      self.geometry("1050x649")
      self.resizable(False, False)
      self.wm_attributes("-transparent", "#FFFFFE")
      # Creating a container
      self.container = Frame(self)  
      self.container.grid(row=0, column=0, sticky="nsew")

      self.container.grid_rowconfigure(0, weight = 1)
      self.container.grid_columnconfigure(0, weight = 1)

      # Initializing frames to an empty array
      self.frames = {}

      # Iterating through a tuple consisting
      # Of the different page layouts
      for F in (StartPage, GamePlay):

          self.frame = F(self.container, self)

          # initializing frame of that object from
          # startpage, page1, page2 respectively with 
          # for loop
          self.frames[F] = self.frame 

          self.frame.grid(row = 0, column = 0, sticky ="nsew")

      self.show_frame(StartPage)

  # To display the current frame passed as
  # Parameter
  def show_frame(self, cont):
      if cont == GamePlay:
        self.frame = GamePlay(self.container, self)
        self.frames[GamePlay] = self.frame
        self.frame.grid(row = 0, column = 0, sticky ="nsew")
      self.frame = self.frames[cont]
      self.frame.tkraise()

class StartPage(Frame, metaclass=SingletonMeta):
  def __init__(self, parent, controller): 
      Frame.__init__(self, parent, width=1050, height=649)
      self.previous_button = None
        
      self.bg = PhotoImage(file = "Assets/bg.png") 

      # Show image using label 
      label1 = Label( self, image = self.bg) 
      label1.place(x = 0, y = 0) 

      # Title label
      label = Label(self, text ="Ares's Adventure", font=("Helvetica", 40, "bold"), fg="white", bg="#222a5c")
      label.place(relx=0.5, y= 100, anchor=CENTER)

      # Level Label
      self.levelLabel = Label(self, text = chosenLevel, font=("Helvetica", 60, "bold"), fg="white", bg="#222a5c")
      self.levelLabel.place(relx=0.5, y= 300, anchor=CENTER)

      # Level Increase Button
      self.levelIncreaseImage = PhotoImage(file= "Assets/rightArr.png") 
      self.levelIncreaseButton = Button(self, bg="#ffffff", image=self.levelIncreaseImage, padx=0, pady=0, command= self.level_increase)
      self.levelIncreaseButton.place(x = 600, y = 285)

      # Level Decrease Button
      self.levelDecreaseeImage = PhotoImage(file= "Assets/leftArr.png") 
      self.levelDecreaseeButton = Button(self, bg="#ffffff", image=self.levelDecreaseeImage, padx=0, pady=0, command= self.level_decrease)
      self.levelDecreaseeButton.place(x = 415, y = 285)

      # Start button
      self.startButton = Button(self, text="Start", fg="white", bg="#222a5c", padx=50, pady=1, command = lambda: self.start_btn_fn(controller))
      self.startButton.configure(font=("Helvetica", 18, "bold"))
      self.startButton.place(relx=0.5, y= 601, anchor=CENTER)


      # BFS button
      self.bfsButton = Button(self, text="BFS", fg="white", bg="#222a5c", padx=50, pady=1, command= lambda: [self.change_color(self.bfsButton), self.choose_algo(1)])
      self.bfsButton.configure(font=("Helvetica", 12, "bold"))
      self.bfsButton.place(x = 700, y = 501)

      # DFS button
      self.dfsButton = Button(self, text="DFS", fg="white", bg="#222a5c", padx=50, pady=1, command= lambda: [self.change_color(self.dfsButton), self.choose_algo(2)])
      self.dfsButton.configure(font=("Helvetica", 12, "bold"))
      self.dfsButton.place(x = 538, y = 501)

      # UCS button
      self.ucsButton = Button(self, text="UCS", fg="white", bg="#222a5c", padx=50, pady=1,  command= lambda: [self.change_color(self.ucsButton), self.choose_algo(3)])
      self.ucsButton.configure(font=("Helvetica", 12, "bold"))
      self.ucsButton.place(x = 376, y = 501)

      # A* button
      self.astarButton = Button(self, text="A*", fg="white", bg="#222a5c", padx=58, pady=1,  command= lambda: [self.change_color(self.astarButton), self.choose_algo(4)])
      self.astarButton.configure(font=("Helvetica", 12, "bold"))
      self.astarButton.place(x = 215, y = 501)

      # Alert
      self.alert = Label(self, text="Please choose an algorithm!", fg="red", bg="#222a5c")
      self.alert.config(font=("Helvetica", 12, "bold"))

  def start_btn_fn(self, controller):
    if self.previous_button != None:
      controller.show_frame(GamePlay)
      gameplay1 = GamePlay.get_instance()
      gameplay1.Action('Testcase/' + filenames[chosenLevel-1])
    else:
      self.alert.place(x = 415, y = 545)
  
  def change_color(self, clicked_button):
    #Change color of previous clicked button to default color
    if self.previous_button and self.previous_button != clicked_button:
      self.previous_button.config(bg="#222a5c")
      clicked_button.config(bg="#38002c")
      self.previous_button = clicked_button
    else:
      clicked_button.config(bg="#38002c")
      self.previous_button = clicked_button
    
    # Alert if no algorithm is selected by destroying the alert
    self.alert.destroy()
  
  def level_increase(self):
    global chosenLevel
    global numberOfFiles
    if chosenLevel < numberOfFiles:
      chosenLevel += 1
      self.levelLabel.config(text=chosenLevel)

  def level_decrease(self):
    global chosenLevel
    if chosenLevel > 1:
      chosenLevel -= 1
      self.levelLabel.config(text=chosenLevel)
  
  def choose_algo(self, algo):
    global chosenAlgo
    chosenAlgo = algo
      
class GamePlay(Frame, metaclass=SingletonMeta):   
  map_frame = None
  player_canvas = None
  map_data = []  # Your map data goes here
  list_rocks_weight = []  # Your weights data goes here
  cweight = 0
  player_position = []
  # Create a dict to store all canvas created
  map_objects = {} 
  # Create some objects to parse in BFS
  walls = [] # A list positions of walls
  goals = [] # A list positions of goals
  boxes = [] # A list positions of boxes
  action = None

  def __init__(self, parent, controller):
    Frame.__init__(self, parent, width=1050, height=649 )
    self.bg = PhotoImage(file = "Assets/bg.png") 

    # Show image using label 
    label1 = Label( self, image = self.bg) 
    label1.place(x = 0, y = 0) 

    self.proccess_label = Label(self, text = "Processing...", font=("Helvetica", 20, "bold"), fg="white", bg="#222a5c")
    self.proccess_label.place(relx=0.5, rely=0.2, anchor="center")

    # Counter steps:
    self.step_counter = 0
    self.counterStep = Label(self, text=f"Step: {self.step_counter}", fg="white", bg="#222a5c", padx=10, pady=10, width=5, height=1)
    self.counterStep.config(font=("Helvetica", 12, "bold"))
    self.counterStep.place(x = 20, y = 20)

    self.weight_counter = 0
    self.counterWeight = Label(self, text=f"Weight: {self.weight_counter}", fg="white", bg="#222a5c", padx=10, pady=10, width=7, height=1)
    self.counterWeight.config(font=("Helvetica", 12, "bold"))
    self.counterWeight.place(x = 120, y = 20)

    # Pause button
    self.isPause = True
    self.pauseImage = PhotoImage(file = "Assets/pause.png") 
    self.startImage = PhotoImage(file = "Assets/start.png")  
    self.pauseButton = Button(self, bg="#222a5c", image=self.startImage, padx=0, pady=0, command=self.change_pause_btn)
    self.pauseButton.place(x = 982, y = 20)

    # Exit button
    self.exitImage = PhotoImage(file = "Assets/exit.png")  
    self.exitButton = Button(self, bg="#222a5c", image=self.exitImage, padx=0, pady=0, command = lambda : self.exit(controller))
    self.exitButton.place(x = 982, y = 82)

    # Restart button
    self.restartImage = PhotoImage(file = "Assets/restart.png") 
    self.restartButton = Button(self, bg="#222a5c", image=self.restartImage, padx=0, pady=0, command= lambda: self.restart(filename='Testcase/' + filenames[chosenLevel-1]))
    self.restartButton.place(x = 982, y = 144)
  
  def exit_component(self):
    """
    This function is used both for exit method and close window method
    """
    try:
      # Stop the algorithm and Delete the action
      if hasattr(self.action, 'manager'):
        self.action.manager.stop()

      if(self.isPause == False): self.change_pause_btn()
      self.action.exit()
      self.check_thread() # Wait for the thread

      del self.action
      self.action = None

    except Exception as e:
      print(f"An error occurred while reseting: {e}")

  def exit(self, controller):
      """
      This function is used for exit method
      """
      self.exit_component()

      # Reset the gameplay
      self.reset_state()

      # Change the frame
      controller.show_frame(StartPage)


  def restart(self, filename):
    """
    This function is used for restart method
    """
    self.exit_component()

    # Reset the gameplay
    self.reset_state()

    # Call the action
    self.Action(filename)


  def check_thread(self):
    """
    This function is used to wait for the thread to finish\n
    The deadlock is the main thread waiting thread1\n
    The thread1 tries to update GUI so it waits main thread\n
    The solution is to check the thread's status periodically and stop waiting if the thread is finished.
    """
    if self.thread1.is_alive():
      self.after(100, self.check_thread)
    else:
      if(self.action):
        if(self.action.done_event):
          self.action.done_event.set()
      self.thread1.join()

  def change_pause_btn(self):
    if self.isPause:
      self.pauseButton.config(image=self.pauseImage)
      self.isPause = False
    else:
      self.pauseButton.config(image=self.startImage)
      self.isPause = True

    if(self.action is not None):
      self.action.pause(self.isPause)

  def create_player_on_canvas(self, row_index, column_index):
      # Create Canvas for player
      self.player_canvas = Canvas(self.map_frame, bg="#222a5c", highlightthickness=0, borderwidth=0, width=32, height=32)
      self.player_canvas.grid(row=row_index, column=column_index)
      self.player_canvas.create_image(16, 16, anchor=CENTER, image=self.player_image, tags="player")

      self.map_objects[f"{row_index}_{column_index}"] = self.player_canvas
      self.player_position = [row_index, column_index]

  def create_box_on_canvas(self, row_index, column_index):
      # Create Canvas for box
      box_canvas = Canvas(self.map_frame, bg="#222a5c", highlightbackground="black", borderwidth=0, width=29, height=29)
      box_canvas.grid(row=row_index, column=column_index)

      box_canvas.create_image(16, 16, image=self.box_image, anchor=CENTER)
      box_canvas.create_image(16, 16, image=self.box_bg, anchor=CENTER)
      box_canvas.create_text(16, 16, text=self.list_rocks_weight[self.cweight], font=("Helvetica", 10, "bold"), tags="weight")
      self.cweight += 1
      self.map_objects[f"{row_index}_{column_index}_box"] = box_canvas  # Store the box reference
      self.boxes.append([row_index, column_index])

  def create_goal_on_canvas(self, row_index, column_index):
      # Create Canvas for goal
      goal_canvas = Canvas(self.map_frame, bg="#222a5c", highlightthickness=0, borderwidth=0, width=32, height=32)
      goal_canvas.grid(row=row_index, column=column_index)
      goal_canvas.create_image(16, 16, image=self.goal_image, anchor=CENTER)

      if(self.map_data[row_index][column_index] == "+"):
        goal_canvas.create_image(16, 16, image=self.player_image, anchor=CENTER, tags="player")
        self.player_position = [row_index,column_index]

      self.map_objects[f"{row_index}_{column_index}"] = goal_canvas
      self.goals.append([row_index, column_index])

  def Action(self, filename):
    """
    First, load all necessary file\n
    Second, perform actions\n
    """
    # Initalize all the data and map
    self.load_file(filename)
    self.draw_map()
    self.action = Actions(self)
    print(self.action)
    if chosenAlgo != 0:
      self.thread1 = threading.Thread(target=self.action.run_algorithm, args=[chosenAlgo,])
      self.thread1.start()
    else:
        print("No algorithm selected")

    print("done at GamePlay")

  def draw_map(self):
    """
    This function is called to draw the map
    """
    # Create a frame to hold the canvas
    self.map_frame = Frame(self, bg="#222a5c")
    self.map_frame.place(relx=0.5, rely=0.5, anchor=CENTER)

    # Load images
    self.wall_image = PhotoImage(file="Assets/wall.png")
    self.goal_image = PhotoImage(file="Assets/goal.png")
    self.player_image = PhotoImage(file="Assets/player.png")
    self.box_image = PhotoImage(file="Assets/box.png")
    self.box_bg = PhotoImage(file="Assets/bgweight.png")

    # Input the map based on the map data
    for row_index, row in enumerate(self.map_data):
      for column_index, item in enumerate(row):
        if item == "#":
          # Wall
          self.wall_canvas = Canvas(self.map_frame, bg="black", highlightthickness=0, borderwidth=0, width=32, height=32)
          self.wall_canvas.grid(row=row_index, column=column_index)
          self.wall_canvas.create_image(16, 16, anchor=CENTER, image=self.wall_image)
          self.map_objects[f"{row_index}_{column_index}"] = self.wall_canvas
          self.walls.append([row_index, column_index])

        elif item == ".":
          # Goal
          self.create_goal_on_canvas(row_index, column_index)

        elif item == "@":
          # Player
          self.create_player_on_canvas(row_index, column_index)

        elif item == "$":
          # Box
          self.create_box_on_canvas(row_index, column_index)

        elif item == "+":
          # Player on goal
          self.create_goal_on_canvas(row_index, column_index)

        elif item == "*":
          # Box on goal
          self.create_goal_on_canvas(row_index, column_index)
          self.create_box_on_canvas(row_index, column_index)
          self.map_objects.get(f"{row_index}_{column_index}_box").config(highlightbackground="yellow")

  def load_file(self, filename):
    """
    This function is called to load the data from text file
    """
    # Read the first line of the file and parse in a list
    try:
      with open(filename, "r") as file_map:
        file_map = open(filename, "r")

        each_rock = file_map.readline().split()
        self.list_rocks_weight = [int(weight) for weight in each_rock]

        # Read the whole file for a map
        for line in file_map:
          self.map_data.append(list(line.rstrip("\n")))

    # Catch exceptions and print
    except FileNotFoundError:
      print(f"File {filename} not found!")
    except Exception as e:
      print(f"An error occurred: {str(e)}")

  def reset_state(self):
    """
    This function is used to reset the state of the gameplay\n
    Very important before loading a new stage
    """

    # Reinitialize all the variables
    self.map_frame.destroy()
    self.map_frame = None
    self.map_data.clear()  # Your map data goes here
    self.list_rocks_weight.clear()  # Your weights data goes here
    self.cweight = 0
    self.player_position.clear()

    # Reset walls, goals and boxes
    self.walls.clear()
    self.goals.clear()
    self.boxes.clear()
    self.map_objects.clear()

    # Reset counter state
    self.step_counter = 0
    self.weight_counter = 0
    self.counterStep.config(text=f"Step: {self.step_counter}")
    self.counterWeight.config(text=f"Weight: {self.weight_counter}")

  def show_popup(self, message, color):
    self.proccess_label.destroy()
    label = Label(self, text =message, font=("Helvetica", 20, "bold"), fg=color, bg="#222a5c")
    label.place(relx=0.5, rely=0.2, anchor="center")
    self.after(8000, label.destroy)
  
  @classmethod
  def get_instance(cls):
    if cls._instances is None:
      cls._instances = cls()  # Create a new instance if it doesn't exist
    return cls._instances[cls]
  
class Actions():
  """
  This class provides actions for character and display valid movements on the map.\n
  The class has 4 methods: up, down, right, left\n
  Test: Using arrow keys to check if works
  """
  def __init__(self, game_play):
    """
    game_play is the frame of the map currently in
    """
    self.game_play = game_play
    self.data = game_play.map_data #Brings the data to the class
    self.done_event = None
    self.isPause = True
    self.isExit = False

  def up(self):
    """
    Moves the player up
    """
    
    current_player_row, current_player_column = self.game_play.player_position

    if current_player_row < 0 or self.data[current_player_row - 1][current_player_column]== "#": return #Check the wall

    # This function is used to move the player
    self.condition_case_player_box_goal(1, 2, 0, 0, current_player_row, current_player_column)

  def down(self):
    """
    Moves the player down
    """
    
    current_player_row, current_player_column = self.game_play.player_position

    if current_player_row < 0 or self.data[current_player_row + 1][current_player_column]== "#": return #Check the wall

    # This function is used to move the player
    self.condition_case_player_box_goal(-1, -2, 0, 0, current_player_row, current_player_column)

  def left(self):
    """
    Moves the player left
    """
    
    current_player_row, current_player_column = self.game_play.player_position

    if current_player_row < 0 or self.data[current_player_row][current_player_column - 1]== "#": return #Check the wall

    # This function is used to move the player
    self.condition_case_player_box_goal(0, 0, -1, -2, current_player_row, current_player_column)

  def right(self):
    """
    Moves the player right
    """
    
    current_player_row, current_player_column = self.game_play.player_position

    if current_player_row < 0 or self.data[current_player_row][current_player_column + 1]== "#": return #Check the wall

    # This function is used to move the player
    self.condition_case_player_box_goal(0, 0, 1, 2, current_player_row, current_player_column)

  def move_player(self, one_move_UD, one_move_LR, row, column, weight):
    #Init
    new_row = row - one_move_UD
    new_column = column + one_move_LR
    player_canvas = self.game_play.player_canvas

    # From current, we will remove or forget the player_canvas
    if(self.data[row][column] == "+"):
      # The player is on the goal
      self.data[row][column] = "."
      key = f"{row}_{column}"
      canva = self.game_play.map_objects.get(key)
      players = canva.find_withtag("player")

      for player in players:
        canva.delete(player)

    else: 
      self.data[row][column] = " "

    # The new spot, we will check if it is a goal or not
    if(self.data[new_row][new_column] in [".", "*"]):
      # The new spot is a goal
      self.data[new_row][new_column] = "+"
      self.game_play.player_position = [new_row, new_column] # Update the position of the player

      # Check if move from a goal to a goal
      # If yes, then there will be no player_canvas
      if(player_canvas is None):
        print("No player_canvas")
        goal_canvas = self.game_play.map_objects.get(f"{new_row}_{new_column}") # Return the address of the new goal canvas
        goal_canvas.create_image(16, 16, image=self.game_play.player_image, anchor=CENTER, tags="player") # Create image of player
        return

      # The player landed on the goal
      goal_canvas = self.game_play.map_objects.get(f"{new_row}_{new_column}") # Return the address of the new goal canvas
      goal_canvas.create_image(16, 16, image=self.game_play.player_image, anchor=CENTER, tags="player") # Create image of player

      # Destroy the old spot
      self.game_play.player_canvas.destroy()
      self.game_play.player_canvas = None
      return

    # The player landed on a space
    self.data[new_row][new_column] = "@"
    # Check if the player is from a goal to a space
    if(player_canvas is None or not player_canvas.winfo_exists()):
      # Create a new canvas for the player
      self.game_play.player_canvas = Canvas(self.game_play.map_frame, bg="#222a5c", highlightthickness=0, borderwidth=0, width=32, height=32)
      self.game_play.player_canvas.create_image(16, 16, image=self.game_play.player_image, anchor=CENTER, tags="player")
    else: 
      self.game_play.player_canvas.grid_remove()

    self.game_play.player_canvas.grid(row= new_row, column= new_column)
    
    # Record the new canvas to the map_objects and player position
    self.game_play.map_objects[f"{new_row}_{new_column}"] = self.game_play.player_canvas
    self.game_play.player_position = [new_row, new_column]
    # Add counter
    self.add_step(weight)

  def condition_case_player_box_goal(self, one_move_UD, two_move_UD, one_move_LR, two_move_LR , row, column):
    # Init
    new_player_row = row - one_move_UD
    new_player_column = column + one_move_LR
    new_box_row = row - two_move_UD
    new_box_column = column + two_move_LR
    weight = 0

    # If in front of player is space, move player
    if (self.data[new_player_row][new_player_column] in [" ", "."]): return self.move_player(one_move_UD, one_move_LR, row, column, weight)

    # In front the player is box
    # Check if in front of the player is another box or wall
    if self.data[new_box_row][new_box_column] in ["#", "$", "*"]: return

    # Move the box
    if self.data[new_player_row][new_player_column] not in ["$", "*"]: return # This function check data, so move player must follow after

    canva = self.game_play.map_objects[f"{new_box_row}_{new_box_column}_box"] = self.game_play.map_objects.pop(f"{new_player_row}_{new_player_column}_box")
    canva.grid_remove()
    if(self.data[new_player_row][new_player_column] == "*"):
      self.data[new_player_row][new_player_column] = "."
      self.game_play.map_objects[f"{new_player_row}_{new_player_column}"].grid()
    else: self.data[new_player_row][new_player_column] = " "

    if(self.data[new_box_row][new_box_column] == "."):
      self.data[new_box_row][new_box_column] = "*"
      canva.config(highlightbackground= "yellow")
      self.game_play.map_objects[f"{new_box_row}_{new_box_column}"].grid_remove()
    else: 
      self.data[new_box_row][new_box_column] = "$"
      canva.config(highlightbackground= "black")
    canva.grid(row= new_box_row, column= new_box_column)

    for item in canva.find_withtag("weight"):
      weight = canva.itemcget(item, "text")
      break
    # Move the player
    self.move_player(one_move_UD, one_move_LR, row, column, weight)
    
  # Add steps to the counter
  def add_step(self, weight):
    # Each move increases step and weight
    self.game_play.weight_counter += int(weight) + 1
    self.game_play.step_counter += 1
    self.game_play.counterStep.config(text=f"Steps: {self.game_play.step_counter}")
    self.game_play.counterWeight.config(text=f"Weight: {self.game_play.weight_counter}")

  def move_character_with_instructions(self, character):
    if character == "u":
      self.up()
    elif character == "d":
      self.down()
    elif character == "l":
      self.left()
    elif character == "r":
      self.right()

  def run_algorithm(self, chosenAlgo):
    self.done_event = threading.Event()
    self.done_event.clear()

    # Bring the data to the class
    gameplay = self.game_play
    self.data = gameplay.map_data

    # Allocate values to variables for BFS
    player_position = gameplay.player_position
    boxes = gameplay.boxes
    walls = gameplay.walls
    goals = gameplay.goals
    stone_weights = gameplay.list_rocks_weight  # Assuming stone weights are available
    
    # Start measuring memory and time
    tracemalloc.start()
    start_time = time.time()
    
    # Initialize the data for BFS
    data = a.Initialized_data(walls, goals, stone_weights)

    # Initialize a manager
    self.manager = a.Manager_Algorithm(data=data)
    if chosenAlgo == 1:
        algo_name = "BFS"
        goal_state, node_counter = self.manager.run_bfs(player_position, boxes)
    elif chosenAlgo == 2:
        algo_name = "DFS"
        goal_state, node_counter = self.manager.run_dfs(player_position, boxes)
    elif chosenAlgo == 3:
        algo_name = "UCS"
        goal_state, node_counter = self.manager.run_ucs(player_position, boxes)
    elif chosenAlgo == 4:
        algo_name = "A*"
        goal_state, node_counter = self.manager.run_astar(player_position, boxes)

    # Stop measuring time and memory after BFS completes
    end_time = time.time()
    elapsed_time = (end_time - start_time)*1000

    # Get the current, peak memory usage
    current, peak = tracemalloc.get_traced_memory()
    current = current >> 20 # To MB
    tracemalloc.stop()

    # Move the player, init a new variable to make sure that string move is unmodified
    if goal_state is not None:
      gameplay.show_popup("Solution found!","#00ff1e")
      # Get the node generated
      string_move = goal_state.string_move.lower()
      length = len(string_move)
      i = 0
      while i < length: 
        if (self.isExit): break

        time.sleep(0.5)
        if (self.isPause): continue

        self.move_character_with_instructions(string_move[i])
        i+=1

      # Other resources
      if (not self.isExit):
        self.write_to_file(f"{algo_name}\n"  +
        f"Steps: {self.game_play.step_counter}, Weights: {self.game_play.weight_counter}," + 
        f" Node: {node_counter}, Time (ms): {elapsed_time}, Memory (MB): {current:.2f}\n" +
        goal_state.string_move +"\n")
    else: 
      gameplay.show_popup("No solution!","#f20707")

    del data # Remove the reference after running
    print("Stop successfully")
    self.done_event.set()

  def exit(self):
    self.isExit = True

  def pause(self, isPause):
    self.isPause = isPause

# A function to write to the file the string of movement
  def write_to_file(self, content):
    with open(f"output-{chosenLevel:02}.txt", 'a') as file:
      file.write(content)

# Driver Code
def application():
  """
  Runs this function to run the program
  """
  app = tkinterApp()

  app.mainloop()
  # if exit in gameplay, app.frame is gameplay
  if(app.frame == GamePlay.get_instance()):
    app.frame.exit_component()

