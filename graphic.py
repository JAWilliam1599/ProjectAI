# Import module  
from tkinter import *
  
"""
Note:
  - The image file is inside assets.
  - #222a5c is the blue background
  - #38002c is the red background of the button
"""

class tkinterApp(Tk):
  # __init__ function for class tkinterApp 
  def __init__(self, *args, **kwargs): 

      # __init__ function for class Tk
      Tk.__init__(self, *args, **kwargs)
      self.title("Ares's Adventure")
      self.geometry("1050x649")
      self.resizable(False, False)
      # Creating a container
      container = Frame(self)  
      container.pack(side = "top", fill = "both", expand = True) 

      container.grid_rowconfigure(0, weight = 1)
      container.grid_columnconfigure(0, weight = 1)

      # Initializing frames to an empty array
      self.frames = {}  

      # Iterating through a tuple consisting
      # Of the different page layouts
      for F in (StartPage, GamePlay):

          frame = F(container, self)

          # initializing frame of that object from
          # startpage, page1, page2 respectively with 
          # for loop
          self.frames[F] = frame 

          frame.grid(row = 0, column = 0, sticky ="nsew")

      self.show_frame(StartPage)

  # To display the current frame passed as
  # Parameter
  def show_frame(self, cont):
      frame = self.frames[cont]
      frame.tkraise()

class StartPage(Frame):
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

      # Start button
      self.startButton = Button(self, text="Start", fg="white", bg="#222a5c", padx=50, pady=1, command = lambda: self.start_btn_fn(controller))
      self.startButton.configure(font=("Helvetica", 18, "bold"))
      self.startButton.place(relx=0.5, y= 601, anchor=CENTER)

      
      # BFS button
      self.bfsButton = Button(self, text="BFS", fg="white", bg="#222a5c", padx=50, pady=1, command= lambda: self.change_color(self.bfsButton))
      self.bfsButton.configure(font=("Helvetica", 12, "bold"))
      self.bfsButton.place(x = 705, y = 501)

      # DFS button
      self.dfsButton = Button(self, text="DFS", fg="white", bg="#222a5c", padx=50, pady=1, command= lambda: self.change_color(self.dfsButton))
      self.dfsButton.configure(font=("Helvetica", 12, "bold"))
      self.dfsButton.place(x = 543, y = 501)

      # UCS button
      self.ucsButton = Button(self, text="UCS", fg="white", bg="#222a5c", padx=50, pady=1,  command= lambda: self.change_color(self.ucsButton))
      self.ucsButton.configure(font=("Helvetica", 12, "bold"))
      self.ucsButton.place(x = 381, y = 501)

      # A* button
      self.astarButton = Button(self, text="A*", fg="white", bg="#222a5c", padx=58, pady=1,  command= lambda: self.change_color(self.astarButton))
      self.astarButton.configure(font=("Helvetica", 12, "bold"))
      self.astarButton.place(x = 220, y = 501)

      # Alert
      self.alert = Label(self, text="Please choose an algorithm!", fg="red", bg="#222a5c")
      self.alert.config(font=("Helvetica", 12, "bold"))

  def start_btn_fn(self, controller):
    if self.previous_button != None:
      controller.show_frame(GamePlay)
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
      
class GamePlay(Frame):     
  def __init__(self, parent, controller):
    Frame.__init__(self, parent, width=1050, height=649 )
    self.bg = PhotoImage(file = "Assets/bg.png") 

    # Show image using label 
    label1 = Label( self, image = self.bg) 
    label1.place(x = 0, y = 0) 

    # Counter steps:
    self.counterStep = Label(self, text="Step: 0", fg="white", bg="#222a5c", padx=10, pady=10, width=5, height=1)
    self.counterStep.config(font=("Helvetica", 12, "bold"))
    self.counterStep.place(x = 20, y = 20)

    # Pause button
    self.isPause = True
    self.pauseImage = PhotoImage(file = "Assets/pause.png") 
    self.startImage = PhotoImage(file = "Assets/start.png")  
    self.pauseButton = Button(self, bg="#222a5c", image=self.startImage, padx=0, pady=0, command=self.change_pause_btn)
    self.pauseButton.place(x = 982, y = 20)


    # Exit button
    self.exitImage = PhotoImage(file = "Assets/exit.png")  
    self.exitButton = Button(self, bg="#222a5c", image=self.exitImage, padx=0, pady=0, command = lambda : controller.show_frame(StartPage))
    self.exitButton.place(x = 982, y = 82)

    # Restart button
    self.restartImage = PhotoImage(file = "Assets/restart.png") 
    self.restartButton = Button(self, bg="#222a5c", image=self.restartImage, padx=0, pady=0)
    self.restartButton.place(x = 982, y = 144)

    # Load test file
    self.load_file("Testcase/input-01.txt")
    self.draw_map()
  
  def change_pause_btn(self):
    if self.isPause:
      self.pauseButton.config(image=self.pauseImage)
      self.isPause = False
    else:
      self.pauseButton.config(image=self.startImage)
      self.isPause = True

  #This function is called to load the map
  def load_file(self, filename):
    #Initialize
    self.map_data = []
    self.list_rocks_weight = []
    self.cweight = 0 #Weight counter for list of rocks' weight

    # Read the first line of the file and parse in a list
    try:
      with open(filename, "r") as file_map:
        file_map = open(filename, "r")

        each_rock = file_map.readline().split()
        self.list_rocks_weight.extend(each_rock)

        # Read the whole file for a map
        for line in file_map:
          self.map_data.append(list(line.rstrip("\n")))

    # Catch exceptions and print
    except FileNotFoundError:
      print(f"File {filename} not found!")
    except Exception as e:
      print(f"An error occurred: {str(e)}")

  # This function is called to draw the map
  def draw_map(self):
    #Create a frame in the middle of the screen
    self.map_frame = Frame(self, bg="#222a5c")
    self.map_frame.place(relx=0.5, rely=0.5 ,anchor=CENTER)

    # Input the map based on the map data
    self.wall_image = PhotoImage(file="Assets/wall.png") #Keep the original wall image
    self.goal_image = PhotoImage(file="Assets/goal.png") #Keep the original goal image
    self.player_image = PhotoImage(file="Assets/player.png")
    self.box_image = PhotoImage(file="Assets/box.png")
    self.box_bg = PhotoImage(file="Assets/bgweight.png")

    for row_index, list in enumerate(self.map_data):
      for column_index, item in enumerate(list):
        if item == "#":
          # Wall
          label = Label(self.map_frame, image = self.wall_image, bg="black")
          label.grid(row=row_index, column=column_index)

        elif item == ".":
          # Goal
          label = Label(self.map_frame, image = self.goal_image, bg="#222a5c")
          label.grid(row=row_index, column=column_index)

        elif item == "@":
          # Player
          label = Label(self.map_frame, image = self.player_image, bg="#222a5c")
          label.grid(row=row_index, column=column_index)

        elif item == "$":
          # Box
          canva_box = Canvas(self.map_frame, bg="black", highlightbackground="black", borderwidth=0, width=29, height=29)
          canva_box.grid(row=row_index, column=column_index)
          canva_box.create_image(16, 16, image=self.box_image, anchor=CENTER)

          canva_box.create_image(16, 16, image=self.box_bg, anchor= CENTER)
          canva_box.create_text(16, 16, text=self.list_rocks_weight[self.cweight], font=("Helvetica", 10, "bold"))
          self.cweight += 1

        elif item == "+":
          # Player on switch
          canva_player_on_switch = Canvas(self.map_frame, bg = "#222a5c", highlightthickness=0, borderwidth=0, width=32, height=32)
          canva_player_on_switch.grid(row=row_index, column=column_index)

          canva_player_on_switch.create_image(16, 16, image=self.goal_image, anchor=CENTER)
          canva_player_on_switch.create_image(16, 16, image=self.player_image, anchor=CENTER)

        elif item == "*":
          # Box on switch...
          canva_box = Canvas(self.map_frame, bg="black", highlightbackground="yellow", borderwidth=0, width=29, height=29)
          canva_box.grid(row=row_index, column=column_index)
          canva_box.create_image(16, 16, image=self.box_image, anchor=CENTER)

          canva_box.create_image(16, 16, image=self.box_bg, anchor= CENTER)
          canva_box.create_text(16, 16, text=self.list_rocks_weight[self.cweight], font=("Helvetica", 10, "bold"))
          self.cweight += 1
        else: continue
        
# Driver Code
def application():
  app = tkinterApp()
  app.mainloop()

