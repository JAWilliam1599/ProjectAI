# Import module  
from tkinter import *
  
"""
Note:
  - The image file is inside assets.
  - #222a5c is the blue background
  - #38002c is the red background
"""

# Create object  
root = Tk() 
  
# Adjust size  
root.title("Ares's adventure")
root.geometry("1050x649") 
root.resizable(0,0)
  
# Add image file 
bg = PhotoImage(file = "Assets/bg.png") 
  
# Show image using label 
label1 = Label( root, image = bg) 
label1.place(x = 0, y = 0) 

# Counter steps:
counterStep = Label(root, text="Step: 0", fg="white", bg="#222a5c", padx=10, pady=10, width=5, height=1)
counterStep.config(font=("Helvetica", 12, "bold"))
counterStep.place(x = 20, y = 20)

#pause button
pauseImage = PhotoImage(file = "Assets/pause.png") 
pauseButton = Button(root, bg="#222a5c", image=pauseImage, padx=0, pady=0)
pauseButton.place(x = 982, y = 20)

# Start button
startImage = PhotoImage(file = "Assets/start.png") 
startButton = Button(root, bg="#222a5c", image=startImage, padx=0, pady=0)
startButton.place(x = 982, y = 82)

# Restart button
restartImage = PhotoImage(file = "Assets/restart.png") 
restartButton = Button(root, bg="#222a5c", image=restartImage, padx=0, pady=0)
restartButton.place(x = 982, y = 144)

# BFS button
bfsButton = Button(root, text="BFS", fg="white", bg="#38002c", padx=50, pady=1)
bfsButton.configure(font=("Helvetica", 12, "bold"))
bfsButton.place(x = 890, y = 601)

# DFS button
dfsButton = Button(root, text="DFS", fg="white", bg="#222a5c", padx=50, pady=1)
dfsButton.configure(font=("Helvetica", 12, "bold"))
dfsButton.place(x = 728, y = 601)

# UCS button
ucsButton = Button(root, text="UCS", fg="white", bg="#222a5c", padx=50, pady=1)
ucsButton.configure(font=("Helvetica", 12, "bold"))
ucsButton.place(x = 566, y = 601)

# A* button
astarButton = Button(root, text="A*", fg="white", bg="#222a5c", padx=58, pady=1)
astarButton.configure(font=("Helvetica", 12, "bold"))
astarButton.place(x = 405, y = 601)

#  button

# Create Frame 
frame1 = Frame(root) 
frame1.pack(pady = 20 ) 
  
# Execute tkinter 
root.mainloop()