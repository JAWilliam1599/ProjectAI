import tkinter as tk

class Application():
    def __init__(self, windowName):
        super().__init__()
        self.__win = tk.Tk()
        self.__win.title(windowName)
        self.__win.geometry("1154x649")
        self.__win.resizable(0,0)

    # Create a frame 

    # Run the application
    def run(self):
        self.__win.mainloop()

def application():
    app = Application("Application")
    app.run()
