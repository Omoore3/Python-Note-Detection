from tkinter import *
from tkinter import ttk

root = Tk()
root.title("Music Transcriber")

mainframe = ttk.Frame(root, padding="10 10 10 10")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

ttk.Label(mainframe, text = "Insert filepath to audio file to transcribe...").grid(column=0, row=0)

fileBox = StringVar()
fileEntry = ttk.Entry(mainframe, width=30, textvariable=fileBox)
fileEntry.grid(column=0, row=1)


ttk.Button(mainframe, text="Transcribe").grid(column=1, row=2, sticky=E)


root.mainloop()