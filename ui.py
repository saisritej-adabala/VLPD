from tkinter import *        
from tkinter import messagebox
from tkinter.ttk import *
import pymongo
import tkinter.font as font
#myFont = font.Font(family='Helvetica', size=20, weight='bold')
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["license"]
mycol = mydb["vehicalsdata"]
def search():
   plate=name.get().upper()
   s=""
   for x in mycol.find({"Numberplate":plate}, { "_id": 0}):
    s+="NumberPlate : "+x['Numberplate']+"\n"
    s+="Time        : "+x['Rtime']+"\n"
    s+="Location    : "+x['Location']+"\n\n\n"
   messagebox.showinfo('Vehicles Found',s)
root = Tk()   
root.geometry('600x600')     
lbl = Label(root, text = "").place(x = 50, y = 70)
btn = Button(root, text = 'Search',command = search)
#btn['font'] = myFont
btn.pack()
name = StringVar()
entry1 = Entry(root,textvariable = name).place(relx = 0.5, rely = 0.4, anchor = CENTER,width=300,height=50)
btn.place(relx = 0.5, rely = 0.5, anchor = CENTER,width=100,height=40)   
root.mainloop()
