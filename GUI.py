import tkinter
from tkinter import *
from tkinter import messagebox
import tkinter.ttk as ttk
from Main import train_data, predict


def call():
    res = messagebox.askquestion('exit',
                                 'Do you really want to exit?')
    if res == 'yes':
        window.destroy()


def training():
    accuracy = train_data()
    canvas2 = tkinter.Canvas(window, bg="#d9d9d9", height=90, width=380)
    canvas2.place(x=110, y=400)
    cvstext = canvas2.create_text(100, 40, text='', font=(
        'Helvetica 12 bold'), anchor=tkinter.NW)
    acc = "Accuracy = " + str(accuracy)
    canvas2.itemconfigure(cvstext, text=acc)


def predicting():
    string = entry0.get()
    res = predict(string)
    canvas2 = tkinter.Canvas(window, bg="#d9d9d9", height=90, width=380)
    canvas2.place(x=110, y=400)
    cvstext = canvas2.create_text(100, 40, text='', font=(
        'Helvetica 12 bold'), anchor=tkinter.NW)
    canvas2.itemconfigure(cvstext, text=res)


window = Tk()

window.geometry("1000x600")
window.title('Analisis Sentimen')
window.configure(bg="#ffffff")


canvas = Canvas(
    window,
    bg="#ffffff",
    height=600,
    width=1000,
    bd=0,
    highlightthickness=0,
    relief="ridge")
canvas.place(x=0, y=0)

background_img = PhotoImage(file=f"assets/background.png")
background = canvas.create_image(
    522.0, 300.0,
    image=background_img)

entry0_img = PhotoImage(file=f"assets/img_textBox0.png")
entry0_bg = canvas.create_image(
    344.5, 238.0,
    image=entry0_img)

entry0 = Entry(
    bd=0,
    bg="#d9d9d9",
    highlightthickness=0)

entry0.place(
    x=112.0, y=176,
    width=465.0,
    height=122)
'''
entry1_img = PhotoImage(file=f"img_textBox1.png")
entry1_bg = canvas.create_image(
    245.5, 443.0,
    image=entry1_img)

entry1 = Entry(
    bd=0,
    bg="#d9d9d9",
    highlightthickness=0)

entry1.place(
    x=110.0, y=400,
    width=271.0,
    height=84)
'''
PrButton = PhotoImage(file=f"assets/predict.png")
b0 = Button(
    image=PrButton,
    borderwidth=0,
    highlightthickness=0,
    command=predicting,
    relief="flat")

b0.place(
    x=253, y=316,
    width=143,
    height=44)

TrButton = PhotoImage(file=f"assets/Training.png")
b1 = Button(
    image=TrButton,
    borderwidth=0,
    highlightthickness=0,
    command=training,
    relief="flat")

b1.place(
    x=102, y=316,
    width=136,
    height=44)

canvas = tkinter.Canvas(window, bg="#d9d9d9", height=90, width=380)
canvas.place(x=110, y=400)

exitButton = PhotoImage(file=f"assets/Exit.png")
b2 = Button(
    image=exitButton,
    borderwidth=0,
    highlightthickness=0,
    command=call,
    relief="flat")

b2.place(
    x=106, y=502,
    width=120,
    height=39)


tombolExit = ttk.Button(
    window, image=exitButton, command=call)

window.resizable(False, False)
window.mainloop()
