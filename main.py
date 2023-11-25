import pickle
import os.path

import tkinter.messagebox
from tkinter import *
from tkinter import simpledialog, filedialog

import PIL
import PIL.Image, PIL.ImageDraw
import cv2 as cv
import numpy as np
from PIL import Image

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


class DrawingClassifier:

    def __init__(self):
        self.class1, self.class2, self.class3 = None, None, None
        self.class1_counter, self.class2_counter, self.class3_counter = None, None, None
        self.clf = None
        self.proj_name = None
        self.root = None
        self.image1 = None

        self.status_label = None
        self.canvas = None
        self.draw = None

        self.brush_width = 15

        self.classes_prompt()
        self.init_gui()

    def classes_prompt(self):
        msg = Tk()
        msg.withdraw()

        self.proj_name = simpledialog.askstring("Project Name", "Please enter your project name", parent=msg)
        if os.path.exists(self.proj_name):
            with open(f"{self.proj_name}/{self.proj_name}_data.pickle", "rb") as f:
                data = pickle.load(f)
            self.class1 = data['c1']
            self.class2 = data['c2']
            self.class3 = data['c3']
            self.class1_counter = data['c1c']
            self.class2_counter = data['c2c']
            self.class3_counter = data['c3c']
            self.clf = data['clf']
            self.proj_name = data['pname']
        else:
            self.class1 = simpledialog.askstring("Class 1", 'What is the first class called?', parent=msg)
            self.class2 = simpledialog.askstring("Class 2", 'What is the second class called?', parent=msg)
            self.class3 = simpledialog.askstring("Class 3", 'What is the third class called?', parent=msg)

            self.class1_counter = 1
            self.class2_counter = 1
            self.class3_counter = 1

            self.clf = LinearSVC(dual=False)
            os.mkdir(self.proj_name)
            os.chdir(self.proj_name)
            os.mkdir(self.class1)
            os.mkdir(self.class2)
            os.mkdir(self.class3)
            os.chdir("..")

    def init_gui(self):
        WIDTH = 500
        HEIGHT = 500
        WHITE = (255, 255, 255)

        self.root = Tk()
        self.root.title(f"Drawings Detection {self.proj_name}")

        self.canvas = Canvas(self.root)
        self.canvas.pack(expand=YES, fill=BOTH)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.image1 = PIL.Image.new("RGB", (WIDTH, HEIGHT), WHITE)
        self.draw = PIL.ImageDraw.Draw(self.image1)

        btn_frame = tkinter.Frame(self.root)
        btn_frame.pack(fill=X, side=BOTTOM)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        btn_frame.columnconfigure(2, weight=1)

        class1_btn = Button(btn_frame, text=self.class1, command=lambda: self.save(1))
        class1_btn.grid(row=0, column=0, sticky=W + E)

        class2_btn = Button(btn_frame, text=self.class2, command=lambda: self.save(2))
        class2_btn.grid(row=0, column=1, sticky=W + E)

        class3_btn = Button(btn_frame, text=self.class3, command=lambda: self.save(3))
        class3_btn.grid(row=0, column=2, sticky=W + E)

        clear_btn = Button(btn_frame, text='Clear', command=self.clear)
        clear_btn.grid(row=1, column=0, sticky=W + E)

        train_btn = Button(btn_frame, text='Train', command=self.train)
        train_btn.grid(row=1, column=1, sticky=W + E)


        change_btn = Button(btn_frame, text='Change', command=self.rotate_model)
        change_btn.grid(row=1, column=2, sticky=W + E)

        predict_btn = Button(btn_frame, text='Predict', command=self.predict)
        predict_btn.grid(row=2, column=0, sticky=W + E)

        self.status_label = Label(btn_frame, text=f"Current Model: {type(self.clf).__name__}")
        self.status_label.config(font=("Aerial", 12))
        self.status_label.grid(row=2, column=1, stick=W + E)

        self.root.protocol("Close window", self.on_closing)
        self.root.attributes("-topmost", True)
        self.root.mainloop()

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill='black', width=self.brush_width)
        self.draw.rectangle([x1, y2, x2 + self.brush_width, y2 + self.brush_width], fill="black",
                            width=self.brush_width)

    def save(self, class_num):
        self.image1.save("temp.png")
        img = Image.open("temp.png")

        img.thumbnail((50, 50), Image.LANCZOS)

        if class_num == 1:
            img.save(f"{self.proj_name}/{self.class1}/{self.class1_counter}.png", "PNG")
            self.class1_counter += 1
        elif class_num == 2:
            img.save(f"{self.proj_name}/{self.class2}/{self.class2_counter}.png", "PNG")
            self.class2_counter += 1
        elif class_num == 3:
            img.save(f"{self.proj_name}/{self.class3}/{self.class3_counter}.png", "PNG")
            self.class3_counter += 1

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 1000, 1000], fill='white')

    def train(self):
        img_list = np.array([])
        class_list = np.array([])

        for x in range(1, self.class1_counter):
            img = cv.imread(f"{self.proj_name}/{self.class1}/{x}.png")[:, :, 0]
            img = img.reshape(2500)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 1)

        for x in range(1, self.class2_counter):
            img = cv.imread(f"{self.proj_name}/{self.class2}/{x}.png")[:, :, 0]
            img = img.reshape(2500)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 2)

        for x in range(1, self.class3_counter):
            img = cv.imread(f"{self.proj_name}/{self.class3}/{x}.png")[:, :, 0]
            img = img.reshape(2500)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 3)

        img_list = img_list.reshape(self.class1_counter - 1 + self.class2_counter - 1 + self.class3_counter - 1, 2500)
        self.clf.fit(img_list, class_list)
        tkinter.messagebox.showinfo("Drawings Detection", "Model Trained", parent=self.root)


    def rotate_model(self):
        if isinstance(self.clf, LinearSVC):
            self.clf = KNeighborsClassifier()
        if isinstance(self.clf, KNeighborsClassifier):
            self.clf = DecisionTreeClassifier()
        elif isinstance(self.clf, KNeighborsClassifier):
            self.clf = RandomForestClassifier()

        self.status_label.config(text=f"Current model: {type(self.clf).__name__}")

    def predict(self):
        self.image1.save("temp.png")
        img = PIL.Image.open("temp.png")
        img.thumbnail((50, 50), PIL.Image.LANCZOS)
        img.save("predictshape.png", "PNG")

        img = cv.imread("predictshape.png")[:, :, 0]
        img = img.reshape(2500)
        prediction = self.clf.predict([img])
        if prediction[0] == 1:
            tkinter.messagebox.showinfo("Drawings Detection", f"The drawing is probably {self.class1}",
                                        parent=self.root)
        elif prediction[0] == 2:
            tkinter.messagebox.showinfo("Drawings Detection", f"The drawing is probably {self.class2}",
                                        parent=self.root)
        elif prediction[0] == 3:
            tkinter.messagebox.showinfo("Drawings Detection", f"The drawing is probably {self.class3}",
                                        parent=self.root)



    def on_closing(self):
        answer = tkinter.messagebox.askokcancel("Quit?", "Do you want to save?", parent=self.root)
        if answer is not None:
            if answer:
                self.save_all()
            self.root.destroy()
            exit()


DrawingClassifier()
