import cv2
import numpy as np
import tkinter as tk
import keras
import os

from main import display_image
from tkinter import *

pixel_multiplier = 20
src = "resources/imgs/"
img = None

def main():
    create_image()

def evaluate_image(event = None):
    model = keras.models.load_model("models/model.keras")

    inp = np.reshape(inputimg, (-1, 28, 28))
    prediction = model.predict(inp)[0]
    round_prediction = np.round(prediction, 2)
    max_val = np.argmax(round_prediction)

    display_image(inputimg, f"{round_prediction}\n => {max_val}")

inputimg = np.zeros((28, 28))
def addToSelection(event):
    canvas = event.widget
    x = canvas.canvasx(event.x)
    y = canvas.canvasy(event.y)

    overlapping = canvas.find_overlapping(x, y, x+pixel_multiplier, y+pixel_multiplier)
    for item in overlapping:
        canvas.itemconfigure(item, fill="black")

    relativeX = int(x/pixel_multiplier)
    relativeY = int(y/pixel_multiplier)

    for i in range(0, 2):
        for j in range(0, 2):
            try:
                inputimg[relativeY + i][relativeX + j] = 1
            except Exception:
                continue

canvas = None
def clear(event):
    global inputimg
    inputimg = np.zeros((28, 28))

    for item in canvas.find_all():
        canvas.itemconfigure(item, fill="white")

def create_image():
    global canvas
    window = tk.Tk("Draw number")
    window.geometry(f"{28 * pixel_multiplier}x{28*pixel_multiplier + 40}")
    canvas = Canvas(window, width=28 * pixel_multiplier, height=28 * pixel_multiplier)
    canvas.pack()
    button = tk.Button(window, command=evaluate_image, text="Evaluate")
    button.place(x=0, y=28 * pixel_multiplier)

    for row in range(28):
        for col in range(28):
            box = canvas.create_rectangle(col * pixel_multiplier, row * pixel_multiplier, (col + 1) * pixel_multiplier, (row + 1) * pixel_multiplier, fill="white", outline="")
    window.bind("<Key-c>", clear)
    window.bind("<Return>", evaluate_image)
    canvas.bind("<Button1-Motion>", addToSelection)
    window.mainloop()

def convert_images(srcimgs):
    result = []
    for src in srcimgs:
        img = cv2.bitwise_not(cv2.imread(src, cv2.IMREAD_GRAYSCALE)).astype(float)/255
        result.append(img)

    return np.array(result).reshape(-1, 28, 28)

if __name__ == "__main__":
    main()