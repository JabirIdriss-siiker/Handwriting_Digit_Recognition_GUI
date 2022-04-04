from tkinter import *

import numpy as np
import cv2
from PIL import ImageGrab
from keras.models import load_model


model = load_model('mnist.h5')
image_folder = "img/"

root = Tk()
root.resizable(0, 0)
root.title("Reconnaisance")

lastx, lasty = None, None
image_number = 0

cv = Canvas(root, width=800, height=600, bg='white')
cv.grid(row=0, column=0, pady=2, sticky=NSEW, columnspan=2)

#effacer le canvas
def clear_widget():
    global cv
    cv.delete('all')

#dessiner sur  le canvas
def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=8, fill='black', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y


def activate_event(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y


cv.bind('<Button-1>', activate_event)



def Recognize_Digit():
    global image_number
    filename = f'img_{image_number}.png'

    x = cv.winfo_rootx()
    y = cv.winfo_rooty()
    w = cv.winfo_width()
    h = cv.winfo_height()
    # enregistrer l'image du canvas
    ImageGrab.grab().crop((x, y, x+w, y+h)).save(image_folder + filename)

    image = cv2.imread(image_folder + filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # dessiner un rectancle sur les contoures
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # retrouver le numeros dans l'image
        digit = th[y:y + h, x:x + w]

        # Resizing le chiffre en 18 18
        resized_digit = cv2.resize(digit, (18, 18))

        # ajouter des zeros pour avoir 28 28
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

        digit = padded_digit.reshape(1, 28, 28, 1)
        digit = digit / 255.0
        #lancer la prediction sur le chiffre croper de l'image
        pred = model.predict([digit])[0]
        final_pred = np.argmax(pred)

        data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '%'
        #quelques variable pour le texte
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (0, 0, 255)
        thickness = 1
        cv2.putText(image, data, (x, y - 5), font, fontScale, color, thickness)

    cv2.imshow('Predictions', image)
    cv2.waitKey(0)
    
   
    
   
btn_save = Button(text='Reconnaitre',width=15, height=2, command=Recognize_Digit)
btn_save.grid(row=2, column=0, pady=1, padx=1)

button_clear = Button(text='Effacer',width=15, height=2, command=clear_widget)
button_clear.grid(row=2, column=1, pady=1, padx=1)


root.mainloop()