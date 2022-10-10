# -*- coding: utf-8 -*-
import sys
import os
import tkinter
import datetime
from PIL import Image, ImageTk
import csv

root = tkinter.Tk()
root.title("犬を見る")
root['bg'] = '#F0F0F0'
root.state('zoomed')
height = root.winfo_height()
width = root.winfo_width()

time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
csv_file = open(time + '.csv', 'w')
writer = csv.writer(csv_file)


def display_photo():
    global num, canvas, root, img, csv_file, writer
    if num >= len(filenames):
        canvas.delete("all")
        csv_file.close()
        sys.exit()
        
    filename = filenames[num]
    num += 1
    # print(filename)
    # リサイズ
    img = Image.open(filename)
    x, y = img.size
    #longer = x if x > y else y
    mag = float(height) / y
    img = img.resize((int(x * mag), int(y * mag)))
    img = ImageTk.PhotoImage(img)
    canvas.create_image((width - int(x * mag))/ 2 , (height - int(y * mag))/ 2, image=img, anchor=tkinter.NW)
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    writer.writerow([time, filename])
    root.after(100, display_photo)


num = 0
filenames = os.listdir('fig')
os.chdir('fig')
canvas = tkinter.Canvas(root, bg="#F0F0F0", height=height, width=width)
# キャンバス表示
canvas.place(x=0, y=0)

display_photo()
root.mainloop()
