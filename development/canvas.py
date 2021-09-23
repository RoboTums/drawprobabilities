from tkinter import *
from PIL import Image, ImageTk
import pandas as pd

size = 400
point_cache = []

def get_x_and_y(event):
    global lasx, lasy
    lasx, lasy = event.x, event.y

def draw_smth(event):
    global lasx, lasy
    canvas.create_line((lasx, lasy, event.x, event.y), 
                      fill='red', 
                      width=2)
    lasx, lasy = event.x, event.y
    point_cache.append((lasx,lasy))
    #print(lasx,lasy)

def send_points_to_fitter(event):
	point_df = pd.DataFrame(point_cache,columns=["X",'Y'])
	point_df['Y'] = size - point_df['Y']
	point_df.to_csv('points.csv')

if __name__ == '__main__':
	app = Tk()

	canvas = Canvas(app, bg='black')
	canvas.pack(anchor='center', fill='both', expand=1)
	
	image = Image.open("grid.jpg")
	image = image.resize((size,size), Image.ANTIALIAS)
	image_tk = ImageTk.PhotoImage(image)
	
	canvas.create_image(0,0, image=image_tk, anchor='nw')
	canvas.bind("<Button-1>", get_x_and_y)
	canvas.bind("<B1-Motion>", draw_smth)
	canvas.bind("<Button-2>", send_points_to_fitter)
	app.geometry("400x400")
	app.mainloop()
