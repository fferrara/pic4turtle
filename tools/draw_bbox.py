# -*- coding: utf-8 -*-

import Tkinter as tk
import os
import random
import sys
import subprocess
import tkFileDialog
import tkMessageBox
import tkFont
from PIL import ImageTk, Image

class DrawBBox(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0

        self.WIDTH = 800
        self.HEIGHT = 600
        self.CANVAS_WIDTH = self.WIDTH - 50
        self.CANVAS_HEIGHT = self.HEIGHT - 50

        self.mainFrame = tk.Frame(self, width=self.WIDTH, height=self.HEIGHT)
        self.mainFrame.grid_propagate(False)  # size fixed
        self.mainFrame.grid(row=0, column=0, padx=20)  # show at top

        self.canvas = tk.Canvas(self.mainFrame, width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT, cursor='cross')
        self.canvas.pack(side='top', fill='both', expand=True)

        # bottom frame
        self.buttonFrame = tk.Frame(self, width=self.WIDTH, height=50)
        self.buttonFrame.grid_propagate(False)  # size fixed
        self.buttonFrame.grid(row=1, column=0, padx=20)  # show at bottom

        self.b_prev = tk.Button(self.buttonFrame, text="<-", command=self.on_prev, padx=20, state=tk.DISABLED)
        self.b_prev.grid(row=0, column=0, sticky=tk.N+tk.E)
        self.b_next = tk.Button(self.buttonFrame, text="->", command=self.on_next, padx=20)
        self.b_next.grid(row=0, column=1)
        self.b_store = tk.Button(self.buttonFrame, text="Confirma", command=self.on_ok, padx=20)
        self.b_store.grid(row=0, column=2)
        self.num_image = tk.StringVar()
        tk.Label(self.buttonFrame, textvariable=self.num_image, font=tkFont.Font(family="Helvetica", size=15)
                 ).grid(row=0, column=3)

        # right frame
        self.classFrame = tk.Frame(self, width=50, height=self.HEIGHT)
        self.classFrame.grid_propagate(False)  # size fixed
        self.classFrame.grid(row=0, column=1, padx=20, pady=20)  # show at top

        self.b_viewer = tk.Button(self.classFrame, text='Abre com visualizador', command=self.on_view)
        self.b_viewer.pack(anchor=tk.S, fill=tk.X, pady=10, ipady=15)

        # handle mouse press, motion and release
        self.canvas.bind('<ButtonPress-1>', self.on_button_press)
        self.canvas.bind('<B1-Motion>', self.on_mouse_motion)
        self.canvas.bind('<ButtonRelease-1>', self.on_button_release)

    def create_data_structure(self, descriptorFile, annotationFile):
        # open file and read images list
        with open(descriptorFile) as f:
            all_images = [
                l.strip().split(' ')[0]
                for l in f.readlines()
            ]

        annotated = []
        with open(annotationFile) as f:
            # file format: filename x0 y0 x1 y1
            annotated.extend( [os.path.split(l.strip().split(' ')[0])[1] for l in f.readlines()] )

        # create image list with no already annotated files
        images = []
        # only with no annotation
        images = [f for f in all_images if os.path.basename(f) not in annotated]

        self.images = images
        self.current_index = 0
        self.annotationFile = annotationFile

        # show image
        self.num_image.set('%d / %d' % (self.current_index, len(self.images)))
        self.__update_image()

    def __clean_state(self):
        self.canvas.delete('bbox')
        self.current_bbox = (0, 0, 0, 0)
        self.num_image.set('%d / %d' % (self.current_index, len(self.images)))

    def __update_image(self):
        self.canvas.delete('bbox')

        # prepare image
        img = Image.open(self.images[self.current_index])
        width, height = img.size
        resize_ratio = (1, 1)
        if width > self.CANVAS_WIDTH or height > self.CANVAS_HEIGHT:  # need to resize
            new_width = self.CANVAS_WIDTH + 0.
            new_height = self.CANVAS_HEIGHT + 0.
            resize_ratio = (width / new_width, height / new_height)

            img.thumbnail((new_width, new_height), Image.ANTIALIAS)

        self.photo = ImageTk.PhotoImage(img)
        self.photo.size = img.size
        self.photo.resize_ratio = resize_ratio
        self.photo.name = self.images[self.current_index]
        self.photo.drawn = self.canvas.create_image((5,5), image=self.photo, anchor=tk.NW)

    # EVENT HANDLERS

    def on_view(self):
        if sys.platform.startswith('linux'):
            ret_code = subprocess.call(['xdg-open', self.photo.name])

        elif sys.platform.startswith('darwin'):
            ret_code = subprocess.call(['open', self.photo.name])

        elif sys.platform.startswith('win'):
            ret_code = subprocess.call(['start', self.photo.name], shell=True)

    def on_next(self):
        self.b_prev.config(state=tk.NORMAL)

        if self.current_index == len(self.images) - 1:
            self.b_next.config(state=tk.DISABLED)
        else:
           self.current_index += 1

           print self.current_index

           self.__update_image()
           self.__clean_state()

    def on_prev(self):
        self.b_next.config(state=tk.NORMAL)

        self.current_index -= 1
        if self.current_index == 0:
            self.b_prev.config(state=tk.DISABLED)

        print self.current_index
        self.__update_image()
        self.__clean_state()

    def on_ok(self):
        try:

            # Store annotations
            (x0, y0, x1, y1) = self.current_bbox
            if x1 - x0 < 10 or y1 - y0 < 10:
                raise ValueError('Desenhe uma bounding box valida')
            line = '%s %d %d %d %d\n' % (self.photo.name, x0, y0, x1, y1)
            with open(self.annotationFile, 'a') as f:
                f.write(line)

            # print 'Deleting index %d' % self.current_index
            # print 'File is %s' % self.images[self.current_index]
            del self.images[self.current_index]
            try:
                self.current_index = random.randrange(len(self.images))
            except ValueError as e: # game over
                exit()

            self.__update_image()
            self.__clean_state()
        except ValueError as e:
            tkMessageBox.showerror('Erro', e.message)
        except AttributeError:
            tkMessageBox.showerror('Erro', 'Desenha a bounding box antes')

    def on_button_press(self, event):
        self.x = event.x
        self.y = event.y

        x0, y0 = (self.x, self.y)
        x1, y1 = (x0+1, y0+1)

        # create small rectangle
        self.canvas.delete('bbox')
        self.bbox = self.canvas.create_rectangle(x0,y0,x1,y1, fill='')
        self.canvas.itemconfig(self.bbox, tags='bbox')

    def on_mouse_motion(self, event):
        x0, y0 = (self.x, self.y)
        x1, y1 = (event.x, event.y)

        self.canvas.coords(self.bbox, (x0, y0, x1, y1))

    def on_button_release(self, event):
        (x0, y0, x1, y1) = tuple(self.canvas.coords(self.bbox))

        # ajust coords
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 > self.photo.size[0]:
            x1 = self.photo.size[0]
        if y1 > self.photo.size[1]:
            y1 = self.photo.size[1]

        # Apply resize ratio
        (x0, y0) = [c * ratio for (c, ratio) in zip((x0, y0), self.photo.resize_ratio)]
        (x1, y1) = [c * ratio for (c, ratio) in zip((x1, y1), self.photo.resize_ratio)]

        self.current_bbox = (x0, y0, x1, y1)

if __name__ == '__main__':
    app = DrawBBox()
    fotoDir = tkFileDialog.askopenfilename(title='Seleciona o arquivo descritor')
    annotationFile = tkFileDialog.askopenfilename(title='Seleciona o arquivo de anotações')

    app.create_data_structure(os.path.abspath(fotoDir), os.path.abspath(annotationFile))
    app.mainloop()