'''
GUI for rendering 3D head objects
'''

import single_head
import numpy as np
import matplotlib.pyplot as plt
import os
import struct
from vpython import *
import pickle


scene.width = scene.height = 800
scene.background = color.white
scene.range = 0.5

run = False

def Runbutton(b):
    global run
    if b.text == 'Pause':
        run = False
        b.text = 'Run'
    else:
        run = True
        b.text = 'Pause'


def Readbutton(b):
    if 'l' in globals():
        global l
        for i in range(len(l)):
            o = l.pop()
            o.visible = False
            del o

    if 'c' in globals():
        global c
        c.visible = False
        del c

    data_file = 'pickled_head/head_spheres.p'
    try:
        with open(data_file, 'rb') as file_object:
            raw_data = file_object.read()
        spheres = pickle.loads(raw_data)
    except:
        raise FileExistsError(f'{data_file} could not be found, create {data_file} by using .save() first ')

    c = points(pos=spheres, size_units='world')
    print(len(spheres))


def ReadMeshbutton(b):
    if 'c' in globals():
        global c
        c.visible = False
        del c

    if 'l' in globals():

        for i in range(len(l)):
            o = l.pop()
            o.visible = False
            del o

    data_file = 'pickled_head/head_mesh.p'
    try:
        with open(data_file, 'rb') as file_object:
            raw_data = file_object.read()
        mesh = pickle.loads(raw_data)
    except:
        raise FileExistsError(f'{data_file} could not be found, create {data_file} by using .save() first ')

    for o in mesh:
        if o['type'] == "pyr":
            p = pyramid(pos=o['pos'])
            l.append(p)
        elif o['type'] == "point":
            col = o['color']
            r = float(o['radius'])
            p = sphere(pos=o['pos'], color=col, radius=r)
            l.append(p)


button(text='Run', bind=Runbutton)
button(text='Read', bind=Readbutton)
button(text='Mesh', bind=ReadMeshbutton)

scene.append_to_caption("""<br>Right button drag or Ctrl-drag to rotate "camera" to view scene.
Middle button or Alt-drag to drag up or down to zoom in or out.
  On a two-button mouse, middle is left + right.
Touch screen: pinch/extend to zoom, swipe or two-finger rotate.""")

data_file = 'pickled_head/head_spheres.p'
try:
    with open(data_file, 'rb') as file_object:
        raw_data = file_object.read()
    spheres = pickle.loads(raw_data)
    print(len(spheres))
except:
    raise FileExistsError(f'{data_file} could not be found, create {data_file} by using .save() first ')

c = points(pos=spheres, size_units='world')

l = []

while True:
    rate(20)
    if run:  # Currently there isn't a way to rotate a points object, so rotate scene.forward:
        scene.forward = scene.forward.rotate(angle=-0.005, axis=vec(0, 1, 0))
