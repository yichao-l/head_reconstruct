'''
GUI for rendering 3D head objects
'''

import sys

from vpython import *
from vpython.no_notebook import stop_server
import pickle
import time
import numpy as np

def Savebutton():
    global name
    if name is None:
        file_name = f"mesh.png"
    else:
        file_name = f"{name}.png"
    scene.capture(file_name)


def Readbutton():
    global name
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
        (spheres,name) = pickle.loads(raw_data)
    except:
        raise FileExistsError(f'{data_file} could not be found, create {data_file} by using .save() first ')

    c = points(pos=spheres, size_units='world')
    print(len(spheres))


def ReadMeshbutton():
    global name
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
        (mesh,name) = pickle.loads(raw_data)
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


args = sys.argv

scene.width = scene.height = 800
scene.background = color.white
scene.range = 0.5

button(text='Save', bind=Savebutton)
button(text='Read', bind=Readbutton)
button(text='Mesh', bind=ReadMeshbutton)

scene.append_to_caption("""<br>Right button drag or Ctrl-drag to rotate "camera" to view scene.
Middle button or Alt-drag to drag up or down to zoom in or out.
  On a two-button mouse, middle is left + right.
Touch screen: pinch/extend to zoom, swipe or two-finger rotate.""")

data_file = 'pickled_head/head_spheres.p'

name = None

try:
    with open(data_file, 'rb') as file_object:
        raw_data = file_object.read()
except:
    raise FileExistsError(f'{data_file} could not be found, create {data_file} by using .save() first ')

(spheres, name) = pickle.loads(raw_data)
print(len(spheres))
c = points(pos=spheres, size_units='world')
rate(20)
l = []

print(args)

if "alpha" in args:
    alpha = int(args[args.index("alpha") + 1])

else:
    alpha = 0

scene.light = [
    distant_light(direction=vector(np.sin(-(45 + alpha) * np.pi / 180), 0.3, np.cos(-(45 + alpha) * np.pi / 180)),
                  color=color.gray(0.3)),
    distant_light(direction=vector(np.sin(-(+45 + alpha) * np.pi / 180), 0.3, np.cos(-(-45 + alpha) * np.pi / 180)),
                  color=color.white)]

print(alpha)
scene.forward = vec(np.sin(alpha * np.pi / 180), 0, -np.cos(alpha * np.pi / 180))

if "save_only" in args:
    time.sleep(len(spheres) // 10000)
    Savebutton()
    time.sleep(len(spheres) // 30000)
    stop_server()
