import HEAD_RECON
import numpy as np
import matplotlib.pyplot as plt
import os
import struct
from vpython import *
import pickle

head = HEAD_RECON.threeD_head.load()
scene.width = scene.height = 800
scene.background = color.white
scene.range = 0.3

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
    global c
    with open('head.p', 'rb') as file_object:
        raw_data = file_object.read()
    head = HEAD_RECON.threeD_head.load()
    c.visible=False
    del c
    c = points(pos=head.spheres, size_units='world')


button(text='Run', bind=Runbutton)
button(text='Read', bind=Readbutton)


scene.append_to_caption("""<br>Right button drag or Ctrl-drag to rotate "camera" to view scene.
Middle button or Alt-drag to drag up or down to zoom in or out.
  On a two-button mouse, middle is left + right.
Touch screen: pinch/extend to zoom, swipe or two-finger rotate.""")

c = points(pos=head.spheres, size_units='world')

while True:
    rate(20)
    if run:  # Currently there isn't a way to rotate a points object, so rotate scene.forward:
        scene.forward = scene.forward.rotate(angle=-0.005, axis=vec(0, 1, 0))