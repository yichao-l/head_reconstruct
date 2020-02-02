import numpy as np
import matplotlib.pyplot as plt
import os
import struct
from vpython import *
import pickle

def float_2_rgb(num):
    packed = struct.pack('!f', num)
    return [c for c in packed][1:]


class ThreeDHead():
    def __init__(self, data_path=None):
        self.data_path=data_path


    @classmethod
    def read_from_file(cls, sequence_id, frame_id):

        data_path = f"./Data/{sequence_id}"
        this = cls(data_path)
        file_name = os.path.join(data_path, f"{sequence_id}_{frame_id}.pcd")

        pc = np.genfromtxt(file_name, skip_header=13)

        this.xyz = pc[:, 0:3]
        this.rgb = np.asarray([float_2_rgb(num) for num in pc[:, 3]])/256
        this.sequence_id=sequence_id
        this.frame_id=frame_id
        return this

    @classmethod
    def load(cls, data_file='head.p'):
        with open('head.p', 'rb') as file_object:
            raw_data = file_object.read()
        return pickle.loads(raw_data)


    def img_coord_from_xyz(self):
        # creates a list of x,y coordinates
        self.index2xy=None

    def filter_1(self):
        self.rgb=self.rgb
        self.xyz=self.xyz


    def filter_depth(self,depth):
        depth_filter = self.xyz[:, 2] < depth
        self.rgb=  self.rgb[depth_filter]
        self.xyz=  self.xyz[depth_filter]

    def filter_nan(self):
        nan_filter = ~np.isnan(self.xyz).any(axis=1)
        self.xyz = self.xyz[nan_filter]
        self.rgb = self.rgb[nan_filter]

    def sparsify(self,sparsity):
        l=self.xyz.shape[0]
        filter = np.random.random((l)) < sparsity
        self.xyz = self.xyz[filter]
        self.rgb = self.rgb[filter]

    def center(self):
        self.xyz = self.xyz - self.xyz.mean(axis=0)

    def remove_edges(self):
        return None

    def create_vpython_spheres(self):
        self.spheres = []
        for i in range(self.xyz.shape[0]):
            next = vec(self.xyz[i,0],-self.xyz[i,1],-self.xyz[i,2])
            self.spheres.append({'pos':next, 'radius':0.003, 'color':(vec(self.rgb[i,0],self.rgb[i,1],self.rgb[i,2]))})

    def save(self, file_name='head.p'):
        pickle.dump(self, open(file_name, 'wb'))

