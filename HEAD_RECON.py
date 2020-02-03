import numpy as np
import matplotlib.pyplot as plt
import os
import struct
from vpython import *
import pickle
import cv2

def float_2_rgb(num):
    packed = struct.pack('!f', num)
    return [c for c in packed][1:]

class estimate_frame_transform():
    '''
    Estimate the reference transformation linking each consecutive pair of frames.
    '''
    def __init__(self,img1, img2):
        '''
        img1, img2: The two images that we want to estimate the 
        transformation for (not sure about the format yet).
        '''
        self.img1 = img1
        self.img2 = img2
    
    @classmethod
    def get_descriptors(cls,image_path):
        '''
        param:
        image (string): path to colored image
        return:
        void
        '''
        # Load the image in BGR
        img = cv2.imread(image_path)
        # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find keypoints
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img,None)
        # img = cv2.drawKeypoints(gray,kp,img)        
        return kp, des

    @classmethod
    def get_matched_points(cls, img1, kp1, des1, img2, kp2, des2):
        '''
        find a set of good matching descriptors given two set of keypoints and descriptors
        '''
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)

        # Apply ratio test
        good = []
        good_without_list = []
        for m,n in matches:
            if m.distance < 0.4*n.distance:
                good.append([m])
                good_without_list.append(m)

        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
        plt.imshow(img3),plt.show()
        plt.imsave("des_match.png",img3)
        return good,good_without_list
        
class threeD_head():
    def __init__(self, data_path=None):
        self.data_path=data_path

    @classmethod
    def read_from_file(cls, sequence_id, frame_id, depth=1.5):
        '''
        Read from a .pcd file and extract its xyz and rgb values as
        two seperate lists, stored in xyz, rgb class variables.
        params:
        sequence_id (int): The person's number, from 1 to 4.
        frame_id (int): The frame number, from 1 to 15.
        depth (float): threshold parameter for the depth filter.
        '''
        # compose the data path
        data_path = f"./Data/{sequence_id}"
        this = cls(data_path)
        file_name = os.path.join(data_path, f"{sequence_id}_{frame_id}.pcd")

        # read the .pcd file
        pc = np.genfromtxt(file_name, skip_header=13)

        # extract the xyz, rgb info
        this.xyz = pc[:, 0:3]
        this.rgb = np.asarray([float_2_rgb(num) for num in pc[:, 3]])/256
        this.sequence_id=sequence_id
        this.frame_id=frame_id
        this.img_coord_from_xyz()

        # perform thresholding in depth axis, remove the nan pixels
        # and the flying pixels.
        # Then center the pixels, create vpython spheres 
        # and save as pickel obj for future use.
        # this.filter_nan()
        # this.filter_depth(depth)
        # this.center()
        # this.create_vpython_spheres()
        # this.save()
        return this

    @classmethod
    def load(cls, data_file='head.p'):
        '''
        :param data_file:  file to load from, default name is the default file used for saving
        :return: object of  threeD_head class
        '''
        try:
            with open('head.p', 'rb') as file_object:
                raw_data = file_object.read()
            return pickle.loads(raw_data)
        except:
            raise FileExistsError (f'{data_file} could not be found, create {data_file} by using .save() first ')


    def img_coord_from_xyz(self):
        '''
        Convert a list of rbg values to a 2D image
        params:
            rgb (list[int,int,int]): a list of rgb values
        return:
            the rgb image in a 480, 640 format
        '''
        image = np.reshape(self.rgb,(480,640,3))
        self.xy_mesh=np.arange(640*480)
        self.twoD_image= image
        return image

    def get_filtered_image(self):
        '''
        get the 2d image after all filters have been applied
        :return: returns image filterd by all filter operations, black where filters have been applied
        '''
        twoD_image= self.twoD_image.copy().reshape(-1, 3)
        img= np.zeros((480*640,3))
        for v in self.xy_mesh:
            img[v]=twoD_image[v]
        return img.reshape(480,640,3)


    def get_bw_image(self):
        '''
        :return: black and white image with all pixels set to the filtered colour
        '''

        img= np.zeros((480*640,3))
        for v in self.xy_mesh:
            img[v]=[1,1,1]
        return img.reshape(480,640,3)


    def filter_1(self):
        self.rgb=self.rgb
        self.xyz=self.xyz


    def filter_depth(self,depth):
        '''
        :param depth: any pixel with depth greater than this value is removed
        :return:
        '''
        depth_filter = self.xyz[:, 2] < depth
        self.xy_mesh=self.xy_mesh[depth_filter]
        self.rgb=  self.rgb[depth_filter]
        self.xyz=  self.xyz[depth_filter]


    def filter_nan(self):
        '''
        removes all entries where any of the xyz coordinates is nan
        '''
        nan_filter = ~np.isnan(self.xyz).any(axis=1)
        self.xy_mesh=self.xy_mesh[nan_filter]
        self.xyz = self.xyz[nan_filter]
        self.rgb = self.rgb[nan_filter]

    def sparsify(self,sparsity):
        '''

        :param sparsity: the fraction of pixles that is retained
        :return: updates the object
        '''
        l=self.xyz.shape[0]
        filter = np.random.random((l)) < sparsity
        self.xy_mesh=self.xy_mesh[filter]
        self.xyz = self.xyz[filter]
        self.rgb = self.rgb[filter]

    def center(self):
        '''
        centers the object
        :return:
        '''
        self.xyz = self.xyz - self.xyz.mean(axis=0)

    def remove_edges(self):
        return None

    def create_vpython_spheres(self):
        '''
        creates the spheres that can be used by vpython
        :return:
        '''
        self.spheres = []
        for i in range(self.xyz.shape[0]):
            next = vec(self.xyz[i,0],-self.xyz[i,1],-self.xyz[i,2])
            self.spheres.append({'pos':next, 'radius':0.003, 'color':(vec(self.rgb[i,0],self.rgb[i,1],self.rgb[i,2]))})

    def save(self, file_name='head.p'):
        '''

        :param file_name:
        :return:
        '''
        pickle.dump(self, open(file_name, 'wb'))

