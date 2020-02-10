import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import struct
from vpython import *
import pickle
import cv2
from scipy.ndimage.morphology import binary_fill_holes
from SIFT import *

def float_2_rgb(num):
    packed = struct.pack('!f', num)
    return [c for c in packed][1:]
        
class threeD_head():
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.background_color = 1
        self.keypoints = []
        self.keypoints_clr = [1, 0, 0]

    @classmethod
    def read_from_file(cls, sequence_id, frame_id):
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
        this.xyz_unfiltered= pc[:, 0:3]
        this.rgb_unfiltered= np.asarray([float_2_rgb(num) for num in pc[:, 3]])/256
        this.sequence_id=sequence_id
        this.frame_id=frame_id
        this.img_coord_from_xyz()

        return this

    @classmethod
    def load_from_pickle(cls, sequence_id, frame_id):
            return cls.load(f"pickled_head/head{sequence_id}_{frame_id}.p")

    @classmethod
    def load(cls, data_file='head.p'):
        '''
        :param data_file:  file to load from, default name is the default file used for saving
        :return: object of  threeD_head class
        '''
        try:
            with open(data_file, 'rb') as file_object:
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
        if not self.background_color is None:
            img = 1 * np.ones((480 * 640, 3)) * self.background_color
        else:
            img = np.zeros((480 * 640, 3))
        for v in self.xy_mesh:
            img[v]=twoD_image[v]
        # save image to head_2d_image
        image_dir = "head_2d_image"
        save_path = os.path.join(image_dir,"head_{}_{}.png".format(self.sequence_id,\
        self.frame_id))
        plt.imsave(save_path,img.reshape(480,640,3))
        return img.reshape(480,640,3), save_path


    def reset_filters(self):
        '''
        Resets all the filters
        :return:
        '''
        self.xy_mesh = np.arange(640 * 480)
        self.xyz=self.xyz_unfiltered
        self.rgb=self.rgb_unfiltered



    def reset_positions(self):
        '''
        Resets all the filters
        :return:
        '''
        self.xyz=self.xyz_unfiltered[self.xy_mesh]

    def reset_colors(self):
        '''
        Resets all the filters
        :return:
        '''
        self.rgb=self.rgb_unfiltered[self.xy_mesh]



    def get_bw_image(self):
        '''
        :return: black and white image with all pixels set to white that have not been filtered out
        '''

        img= np.zeros((480*640,3))
        for v in self.xy_mesh:
            img[v]=[1,1,1]
        return img.reshape(480,640,3)

    def get_bool_image(self):
        '''
        :return: a boolean image, True for all pixels thah have not been filtered out
        '''

        img= np.zeros(480*640)>0
        for v in self.xy_mesh:
            img[v]=True
        return img.reshape(480,640)

    def transform(self, R, c,t):
        '''
        transform the image:  XYZ*cR + t
        '''
        self.xyz= self.xyz.dot(c*R)+t

    def transform_homo(self, T):
        '''
        transform the image:  XYZ*T
        '''
        ones = np.ones((1,self.xyz.shape[0]))
        self.xyz = np.concatenate((self.xyz, ones.T), axis=1)
        self.xyz = self.xyz.dot(T)
        self.xyz = self.xyz[:,:3]

    def paint(self, color):
        '''
        transform the image:  XYZ*cR + t
        '''
        color=np.asarray(color).reshape(-1)
        # self.rgb = self.rgb.mean(axis=1).reshape((-1,1)).dot(np.asarray([[1,1,1]])) * color
        self.rgb = self.rgb.mean(axis=1).reshape((-1,1)).dot(np.asarray([[0,0,0]])) + color

    def full_filter(self,  depth=1.5):
        # perform thresholding in depth axis, remove the nan pixels
        # and the flying pixels.
        # Then center the pixels, create vpython spheres
        # and save as pickel obj for future use.
        self.reset_filters()
        self.filter_nan()
        self.filter_depth(depth)
        print("depth filter done.")
        self.remove_dangling()
        print("dangling removal done")
        self.remove_background_color()
        print("color filter done.")
        self.center()
        self.create_vpython_spheres()
        self.save()

    def edge_based_filter(self):
        '''
        Take the twoD_image attribute and generate a binary 
        filter based on edge detection and binary fill holes.
        '''
        # extract the s layer of the HSV image
        image = self.twoD_image.copy()
        # plt.imshow(image)
        
        hsv = matplotlib.colors.rgb_to_hsv(image)
        s = np.uint8(hsv[:,:,1])

        # edge detection
        edge = cv2.Canny(s,150,200)
        dilation_kernel = np.ones((2,2))
        dilation = cv2.dilate(edge,dilation_kernel,iterations=3)
        plt.imshow(edge);plt.show()
        # closing and binary fill
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, np.ones((10,10)))
        im_fill = binary_fill_holes(closing)
        im_fill = im_fill*255
        im_fill = np.uint8(im_fill)
        plt.imshow(im_fill);plt.show()

        # dilate and erode again
        kernel = np.ones((4,4))
        dilation = cv2.dilate(im_fill,kernel,iterations=6)
        erode = cv2.erode(dilation,kernel,iterations=6)

        # filter
        edge_filter = erode > 0
        self.xy_mesh = self.xy_mesh[np.ravel(edge_filter)]
        self.rgb = self.rgb[edge_filter]
        self.xyz = self.xyz[edge_filter]



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
        l = self.xyz.shape[0]
        filter = np.random.random((l)) < sparsity
        self.sparse_xy_mesh = self.xy_mesh[filter]

        self.sparse_xyz = self.xyz[filter]
        self.sparse_rgb = self.rgb[filter]

    def center(self):
        '''
        centers the object
        :return:
        '''
        self.center_pos= self.xyz.mean(axis=0)
        self.xyz = self.xyz - self.center_pos
        self.xyz_unfiltered = self.xyz_unfiltered - self.center_pos

    def remove_dangling(self):
        filter =np.ones(self.xy_mesh.shape) >0
        start_cnt=np.sum(filter)
        end_cnt = 0
        while end_cnt < start_cnt:
            filter = np.ones(self.xy_mesh.shape) > 0
            start_cnt = np.sum(filter)
            bool_img = self.get_bool_image()
            # print("remove danlging start" , np.sum(filter))
            for i, index in enumerate (self.xy_mesh):
                y=index//640
                x=index % 640
                small_bool = bool_img[max(y-1,0):y + 2, max(x-1,0):x + 2]
                # print(i, y,x ,'\n', small_bool)
                # print(np.sum(small_bool))
                if np.sum(small_bool)<=2:
                    filter[i]=False
            self.xy_mesh=self.xy_mesh[filter]
            self.xyz = self.xyz[filter]
            self.rgb = self.rgb[filter]
            # print("remove dangling end" ,np.sum(filter))
            end_cnt = np.sum(filter)
            # print(end_cnt)


    def remove_color(self):
        '''
        Updates the filters by removing colors on the edge
        This is not the best function
        remove_background_background_color do
        :return:
        '''
        verbose = False
        min_grad=0.2
        fudge=0.6
        size=5
        lb=size//2
        ub=size//2+1
        filter =np.ones(self.xy_mesh.shape) >0
        start_cnt=np.sum(filter)
        end_cnt=0
        # print(start_cnt)
        self.background_color=[]
        while end_cnt<start_cnt:
            bool_img = self.get_bool_image()
            filter = np.ones(self.xy_mesh.shape) > 0
            start_cnt = np.sum(filter)
            for i, index in enumerate (self.xy_mesh):
                y=index//640
                x=index%640
                if x>0 and y >0 and x < 640-lb and y < 480-lb:

                    small_bool = bool_img[y-1:y + 2, x-1:x + 2]
                    if np.sum(small_bool)<=6:
                        small_rgb = self.twoD_image[y-lb:y + ub, x-lb:x + ub]
                        ctr_color=small_rgb[lb,lb]
                        top_cnt= np.sum(small_bool[0,:])
                        btm_cnt= np.sum(small_bool[2,:])
                        left_cnt= np.sum(small_bool[:,0])
                        right_cnt= np.sum(small_bool[:,2])

                        left_color=np.mean(small_rgb[:,0],axis=0)
                        right_color=np.mean(small_rgb[:,size-1],axis=0)
                        top_color=np.mean(small_rgb[0,:],axis=0)
                        btm_color=np.mean(small_rgb[size-1,:],axis=0)
                        Background = True
                        if left_cnt == 0 and right_cnt >= 1:  # that means left is background
                            bg_clr=left_color
                            fg_clr=right_color
                        elif right_cnt == 0 and left_cnt >= 1:  # that means right is background
                            bg_clr=right_color
                            fg_clr=left_color
                        elif top_cnt == 0 and btm_cnt >= 1:  # that means top is background
                            bg_clr=top_color
                            fg_clr=btm_color
                        elif btm_cnt == 0 and top_cnt >= 1:  # that means top is background
                            bg_clr=btm_color
                            fg_clr=top_color
                        else:
                            Background=False
                        if Background:
                            if verbose:
                                print('t:', top_cnt, 'b:', btm_cnt, 'l:', left_cnt, 'r:', right_cnt)
                            if left_cnt==0 and right_cnt>1: # that means left is background
                                if verbose:
                                    print(np.linalg.norm(fg_clr - ctr_color))
                                    print(np.linalg.norm(bg_clr - ctr_color))
                                    print(np.linalg.norm(fg_clr - bg_clr))
                                if np.linalg.norm(bg_clr - ctr_color) * fudge < np.linalg.norm(
                                            fg_clr - ctr_color) or (np.linalg.norm(
                                            fg_clr - bg_clr) < min_grad):  # if center color is closest to background
                                        if verbose:
                                            print('Remove', top_color)
                                        filter[i] = False
                                        self.background_color.append(self.rgb[i])
            end_cnt = np.sum(filter)
            # print(end_cnt)
            self.xy_mesh=self.xy_mesh[filter]
            self.xyz = self.xyz[filter]
            self.rgb = self.rgb[filter]
        self.background_color=np.mean(self.background_color, axis=0)

    def remove_background_color(self):
        '''
        Updates the filters by removing colors on the edge
        :return:
        '''

        verbose = False
        min_grad=0.15
        min_grad=0.15
        size=3
        lb=size//2
        ub=size//2+1
        filter = np.ones(self.xy_mesh.shape) > 0
        start_cnt=np.sum(filter)

        m1=np.nanmean(self.rgb_unfiltered,axis=0)
        # print(m1)
        m2=np.nanmean(self.rgb,axis=0)
        # print(m2)

        self.background_color = (m1*640*480-m2*start_cnt)/(640*480-start_cnt)
        # print(self.background_color)

        end_cnt=0
        # print(start_cnt)
        while end_cnt<start_cnt:
            bool_img = self.get_bool_image()
            filter = np.ones(self.xy_mesh.shape) > 0
            start_cnt = np.sum(filter)
            for i, index in enumerate (self.xy_mesh):
                y=index//640
                x=index%640
                verbose =(y==200 and x > 270  and x < 300)
                verbose=False
                if x>0 and y >0 and x < 640-lb and y < 480-lb:
                    small_bool = bool_img[y-1:y + 2, x-1:x + 2]
                    small_rgb = self.twoD_image[y - lb:y + ub, x - lb:x + ub]
                    ctr_color = small_rgb[lb, lb]
                    if np.sum(small_bool)<= 6:
                         if verbose:
                             print(small_bool)
                             print(x, y, ctr_color, self.background_color)
                             print(np.linalg.norm(ctr_color - self.background_color))
                         if np.linalg.norm(ctr_color-self.background_color) < min_grad:
                                # print(np.linalg.norm(ctr_color - self.background_color))
                                if verbose:
                                    print ("remove")
                                filter[i] = False

            end_cnt = np.sum(filter)
            # print(end_cnt)
            self.xy_mesh=self.xy_mesh[filter]
            self.xyz = self.xyz[filter]
            self.rgb = self.rgb[filter]
            # print(self.rgb.shape)

    def create_vpython_spheres(self, force_sparce=False):
        '''
        creates the spheres that can be used by vpython
        :return:
        '''
        if force_sparce:
            sparce_xyz = self.sparse_xyz
            sparce_rgb = self.sparse_rgb
        else:
            sparce_xyz = self.xyz
            sparce_rgb = self.rgb
        radius = np.ones(sparce_xyz.shape[0]) * 0.0015
        rad = 0.001

        self.spheres = [{'pos': vec(sparce_xyz[i, 0], -sparce_xyz[i, 1], -sparce_xyz[i, 2]), 'radius': 0.0015,
                         'color': (vec(sparce_rgb[i, 0], sparce_rgb[i, 1], sparce_rgb[i, 2]))} for i in
                        range(sparce_xyz.shape[0])]

        # self.spheres += [
        #     {'pos': vec(self.keypoints[i, 0], -self.keypoints[i, 1], -self.keypoints[i, 2]), 'radius': 0.005,
        #      'color': (vec(self.keypoints_clr[0], self.keypoints_clr[1], self.keypoints_clr[2]))} for i in
        #     range(self.keypoints.shape[0])]

    def save(self, file_name=None):
        '''
        :param file_name:
        :return:
        '''
        if not os.path.isdir("pickled_head"):
            os.mkdir("pickled_head")
        if file_name is None:
            file_name=f"pickled_head/head{self.sequence_id}_{self.frame_id}.p"
        pickle.dump(self, open(file_name, 'wb'))
        data_file=f"pickled_head/head_spheres.p"
        pickle.dump(self.spheres, open(data_file, 'wb'))

