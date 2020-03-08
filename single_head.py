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
from tqdm.notebook import tqdm
from sklearn.neighbors import NearestNeighbors


def float_2_rgb(num):
    packed = struct.pack('!f', num)
    return [c for c in packed][1:]


class SingleHead():
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.background_color = 255
        self.keypoints_clr = [1, 0, 0]
        self.visible = True

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
        # self.edge_based_filter(erode_ite=1)
        self.center()
        self.create_vpython_spheres()
        self.save()

    def color_eye(self,row,column):
        self.left_eye_ind = row * 640 + column
        image = self.twoD_image.copy()
        image = image.reshape(-1,3)
        image[self.left_eye_ind] = [1,0,0]
        image = image.reshape(480,640,3)
        plt.imshow(image); plt.show()
        plt.imsave("left_eye_mark.png",image)

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
        twoD_image = self.twoD_image.copy().reshape(-1, 3)
        if not self.background_color is None:
            img = 0 * np.ones((480 * 640, 3)) * self.background_color
        else:
            img = np.zeros((480 * 640, 3))
        for v in self.xy_mesh:
            img[v] = twoD_image[v]
        # save image to head_2d_image
        image_dir = "head_2d_image"
        save_path = os.path.join(image_dir,"head_{}_{}.png".format(self.sequence_id,\
        self.frame_id))
        plt.imsave(save_path,img.reshape(480,640,3))
        return img.reshape(480, 640, 3)

    def reset_filters(self):
        '''
        Resets all the filters and create a xy_mesh variable storing the indices of the original/unfiltered pixels, which becomes
        useful when some pixels are filtered out later (when the length of xy_mesh is less than 640*480). So e.g. we can say after several
        filters that the first element in the xy_mesh corresponds to the pixel 105731 in the original image.
        '''
        self.xy_mesh = np.arange(640 * 480)
        self.xyz=self.xyz_unfiltered
        self.rgb=self.rgb_unfiltered



    def reset_positions(self):
        '''
        Resets the xyz position to the untransformed position.
        '''
        self.xyz=self.xyz_unfiltered[self.xy_mesh]

    def reset_colors(self):
        '''
        Resets the rgb values to the unfiltered, unmarked rgb values
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

        img = np.zeros(480 * 640) > 0
        for v in self.xy_mesh:
            img[v] = True
        return img.reshape(480, 640)

    def transform(self, tform):

        '''
        transform the image:  XYZ*cR + t
        '''

        R, c, t = tform['rotation'], tform['scale'], tform['translation']

        self.xyz = self.xyz.dot(c * R) + t
        self.keypoints = self.keypoints.dot(c * R) + t

    def transform_homo(self, T):
        '''
        transform the image:  XYZ*T
        '''
        ones = np.ones((1, self.xyz.shape[0]))
        self.xyz = np.concatenate((self.xyz, ones.T), axis=1)
        self.xyz = self.xyz.dot(T)
        self.xyz = self.xyz[:, :3]


    def paint(self, color):
        '''
       
        '''
        color=np.asarray(color).reshape(-1)
        # self.rgb = self.rgb.mean(axis=1).reshape((-1,1)).dot(np.asarray([[1,1,1]])) * color
        self.rgb = self.rgb.mean(axis=1).reshape((-1,1)).dot(np.asarray([[0,0,0]])) + color



    def edge_based_filter(self,up=150,down=370,left=260,right=480):
        '''
        Take the twoD_image attribute and generate a binary 
        filter based on edge detection and binary fill holes.
        '''
        # extract the s layer of the HSV image
        image = self.twoD_image.copy()
        plt.imsave("head_2d_image/full_{}_{}.png".format(self.sequence_id,self.frame_id),image)
        # plt.imsave("head_2d_image/filter_{}_{}.png".format(self.sequence_id,self.frame_id),image)
        
        image = cv2.imread("head_2d_image/full_{}_{}.png".format(self.sequence_id,self.frame_id))
        # image[right:,:]=0
        edge = cv2.Canny(image,0,250)
        
        for i in range (3):
            _, contours, hierarchy = cv2.findContours(edge,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            image_another = image.copy()
            cv2.drawContours(image_another, contours, -1, (255,0,0), 0)
            edge = cv2.Canny(image_another,0,250)

        plt.imshow(edge);plt.show()
        edge[479,:]=1
        kernel = np.ones((3,3))
        dilation = cv2.dilate(edge,kernel,iterations =4)
        # dilation[:up,:]=0
        # dilation[down:,:]=0
        # dilation[:,right:]=0
        # dilation[:,:left]=0
        im_floodfill = binary_fill_holes(dilation)
        im_floodfill = im_floodfill*1
        im_floodfill = np.uint8(im_floodfill)
 
        erode = cv2.erode(im_floodfill,kernel,iterations=0)
        # kernel = np.ones((5,5))
        # dilation = cv2.dilate(erode,kernel,iterations =6)

        # erode = np.sum(erode,axis=1)
        # store the filter
        edge_filter = erode > 0
        edge_filter = np.ravel(edge_filter)

        filter = [edge_filter[i] for i in self.xy_mesh]
        # print(filter)
        self.xy_mesh = self.xy_mesh[filter]
        self.rgb = self.rgb[filter]
        self.xyz = self.xyz[filter]

    def erode_filter(self,erode_iteration=1):

        self.get_filtered_image()
        # plt.imsave("head_2d_image/filter_{}_{}.png".format(self.sequence_id,self.frame_id),image)

        image = cv2.imread("head_2d_image/filtered_{}_{}.png".format(self.sequence_id,self.frame_id))
        kernel = np.ones((3,3))

        erode = cv2.erode(image,kernel,iterations=erode_iteration)
        erode = erode.reshape(-1,3)
        erode = np.sum(erode,axis=1)
        edge_filter = erode > 0

        filter = [edge_filter[i] for i in self.xy_mesh]
        # print(filter)
        self.xy_mesh = self.xy_mesh[filter]
        self.rgb = self.rgb[filter]
        self.xyz = self.xyz[filter]


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

    def parzen_filter(self,p=2,r=0.02):
        '''
        Remove the "flying pixels" if there are less than p pixels within a distance r from the 3D pixel.
        '''
        filter = np.ones(self.xy_mesh.shape) > 0
        start_cnt = np.sum(filter)
        end_cnt = 0
        remove_count = 0
        for _ in range(10):
            filter = np.ones(self.xy_mesh.shape) > 0
            start_cnt = np.sum(filter)
            length = len(filter)
            bool_img = self.get_bool_image()
            NN = NearestNeighbors()
            NN.fit(self.xyz)
            for i,(coord,index) in enumerate (zip(self.xyz,self.xy_mesh)):
                y=index//640
                x=index%640
                small_bool = bool_img[max(y-1,0):y + 2, max(x-1,0):x + 2]
                    
                if np.sum(small_bool)<9:
                    # calculate the number of points near coord with a radius of r
                    num_within = len(NN.radius_neighbors([coord],radius=r,return_distance=False)[0])
                    if num_within < p :
                        remove_count+=1
                        filter[i] = False
                
            self.xy_mesh=self.xy_mesh[filter]
            self.xyz = self.xyz[filter]
            self.rgb = self.rgb[filter]
            end_cnt = np.sum(filter)
        print(remove_count)

    def remove_dangling(self):
        '''
        Remove the "flying pixels" if there are less than 2 pixels in a 3x3 window around that pixel.
        '''
        filter =np.ones(self.xy_mesh.shape) >0
        start_cnt=np.sum(filter)
        end_cnt = 0
        while end_cnt < start_cnt:
            filter = np.ones(self.xy_mesh.shape) > 0
            start_cnt = np.sum(filter)
            bool_img = self.get_bool_image()

            for i, index in enumerate (self.xy_mesh):
                y=index//640
                x=index % 640
                small_bool = bool_img[max(y-1,0):y + 2, max(x-1,0):x + 2]

                if np.sum(small_bool)<=2:
                    filter[i]=False
            self.xy_mesh=self.xy_mesh[filter]
            self.xyz = self.xyz[filter]
            self.rgb = self.rgb[filter]
            end_cnt = np.sum(filter)


    def remove_color(self):
        '''
        Updates the filters by removing colors on the edge
        This is not the best function
        remove_background_background_color do
        :return:
        '''
        verbose = False
        min_grad=0.15
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
        min_grad=0.08
        size=3
        lb=size//2 # 1
        ub=size//2+1 # 2
        filter = np.ones(self.xy_mesh.shape) > 0
        start_cnt=np.sum(filter)
        m1=np.nanmean(self.rgb_unfiltered,axis=0)
        m2=np.nanmean(self.rgb,axis=0)
        self.background_color = (m1*640*480-m2*start_cnt)/(640*480-start_cnt)
        end_cnt=0

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
                                if verbose:
                                    print("remove")
                                filter[i] = False

            end_cnt = np.sum(filter)
            # print(end_cnt)
            self.xy_mesh = self.xy_mesh[filter]
            self.xyz = self.xyz[filter]
            self.rgb = self.rgb[filter]
            # print(self.rgb.shape)

    def create_keypoints(self, SIFT_contrastThreshold, SIFT_edgeThreshold, SIFT_sigma):
        '''
        Get the SIFT keypoints and descriptions with the specified parameters and remove the keypoints 
        located at the edge of the foreground object.
        '''
        img1 = self.get_filtered_image()
        self.kp, self.des = get_descriptors(img1, SIFT_contrastThreshold, SIFT_edgeThreshold, SIFT_sigma)
        # remove the edges that are on the edge
        rad = 5  # minimum distance from the edge
        self.kp, self.des = self.remove_edge_points(self.kp, self.des, rad=rad)

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
        # if hasattr(self, "keypoints"):
        #     self.spheres += [
        #         {'pos': vec(self.keypoints[i, 0], -self.keypoints[i, 1], -self.keypoints[i, 2]), 'radius': 0.005,
        #          'color': (vec(self.keypoints_clr[0], self.keypoints_clr[1], self.keypoints_clr[2]))} for i in
        #         range(self.keypoints.shape[0])]

    def save(self, file_name=None):
        '''
        :param file_name:
        :return:
        '''
        if not os.path.isdir("pickled_head"):
            os.mkdir("pickled_head")
        if file_name is None:
            file_name = f"pickled_head/head{self.sequence_id}_{self.frame_id}.p"
        pickle.dump(self, open(file_name, 'wb'))
        data_file = f"pickled_head/head_spheres.p"
        pickle.dump(self.spheres, open(data_file, 'wb'))

    def remove_edge_points(self, kps, des, rad):
        '''
        Works on input sift keypoint and  descriptors and filter out the ones that are on the edge.
        '''
        coords = np.round([kp.pt for kp in kps]).astype("int")
        bw = self.get_bool_image()
        offset = rad // 2
        filter = [np.sum(bw[coord[1] - offset:coord[1] - offset + rad,
                         coord[0] - offset:coord[0] - offset + rad]) == rad ** 2 for coord in coords]
        return [kp for i, kp in enumerate(kps) if filter[i]], np.asarray([d for i, d in enumerate(des) if filter[i]])

    def create_profile(self):
        big_angle_step = 10
        small_angle_step = 1
        O = np.asarray([0, 0, 0.2])
        y_range = (-0.25, 0.1)
        y_step = 0.04
        small_angle_step = 1
        points = self.xyz
        filter_s = np.logical_and(points[:, 1] > y_range[0] - 0.5 * y_step, points[:, 1] < y_range[1] + 0.5 * y_step)
        points_s = points[filter_s]
        colors_s = self.rgb[filter_s]
        self.r_profile = np.zeros((int(360 / small_angle_step), int(1 + (y_range[1] - y_range[0]) // y_step)))
        self.rgb_profile = np.zeros((self.r_profile.shape[0], self.r_profile.shape[1], 3))
        y_values = np.arange(y_range[0], y_range[1], step=y_step)
        for y_i, y in enumerate(y_values):
            O_for_y = np.asarray([0, y, 0]) + O
            filter_ss = np.logical_and(points_s[:, 1] > y - 2 * y_step, points_s[:, 1] < y + 2 * y_step)
            points_ss = points_s[filter_ss]
            colors_ss = colors_s[filter_ss]

            vec_from_O_ss = points_ss - O_for_y
            angles_ss = np.mod(np.angle(vec_from_O_ss[:, 0] + 1j * vec_from_O_ss[:, 2]), 2 * np.pi)
            for theta_0 in np.pi * np.arange(0., 360, step=big_angle_step) / 180:
                filter_sss = np.logical_or(
                    np.mod(angles_ss - theta_0, 2 * np.pi) > np.mod(-(np.pi * small_angle_step / 180), 2 * np.pi),
                    np.mod(angles_ss - theta_0, 2 * np.pi) < (big_angle_step + small_angle_step) * np.pi / 180)
                points_sss = points_ss[filter_sss]
                vec_from_O_sss = vec_from_O_ss[filter_sss]
                colors_sss = colors_ss[filter_sss]

                if points_sss.size > 0:
                    for theta_1 in np.pi * np.arange(0, big_angle_step, step=small_angle_step) / 180:
                        theta = theta_0 + theta_1
                        U = np.cos(theta), 0, np.sin(theta)
                        # o = {'type': "point", 'pos': prj(U+O_for_y), 'radius': "0.01", 'color': vec(1.0,0,1.0)}
                        # self.objects.append(o)
                        filter1 = np.linalg.norm(vec_from_O_sss - np.inner(U, vec_from_O_sss).reshape((-1, 1)) * U,
                                                 axis=1) < 0.005
                        filter2 = np.inner(U, vec_from_O_sss) > 0
                        filter = np.logical_and(filter1, filter2)
                        filtered_points = points_sss[filter]
                        if len(filtered_points) > 0:
                            angles_sss = angles_ss[filter_sss]
                            # print(f"{np.mean(angles_sss[filter])*180/np.pi:.2f}\t{theta*180/np.pi:.2f}" )
                            self.r_profile[int(np.round(theta * 180 / np.pi)), y_i] = np.linalg.norm(
                                np.max(filtered_points, axis=0) - O_for_y)
                            self.rgb_profile[int(np.round(theta * 180 / np.pi)), y_i] = np.mean(colors_sss[filter],
                                                                                                axis=0)
        return self.r_profile, self.rgb_profile, y_values

    def find_angles(self):
        big_angle_step = 10
        O = np.asarray([0, 0, 0.2])
        y_range = (-0.25, 0.1)
        y_step = 0.04
        y_steps = (y_range[1] - y_range[0]) // y_step

        small_angle_step = 1
        points = self.xyz
        self.angles = np.ones((int(y_steps + 1), 360 // small_angle_step)) > 0

        filter_s = np.logical_and(points[:, 1] > y_range[0] + 0.5 * y_step, points[:, 1] < y_range[1])
        points_s = points[filter_s]
        y_values = np.arange(y_range[0], y_range[1], step=y_step)
        for y_i, y in enumerate(y_values):
            O_for_y = np.asarray([0, y, 0]) + O
            filter_ss = np.logical_and(points_s[:, 1] > y - 2 * y_step, points_s[:, 1] < y + 2 * y_step)
            points_ss = points_s[filter_ss]
            vec_from_O_ss = points_ss - O_for_y
            angles_ss = np.mod(np.angle(vec_from_O_ss[:, 0] + 1j * vec_from_O_ss[:, 2]), 2 * np.pi)
            for theta in np.pi * np.arange(0, 360, step=small_angle_step) / 180:
                filter_sss1 = (
                        np.mod(angles_ss, 2 * np.pi) > np.mod(theta - 0.5 * np.pi * small_angle_step / 180, 2 * np.pi))
                filter_sss2 = (
                        np.mod(angles_ss, 2 * np.pi) < np.mod(theta + 0.5 * np.pi * small_angle_step / 180, 2 * np.pi))
                self.angles[y_i, int(np.round(theta * 180 / np.pi))] &= np.sum(filter_sss1) > 0 and np.sum(
                    filter_sss2) > 0

        return self.angles, np.all(self.angles, axis=0), y_values
