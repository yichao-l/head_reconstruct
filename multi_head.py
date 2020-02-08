from Procrustes2 import *
import pickle
import HEAD_RECON
import icp
from SIFT import *


class MultiHead():

    def __init__(self):
        self.heads = []

    @classmethod
    def joined_heads(cls, head1, head2):

        this= cls()
        this.heads.append(head1)
        this.heads.append(head2)
        this.join_heads(0, 1)
        return this




    def join_heads(self, index1, index2):
        head1 = self.heads[index1]
        head2 = self.heads[index2]

        img1, path1 = head1.get_filtered_image()
        img2, path2 = head2.get_filtered_image()
        kp1, des1 = get_descriptors(path1)
        kp2, des2 = get_descriptors(path2)
        good_without_list = get_matched_points(path1, kp1, des1, path2, kp2, des2, 0.8)

        cleaned_match = clean_matches(kp1, path1, kp2, path2, good_without_list)

        head1.reset_colors()
        head2.reset_colors()
        # code below can be used to create
        # head1.paint([0, 0, 1])
        # head2.paint([1, 1, 0])
        head1.reset_positions()
        head2.reset_positions()
        head1.center()
        head2.center()

        matches = cleaned_match[2:]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        xy1 = np.round(pts1).astype("int").reshape(-1, 2)
        xyindex1 = xy1[:, 1] * 640 + xy1[:, 0]
        xyz1 = head1.xyz_unfiltered[xyindex1]

        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        xy2 = np.round(pts2).astype("int").reshape(-1, 2)
        xyindex2 = xy2[:, 1] * 640 + xy2[:, 0]
        xyz2 = head2.xyz_unfiltered[xyindex2]

        list_query_idx = [m.queryIdx for m in matches]
        list_train_idx = [m.queryIdx for m in matches]
        if len(list_train_idx) != len(set(list_train_idx)):
            print("ids are not unique")

        for i in xyindex1:
            pos = np.where(head1.xy_mesh == i)
            head1.rgb[pos] = [0, 1, 0]
        head1.create_vpython_spheres()
        # head1.save()

        for i in xyindex2:
            pos = np.where(head2.xy_mesh == i)
            head2.rgb[pos] = [0, 0, 1]
            head2.xyz

        # todo, make list of points unique

        A = head1.xyz_unfiltered[xyindex1] - head1.center_pos
        B = head2.xyz_unfiltered[xyindex2] - head2.center_pos
        print(A)
        print(B)


        d, Z, tform = procrustes(A, B, scaling=False, reflection='best')

        R, c, t = tform['rotation'], tform['scale'], tform['translation']

        head2.transform(R, c, t)

        head1.create_vpython_spheres()
        head2.create_vpython_spheres()
        # todo create spheres as part of the mulit-head object
        # head1.save()
        # head2.save()

        self.spheres = head1.spheres + head2.spheres
        pickle.dump(self.spheres, open("pickled_head/before_icp.p", 'wb'))

    def icp_transform(self,index1,index2,r=0.01,file_name='pickled_head/after_icp.p'):
        '''
        param:
        r (float): sampleing rate for head1 
        file_name (string): file name of combined spheres
        '''
        # perform one iteration of icp algorithm
        head1 = self.heads[index1]
        head2 = self.heads[index2]

        # sample both array to the same size
        n_sample = int(head1.xyz.shape[0]*r)
        n_1 = head1.xyz.shape[0]
        n_2 = head2.xyz.shape[0] 
        sample_1 = np.random.choice(np.arange(n_1),n_sample)
        sample_2 = np.random.choice(np.arange(n_2),n_sample) 
        T, distance, ite = icp.icp(head1.xyz[sample_1], head2.xyz[sample_2])


        head2.transform_homo(T)

        head1.create_vpython_spheres()
        head2.create_vpython_spheres()
        self.spheres = head1.spheres + head2.spheres
        pickle.dump(self.spheres, open(file_name, 'wb'))
        print("icp processing done.")
        return

    def save_spheres(self):
        pickle.dump(self.spheres, open("head_spheres.p", 'wb'))
