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
        head1.reset_positions()
        head2.reset_positions()
        head1.center()
        head2.center()
        this.heads.append(head1)
        this.heads.append(head2)
        this.join_heads_wraper(0, 1)
        # this.join_heads(0,1,SIFT_contrastThreshold=0.04,SIFT_edgeThreshold=10,SIFT_sigma=0.5)
        return this

    def append_head(self, head):
        head.reset_positions()
        head.center()
        # head1.center()
        # head2.center()
        self.heads.append(head)


    def join_heads(self, index1, index2, SIFT_contrastThreshold=0.02,SIFT_edgeThreshold=10,SIFT_sigma = 1,searching=1):
        head1 = self.heads[index1]
        head2 = self.heads[index2]

        img1, path1 = head1.get_filtered_image()
        img2, path2 = head2.get_filtered_image()
        kp1, des1 = get_descriptors(path1,SIFT_contrastThreshold,SIFT_edgeThreshold,SIFT_sigma)
        kp2, des2 = get_descriptors(path2,SIFT_contrastThreshold,SIFT_edgeThreshold,SIFT_sigma)
        good_without_list = get_matched_points(path1, kp1, des1, path2, kp2, des2, 1)

        cleaned_match = clean_matches(kp1, path1, kp2, path2, good_without_list)

        # head1.reset_colors()
        # head2.reset_colors()
        # code below can be used to create
        # head1.paint([0, 0, 1])
        # head2.paint([1, 1, 0])
        # head1.reset_positions()
        # head2.reset_positions()
        # head1.center()
        # head2.center()

        matches = cleaned_match[2:]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        xy1 = np.round(pts1).astype("int").reshape(-1, 2)
        xyindex1 = xy1[:, 1] * 640 + xy1[:, 0]
        indices1=[np.argwhere(head1.xy_mesh==ind) for ind in xyindex1]
        filter1=[len(ind) >0 for ind in indices1]

        xyz1 = head1.xyz_unfiltered[xyindex1]
        # xyz1 = head1.xyz[[xyindex1]

        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        xy2 = np.round(pts2).astype("int").reshape(-1, 2)
        xyindex2 = xy2[:, 1] * 640 + xy2[:, 0]
        indices2=[np.argwhere(head2.xy_mesh==ind) for ind in xyindex2]
        filter2=[len(ind) >0 for ind in indices2]
        filter = np.asarray(filter1) & np.asarray(filter2)
        # print (filter)

        indices1 = np.asarray(indices1)[filter]
        # print(indices1)
        indices2 = np.asarray(indices2)[filter]
        # print([ind[0][0] for ind in indices1])
        xyz1 = np.asarray([head1.xyz[ind[0][0]] for ind in indices1])
        xyz2 = np.asarray([head2.xyz[ind[0][0]] for ind in indices2])

        # print(xyz1)
        # print(xyz2)

        # xyz2 = head2.xyz_unfiltered[xyindex2]
        #
        # print(~np.isnan(xyz1).any(axis=1))
        # print(~np.isnan(xyz2).any(axis=1))
        # filter =~np.isnan(xyz1).any(axis=1) & ~np.isnan(xyz2).any(axis=1)

        # xyindex1=xyindex2[filter]
        # xyindex2=xyindex2[filter]

        list_train_idx = [matches[i].trainIdx for i in range(len(matches)) if filter[i]]
        list_query_idx = [matches[i].queryIdx for i in range(len(matches)) if filter[i]]

        # list_train_idx = [m.queryIdx for m in matches]

        if len(list_train_idx) != len(set(list_train_idx)):
            print("ids are not unique")

        # creating markers
        for i in xyindex1:
            pos = np.where(head1.xy_mesh == i)
            head1.rgb[pos] = [0, 1, 0]

        for i in xyindex2:
            pos = np.where(head2.xy_mesh == i)
            head2.rgb[pos] = [0, 0, 1]

        d, Z, tform = procrustes(xyz1, xyz2, scaling=False, reflection='best')

        R, c, t = tform['rotation'], tform['scale'], tform['translation']
        print(c,t)
        head2.transform(R, c, t)

        if searching:
            return self.get_mean_distance(index1,index2)
        # verify the transform
        # if not verified_tranform(index1,index2):
        #     # transform head2 back to its original position?
        #     return False

        # head1.create_vpython_spheres()
        # head2.create_vpython_spheres()
        # todo create spheres as part of the mulit-head object
        # head1.save()
        # head2.save()

        # self.spheres = head1.spheres + head2.spheres

        # comment out for speed
        head1.create_vpython_spheres()
        head2.create_vpython_spheres()

        self.spheres = head1.spheres + head2.spheres

        self.create_spheres()
        self.save()
        return True
    
    def get_mean_distance(self,index1,index2,r=0.01,distance_threshold=0.04):
        # perform one iteration of icp algorithm
        head1 = self.heads[index1]
        head2 = self.heads[index2]

        # sample both array to the same size
        n_sample = int(head1.xyz.shape[0]*r)
        n_1 = head1.xyz.shape[0]
        n_2 = head2.xyz.shape[0] 
        sample_1 = np.random.choice(np.arange(n_1),n_sample)
        sample_2 = np.random.choice(np.arange(n_2),n_sample)  
        A = head1.xyz[sample_1]
        B = head2.xyz[sample_2]
        # get mean distance
        # # get number of dimensions
        # m = sample_1.shape[1]

        # # make points homogeneous, copy them to maintain the originals
        # src = np.ones((m+1,sample_1.shape[0]))
        # dst = np.ones((m+1,sample_2.shape[0]))
        # src[:m,:] = np.copy(sample_1.T)
        # dst[:m,:] = np.copy(sample_2.T)
        # distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)
        distances, _ = icp.nearest_neighbor(A,B)

        # reset
        head2.reset_positions()
        head2.reset_colors()
        return np.mean(distances)
            

    def join_heads_wraper (self, index1, index2):
        distances = []
        con_threshes =[0.02,0.04,0.06,0.08]
        edge_threshes = [10,20,30]
        sigmas =[0.5,1,2,3,4,5,6]
        params = np.array(np.meshgrid(con_threshes,edge_threshes,sigmas)).T.reshape(-1,3)

        for con_thresh,edge_thresh,sigma in params:
                try: # catch bad parameters
                    distance = self.join_heads(index1,index2,con_thresh,edge_thresh,sigma,searching=1)
                except: 
                    distance = 100000
                distances.append(distance)
        min_idx = np.argmin(distances)
        print(min_idx,"min_dist",distances[min_idx],"params:",params[min_idx])
        self.join_heads(index1,index2,params[min_idx][0],params[min_idx][1],params[min_idx][2],searching=0)
        return


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

        return

    def save_spheres(self):
        pickle.dump(self.spheres, open("pickled_head/head_spheres.p", 'wb'))

    def create_spheres(self, sparcity=1.0):
        self.spheres = []
        for head in self.heads:
            if sparcity < 1:
                head.sparsify(sparcity)

                head.create_vpython_spheres(force_sparce=True)
            else:
                head.create_vpython_spheres(force_sparce=False)
            self.spheres += head.spheres

    def save(self, sparcity=1.0):
        self.create_spheres(sparcity)
        self.save_spheres()
        pickle.dump(self, open(f"pickled_head/mhead{self.heads[0].sequence_id}.p", 'wb'))
