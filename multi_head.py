from Procrustes2 import *
import pickle
import HEAD_RECON
import icp
from vpython import *
from SIFT import *
from tqdm import tqdm_notebook as tqdm

class MultiHead():

    def __init__(self):
        self.heads = []

    @classmethod
    def joined_heads(cls, head1, head2):

        this = cls()
        head1.reset_positions()
        head2.reset_positions()
        head1.center()
        head2.center()
        this.heads.append(head1)
        this.heads.append(head2)
        # without search
        this.join_heads(0, 1)
        return this

    def append_head(self, head):
        head.reset_positions()
        head.center()
        head.visible = False
        self.heads.append(head)

    def join_heads(self, index1, index2, SIFT_contrastThreshold=0.02, SIFT_edgeThreshold=14, SIFT_sigma=0.50,
                   searching=False):
        head1 = self.heads[index1]
        head2 = self.heads[index2]

        kp1, kp2, matches = get_matches(head1, head2, SIFT_contrastThreshold=SIFT_contrastThreshold,
                                        SIFT_edgeThreshold=SIFT_edgeThreshold, SIFT_sigma=SIFT_sigma)
        tform, inliers, err, matches = ransac(head1, head2, kp1, kp2, matches)
        head2.transform(tform)
        draw_matches(head1, head2, kp1, kp2, matches, inliers)
        return True
    
    def get_mean_distance(self,index1,index2,r=0.1):
        '''
        r (float): the percentage of pixels in head1 used for calculating distances.
        '''
        # perform one iteration of icp algorithm
        head1 = self.heads[index1]
        head2 = self.heads[index2]

        # sample both array to the same size
        n_sample = int(head1.xyz.shape[0]*r)
        n_1 = head1.xyz.shape[0]
        n_2 = head2.xyz.shape[0]
        sample_1 = np.random.choice(np.arange(n_1), n_sample)
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
        con_threshes = [0.04]  # [0.02,0.04,0.06,0.08]
        edge_threshes = [10]  # [10,20,30]
        sigmas = [1.6]  # [0.5,1,2,3,4,5,6]
        params = np.array(np.meshgrid(con_threshes, edge_threshes, sigmas)).T.reshape(-1, 3)
        num_param = params.shape[0]
        distance = 0
        i = 0
        with tqdm(total=len(params)) as progressbar:
            progressbar.set_description("Searching (head {} and {}), done, Error: {}".format(index1, index2, distance))
            for [con_thresh, edge_thresh, sigma] in params:
                try:  # catch bad parameters
                    distance = self.join_heads(index1, index2, con_thresh, edge_thresh, sigma, searching=True)
                    # print("Searching (head {} and {}), {}/{} done, Error: {}".format(index1,index2,i,num_param,distance))
                    progressbar.update(1)
                except:
                    distance = 100000
                distances.append(distance)
        min_idx = np.argmin(distances)
        print(min_idx, "min_error", distances[min_idx], "params:", params[min_idx])
        self.join_heads(index1, index2, params[min_idx][0], params[min_idx][1], params[min_idx][2], searching=False)
        return


    def icp_transform(self,index1,index2,r=0.05,file_name='pickled_head/after_icp.p'):
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

        # transform head2
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
        print("Saving Completed")

    def create_mesh(self):
        def n2v(p):
            return vec(p[0], p[1], p[2])

        def v2n(v):
            return np.asarray([v.x, v.y, v.z])

        def prj(p):
            return vec(p[0], -p[1], -p[2])

        self.objects = []
        O = np.asarray([0, 0, 0.2])
        o = {'type': "point", 'pos': prj(O), 'radius': "0.01", 'color': vec(1, 0, 0)}
        self.objects.append(o)
        y_range = (-0.3, 0.17)
        y_step = 0.003
        big_angle_step = 10
        small_angle_step = 1

        colors = np.vstack([this_head.rgb for this_head in self.heads])
        points = np.vstack([this_head.xyz for this_head in self.heads])
        filter_s = np.logical_and(points[:, 1] > y_range[0], points[:, 1] < y_range[1])
        points_s = points[filter_s]
        colors_s = colors[filter_s]

        for y in np.arange(y_range[0], y_range[1], step=y_step):
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
                colors_sss = colors_ss[filter_sss]
                vec_from_O_sss = vec_from_O_ss[filter_sss]
                # print(points_ss.size, points_sss.size)
                if points_sss.size > 0:
                    for theta_1 in np.pi * np.arange(0, big_angle_step, step=small_angle_step) / 180:
                        theta = theta_0 + theta_1
                        U = np.cos(theta), 0, np.sin(theta)
                        # o = {'type': "point", 'pos': prj(U+O_for_y), 'radius': "0.01", 'color': vec(1.0,0,1.0)}
                        # self.objects.append(o)
                        filter1 = np.linalg.norm(vec_from_O_sss - np.inner(U, vec_from_O_sss).reshape((-1, 1)) * U,
                                                 axis=1) < 0.003
                        filter2 = np.inner(U, vec_from_O_sss) > 0
                        filter = np.logical_and(filter1, filter2)
                        filtered_points = points_sss[filter]
                        if len(filtered_points) > 0:
                            angles_sss = angles_ss[filter_sss]
                            # print(f"{np.mean(angles_sss[filter])*180/np.pi:.2f}\t{theta*180/np.pi:.2f}" )
                            mean = np.median(filtered_points, axis=0)
                            mean_col = np.mean(colors_sss[filter], axis=0)
                            self.objects.append(
                                {'type': "point", 'pos': prj(mean), 'radius': "0.003", 'color': n2v(mean_col)})
        pickle.dump(self.objects, open(f"pickled_head/head_mesh.p", 'wb'))
