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

        # with search
        # this.join_heads_wraper(0,1,SIFT_contrastThreshold=0.04,SIFT_edgeThreshold=10,SIFT_sigma=1.6)
        return this

    def append_head(self, head):
        head.reset_positions()
        head.center()
        # head1.center()
        # head2.center()
        self.heads.append(head)

    def join_heads(self, index1, index2, SIFT_contrastThreshold=0.02, SIFT_edgeThreshold=14, SIFT_sigma=0.50,
                   searching=False):

        head1 = self.heads[index1]
        head2 = self.heads[index2]

        img1, path1 = head1.get_filtered_image()
        img2, path2 = head2.get_filtered_image()
        kp1, des1 = get_descriptors(img1, SIFT_contrastThreshold, SIFT_edgeThreshold, SIFT_sigma)
        kp2, des2 = get_descriptors(img2, SIFT_contrastThreshold, SIFT_edgeThreshold, SIFT_sigma)
        print(des2.shape)
        rad = 5

        kp1, des1 = head1.remove_edge_points(kp1, des1, rad=rad)
        kp2, des2 = head2.remove_edge_points(kp2, des2, rad=rad)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        matches = [val for sublist in matches for val in sublist]
        # remove big variation in y
        matches = [m for m in matches if (abs(kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1]) < 15)]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        xy1 = np.round(pts1).astype("int").reshape(-1, 2)
        xyindex1 = xy1[:, 1] * 640 + xy1[:, 0]
        indices1 = [np.argwhere(head1.xy_mesh == ind) for ind in xyindex1]
        filter1 = [len(ind) > 0 for ind in indices1]
        xyz1 = head1.xyz_unfiltered[xyindex1]
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        xy2 = np.round(pts2).astype("int").reshape(-1, 2)
        xyindex2 = xy2[:, 1] * 640 + xy2[:, 0]
        indices2 = [np.argwhere(head2.xy_mesh == ind) for ind in xyindex2]
        filter2 = [len(ind) > 0 for ind in indices2]
        filter = np.asarray(filter1) & np.asarray(filter2)
        indices1 = np.asarray(indices1)[filter]
        indices2 = np.asarray(indices2)[filter]
        xyz1 = np.asarray([head1.xyz[ind[0][0]] for ind in indices1])
        xyz2 = np.asarray([head2.xyz[ind[0][0]] for ind in indices2])

        max_dist = 0.015
        best_count = 0
        best_inliers = []
        No_Iterations = 2500
        with tqdm(total=No_Iterations) as progressbar:
            progressbar.set_description("Best Count {} ".format(best_count))
            for j in range(No_Iterations):
                inliers = np.random.rand(xyz2.shape[0]) > 0.05
                progressbar.set_description("Best Count {} ".format(best_count))
                progressbar.update(1)
                for i in range(20):
                    if np.sum(inliers) >= 5:
                        try:
                            d, Z, tform = procrustes(xyz1[inliers], xyz2[inliers], scaling=False, reflection='best')
                            R, c, t = tform['rotation'], tform['scale'], tform['translation']
                            dist = np.linalg.norm(xyz2.dot(c * R) + t - xyz1, axis=1)
                            last_inliers = inliers.copy()
                            inliers = dist < max_dist
                            if (np.sum(inliers) > best_count):
                                best_count = np.sum(inliers)
                                best_inliers = inliers.copy()
                                best_tform = tform
                            if np.all(last_inliers == inliers):
                                break
                        except:
                            pass
                    else:
                        break
        head1.keypoints = xyz1[best_inliers]
        head1.keypoints_clr = [1, 0, 0]
        head2.keypoints = xyz2[best_inliers]
        head2.keypoints_clr = [0, 1, 0]

        if best_count < 6:
            raise ValueError('Could not match')

        R, c, t = best_tform['rotation'], best_tform['scale'], best_tform['translation']
        head2.transform(R, c, t)

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None, flags=2)

        img1 = cv2.cvtColor((img1 * 256).astype("uint8"), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor((img2 * 256).astype("uint8"), cv2.COLOR_BGR2RGB)

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, [match for i, match in enumerate(matches) if best_inliers[i]],
                               None, **draw_params)
        cv2.imwrite("des_match_cleaned.png", img3)

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
        con_threshes = [0.02]  # [0.02,0.04,0.06,0.08]
        edge_threshes = [14]  # [10,20,30]
        sigmas = [0.5]  # [0.5,1,2,3,4,5,6]
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
                            mean = np.mean(filtered_points, axis=0)
                            mean_col = np.mean(colors_sss[filter], axis=0)
                            self.objects.append(
                                {'type': "point", 'pos': prj(mean), 'radius': "0.003", 'color': n2v(mean_col)})
        pickle.dump(self.objects, open(f"pickled_head/head_mesh.p", 'wb'))
