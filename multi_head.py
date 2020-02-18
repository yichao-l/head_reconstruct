from Procrustes2 import *
import pickle
import HEAD_RECON
import icp
from vpython import *
from SIFT import *
from tqdm import tqdm_notebook as tqdm


class Link():
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def add_matches(self, matches):
        self.matches = matches

    def reset(self):
        if hasattr(self, "tform"):
            del self.tform
        if hasattr(self, "matches"):
            del self.matches
        if hasattr(self, "inliers"):
            del self.inliers
        if hasattr(self, "err"):
            del self.err

    def add_ransac_results(self, tform, inliers, err, matches):
        self.tform = tform
        self.inliers = inliers
        self.err = err
        self.matches = matches

    def print(self):
        print("left:", self.left, "\tright:", self.right)
        if hasattr(self, "tform"):
            print(self.tform)
        if hasattr(self, "matches"):
            print(f"# Matches {len(self.matches)}")
        if hasattr(self, "inliers"):
            print(f"# Inliers {len(self.inliers)}")

    def print_short(self):
        print(f"{self.left}-{self.right}, Count={np.sum(self.inliers)}, Err={self.err:.4f}")


class MultiHead():
    def __init__(self):
        self.heads = []
        self.links = []

    @classmethod
    def joined_heads(cls, head1, head2):
        this = cls()
        for head in [head1, head2]:
            this.append(head)
        return this

    def left_eye_deviation(self,sequence_id):
        '''
        param: sequence_id (int) the person's number
        Return: the mean and the deviations of the left eyes in the 3D model for the person with sequence_id.
        '''
        all_eye_ind = [[172576,172581,None,None,None,None,None,None,None,None,None,None,169336,170000,169380],
                       [153974,150143,150124,None,None,None,None,None,None,None,None,None,None,147576,150775],
                       [152103,152745,None,None,None,None,None,None,None,None,None,None,150113,149516,151469],
                       [116190,116191,114265,None,None,None,None,None,None,None,None,None,None,110410,111688]]
        my_eye_ind = all_eye_ind[sequence_id]

        eye_coord = []
        for frame, ind in enumerate(my_eye_ind):
            if ind:
                print(frame, ind)
                ind_xy = np.argwhere(self.heads[frame].xy_mesh == ind)
                eye_coord.append(self.heads[frame].xyz[ind_xy][0][0])
                print("coordinate of the left eye: {} in frame {}".format(self.heads[frame].xyz[ind_xy][0][0], frame))

        eye_coord = np.array(eye_coord)
        mean_coord = np.mean(eye_coord, axis=0)
        print(mean_coord)
        sub_mean = eye_coord - mean_coord
        print(sub_mean)
        distances = np.linalg.norm(sub_mean, axis=1)

        print("mean coordinate: {}. Distance to each points: {}.".format(mean_coord, distances))
        return mean_coord, distances

    def calc_keypoints(self, SIFT_contrastThreshold=0.02, SIFT_edgeThreshold=14, SIFT_sigma=0.50):
        for head in self.heads:
            head.create_keypoints(SIFT_contrastThreshold, SIFT_edgeThreshold, SIFT_sigma)

    def append(self, head):
        head.reset_positions()
        head.visible = False
        head.center()
        self.heads.append(head)

    def head_id_from_frame_id(self, frame_id):
        for i, head in enumerate(self.heads):
            if head.frame_id == frame_id:
                return i
        raise ValueError

    def join_heads(self, frame1, frame2):
        head1 = self.heads[self.head_id_from_frame_id(frame1)]
        head2 = self.heads[self.head_id_from_frame_id(frame2)]
        matches = get_matches(head1, head2)
        tform, inliers, err, matches = ransac(head1, head2, matches)
        head2.transform(tform)
        head1.visible = True
        head2.visible = True
        draw_matches(head1, head2, matches, inliers)
        return True

    def get_next_unpositioned_link(self):
        best_error = 1
        best_link_index = None
        any_head_positioned = False
        for head in self.heads:
            if head.visible:
                any_head_positioned = True
        print("any_head_positioned", any_head_positioned)

        for i, link in enumerate(self.links):
            head_left = self.heads[self.head_id_from_frame_id(link.left)]
            head_right = self.heads[self.head_id_from_frame_id(link.right)]
            # if this is the best error so far and either (one of the 2 heads is visible, or none of all the heads is visible)
            if link.err < best_error and ((head_left.visible and not head_right.visible) or (
                    (not head_left.visible) and head_right.visible) or not any_head_positioned):
                best_error = link.err
                best_link_index = i
        return best_link_index, best_error

    def reset_all_head_positions(self):
        for head in self.heads:
            head.reset_positions()
            head.reset_colors()
            head.visible = False

    def ransac_from_link(self, link):
        head1 = self.heads[self.head_id_from_frame_id(link.right)]
        head2 = self.heads[self.head_id_from_frame_id(link.left)]
        if not hasattr(link, "matches"):
            matches = get_matches(head1, head2)
            tform, inliers, err, matches = ransac(head1, head2, matches)
            link.add_ransac_results(tform, inliers, err, matches)
        return link

    def transform_from_link(self, link):
        head1 = self.heads[self.head_id_from_frame_id(link.right)]
        head2 = self.heads[self.head_id_from_frame_id(link.left)]
        # if not hasattr(link, "matches"):
        link = self.ransac_from_link(link)

        xyz1, xyz2, matches = get_xyz_from_matches(head1, head2, link.matches)

        head1.keypoints = xyz1[link.inliers]
        head1.keypoints_clr = [1, 0, 0]
        head2.keypoints = xyz2[link.inliers]
        head2.keypoints_clr = [0, 1, 0]

        d, Z, tform21 = procrustes(xyz1[link.inliers], xyz2[link.inliers], scaling=False, reflection='best')

        if head2.visible and not head1.visible:
            d, Z, tform12 = procrustes(xyz2[link.inliers], xyz1[link.inliers], scaling=False, reflection='best')
            head1.transform(tform12)



        else:
            head2.transform(tform21)
        head1.visible = True
        head2.visible = True
        draw_matches(head1, head2, link.matches, link.inliers)
        return link

    def get_mean_distance(self, index1, index2, r=0.1):
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

    def icp_transform(self, frame1, frame2, r=0.05, file_name='pickled_head/after_icp.p'):
        '''
        param:
        r (float): sampleing rate for head1 
        file_name (string): file name of combined spheres
        '''
        # perform one iteration of icp algorithm
        head1 = self.heads[self.headshead_id_from_frame_id(frame1)]
        head2 = self.heads[self.headshead_id_from_frame_id(frame2)]

        # sample both array to the same size
        n_sample = int(head1.xyz.shape[0] * r)
        n_1 = head1.xyz.shape[0]
        n_2 = head2.xyz.shape[0]
        sample_1 = np.random.choice(np.arange(n_1), n_sample)
        sample_2 = np.random.choice(np.arange(n_2), n_sample)
        T, distance, ite = icp.icp(head1.xyz[sample_1], head2.xyz[sample_2])

        # transform head2
        head2.transform_homo(T)
        return

    def save_spheres(self):
        pickle.dump(self.spheres, open("pickled_head/head_spheres.p", 'wb'))

    def create_spheres(self, sparcity=1.0):
        self.spheres = []
        for head in self.heads:
            if head.visible:
                if sparcity < 1:
                    head.sparsify(sparcity)
                    head.create_vpython_spheres(force_sparce=True)
                else:
                    head.create_vpython_spheres(force_sparce=False)
                self.spheres += head.spheres
        self.save_spheres()

    def save(self):
        for head in self.heads:
            if hasattr(head, 'kp'):
                head.kp = [(point.pt, point.size, point.angle, point.response, point.octave,
                            point.class_id) for point in head.kp]
        for link in self.links:
            if hasattr(link, 'matches'):
                link.matches = [(match.distance, match.imgIdx, match.queryIdx, match.trainIdx) for match in
                                link.matches]

        pickle.dump(self, open(f"pickled_head/mhead{self.heads[0].sequence_id}.p", 'wb'))
        for head in self.heads:
            if hasattr(head, 'kp'):
                head.kp = [cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2],
                                        _response=point[3], _octave=point[4], _class_id=point[5]) for point in head.kp]

        for link in self.links:
            if hasattr(link, 'matches'):
                link.matches = [cv2.DMatch(_distance=match[0], _imgIdx=match[1], _queryIdx=match[2], _trainIdx=match[3])
                                for match in link.matches]

        print("Saving Completed")

    def create_mesh(self):
        def n2v(p):
            return vec(p[0], p[1], p[2])

        def v2n(v):
            return np.asarray([v.x, v.y, v.z])

        def prj(p):
            return vec(p[0], -p[1], -p[2])

        self.objects = []
        O = np.asarray([0, 0, 0.05])

        y_range = (-0.35, 0.20)
        o = {'type': "point", 'pos': prj(O + np.asarray([0, 1, 0]) * y_range[0]), 'radius': "0.01",
             'color': vec(0, 1, 0)}
        self.objects.append(o)
        o = {'type': "point", 'pos': prj(O + np.asarray([0, 1, 0]) * y_range[1]), 'radius': "0.01",
             'color': vec(1, 0, 0)}
        self.objects.append(o)

        y_step = 0.003
        big_angle_step = 20
        small_angle_step = 2

        colors = np.vstack([this_head.rgb for this_head in self.heads if this_head.visible])
        points = np.vstack([this_head.xyz for this_head in self.heads if this_head.visible])

        filter_s = np.logical_and(points[:, 1] > y_range[0], points[:, 1] < y_range[1])
        points_s = points[filter_s]
        colors_s = colors[filter_s]
        y_range = np.arange(y_range[0], y_range[1], step=y_step)
        with tqdm(total=y_range.size) as progressbar:
            for y in y_range:
                O_for_y = np.asarray([0, y, 0]) + O
                filter_ss = np.logical_and(points_s[:, 1] > y - 2 * y_step, points_s[:, 1] < y + 2 * y_step)
                points_ss = points_s[filter_ss]
                colors_ss = colors_s[filter_ss]
                vec_from_O_ss = points_ss - O_for_y
                angles_ss = np.mod(np.angle(vec_from_O_ss[:, 0] + 1j * vec_from_O_ss[:, 2]), 2 * np.pi)
                for theta_0 in np.pi * np.arange(0., 360, step=big_angle_step) / 180:
                    filter_sss = np.logical_and(
                        angles_ss > theta_0 - np.pi * small_angle_step / 180,
                        angles_ss < theta_0 + (big_angle_step + small_angle_step) * np.pi / 180)
                    points_sss = points_ss[filter_sss]
                    colors_sss = colors_ss[filter_sss]
                    vec_from_O_sss = vec_from_O_ss[filter_sss]
                    # print(points_ss.size, points_sss.size)
                    if points_sss.size > 0:
                        for theta_1 in np.pi * np.arange(0, big_angle_step, step=small_angle_step) / 180:
                            theta = theta_0 + theta_1
                            U = np.cos(theta), 0, np.sin(theta)
                            filter1 = np.linalg.norm(vec_from_O_sss - np.inner(U, vec_from_O_sss).reshape((-1, 1)) * U,
                                                     axis=1) < 0.003
                            filter2 = np.inner(U, vec_from_O_sss) > 0
                            filter = np.logical_and(filter1, filter2)
                            filtered_points = points_sss[filter]
                            if len(filtered_points) > 0:
                                angles_sss = angles_ss[filter_sss]
                                # print(f"{np.mean(angles_sss[filter])*180/np.pi:.2f}\t{theta*180/np.pi:.2f}" )
                                mean = np.mean(filtered_points, axis=0)
                                mean = np.inner(U, mean - O_for_y) * np.asarray(U) + O_for_y

                                mean_col = np.mean(colors_sss[filter], axis=0)
                                self.objects.append(
                                    {'type': "point", 'pos': prj(mean), 'radius': "0.003", 'color': n2v(mean_col)})
                progressbar.set_description(f"Scanning{np.arange(0, big_angle_step, step=small_angle_step).size}")
                progressbar.update(1)

        pickle.dump(self.objects, open(f"pickled_head/head_mesh.p", 'wb'))
        print("completed")
