import pickle
import icp
from vpython import *
from SIFT import *
from tqdm.autonotebook import tqdm
from refine import *
from link import Link
import numpy as np


class MultiHead():
    def __init__(self):
        self.heads = []
        self.links = []

    @classmethod
    def joined_heads(cls, head1, head2):
        '''
        Class method that initiate MultiHead object, with head1, head2 being two SingleHead objects.
        '''
        this = cls()
        for head in [head1, head2]:
            this.append(head)
        return this

    @classmethod
    def load_from_pickle(cls, sequence_id):

        '''
        :param data_file:  file to load from, default name is the default file used for saving
        :return: object of  threeD_head class
        '''
        print(f"Loading Sequence {sequence_id}...", end="")
        data_file = f"pickled_head/mhead{sequence_id}.p"
        try:

            with open(data_file, 'rb') as file_object:
                raw_data = file_object.read()
            this = pickle.loads(raw_data)
        except:
            raise FileExistsError(f'{data_file} could not be found, create {data_file} by using .save() first ')

        for head in this.heads:
            if hasattr(head, 'kp'):
                head.kp = [cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2],
                                        _response=point[3], _octave=point[4], _class_id=point[5]) for point in head.kp]

        for link in this.links:
            if hasattr(link, 'matches'):
                link.matches = [cv2.DMatch(_distance=match[0], _imgIdx=match[1], _queryIdx=match[2], _trainIdx=match[3])
                                for match in link.matches]

        print("done")
        return this

    @classmethod
    def create_from_heads(cls, list_of_heads, first=0, last=14):
        '''

        :param list_of_heads:
        :param first:
        :param last:
        :return:
        '''
        print("creating mhead object from a list of SingleHead objects")
        list_of_heads[first].reset_positions()
        list_of_heads[first].reset_colors()
        list_of_heads[first + 1].reset_positions()
        list_of_heads[first + 1].reset_colors()
        this = MultiHead.joined_heads(list_of_heads[first], list_of_heads[first + 1])
        this.links.append(Link(left=list_of_heads[first + 1].frame_id, right=list_of_heads[first].frame_id))
        for i in range(first + 2, last + 1):
            this.links.append(Link(left=list_of_heads[i].frame_id, right=list_of_heads[i - 1].frame_id))
            list_of_heads[i].reset_positions()
            list_of_heads[i].reset_colors()
            this.append(list_of_heads[i])
            if i == last:
                this.links.append(Link(left=list_of_heads[first].frame_id, right=list_of_heads[i].frame_id))
        return this

    def left_eye_deviation(self):
        '''
        param: sequence_id (int) the person's number
        Return: the mean and the deviations of the left eyes in the 3D model for the person with sequence_id.
        '''
        all_eye_ind = [
            [172576, 172581, None, None, None, None, None, None, None, None, None, None, 169336, 170000, 169380],
            [153974, 150143, 150124, None, None, None, None, None, None, None, None, None, None, 147576, 150775],
            [152103, 152745, None, None, None, None, None, None, None, None, None, None, 150113, 149516, 151469],
            [116190, 116191, 114265, None, None, None, None, None, None, None, None, None, None, 110410, 111688]]
        my_eye_ind = all_eye_ind[self.heads[0].sequence_id - 1]

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
        print(f"mean distance: {np.mean(distances)}")
        return np.mean(distances)

    def calc_all_sift_keypoints(self, SIFT_contrastThreshold=0.02, SIFT_edgeThreshold=10, SIFT_sigma=0.50):
        '''
        Calculate the keypoints for all the SingleHead object in self.heads.
        '''
        for head in self.heads:
            head.create_keypoints(SIFT_contrastThreshold, SIFT_edgeThreshold, SIFT_sigma)

    def calc_all_sift_transforms(self):
        '''
        calculates SIFT transforms for each of the heads
        :return:
        '''
        for link in self.links:
            link.reset()
            self.ransac_from_link(link)

    def append(self, head):
        head.reset_positions()
        head.visible = False
        head.center()
        self.heads.append(head)

    def head_id_from_frame_id(self, frame_id):
        '''
        Map from actual frame id to the index of this list of heads.
        '''
        for i, head in enumerate(self.heads):
            if head.frame_id == frame_id:
                return i
        raise ValueError

    # def join_heads(self, frame1, frame2):
    #     '''
    #     :param frame1:
    #     :param frame2:
    #     :return:
    #     '''
    #     head1 = self.heads[self.head_id_from_frame_id(frame1)]
    #     head2 = self.heads[self.head_id_from_frame_id(frame2)]
    #     matches = get_matches(head1, head2)
    #     tform, inliers, err, matches = estimate_transform(head1, head2, matches)
    #     head2.transform(tform)
    #     head1.visible = True
    #     head2.visible = True
    #     draw_matches(head1, head2, matches, inliers)
    #     return True

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

    def reset_all_head_colors(self):
        for head in self.heads:
            head.reset_colors()

    def ransac_from_link(self, link):
        '''
        Computes the matching keypoints and estimate the Procrustes transformation,
        then stores the tranformation result as link attributes. 
        '''
        head1 = self.heads[self.head_id_from_frame_id(link.right)]
        head2 = self.heads[self.head_id_from_frame_id(link.left)]
        if not hasattr(link, "matches"):
            matches = get_matches(head1, head2)
            inliers_all_points, coverage_all_points, inliers_matches, err_matches, matches = estimate_transform(head1,
                                                                                                                head2,
                                                                                                                matches)
            link.add_ransac_results(inliers_all_points, coverage_all_points, inliers_matches, err_matches, matches)
        return link

    def sift_transform_from_link(self, link, right_to_left, all_points=True):

        head1 = self.heads[self.head_id_from_frame_id(link.right)]
        head2 = self.heads[self.head_id_from_frame_id(link.left)]
        # if not hasattr(link, "matches"):
        link = self.ransac_from_link(link)
        xyz1, xyz2, matches = get_xyz_from_matches(head1, head2, link.matches)
        if all_points:
            inliers = link.inliers_all_points
        else:
            inliers = link.inliers_matches
        head1.keypoints = xyz1[inliers]
        head2.keypoints = xyz2[inliers]

        head1.keypoints_clr = [1, 0, 0]
        head2.keypoints_clr = [0, 1, 0]

        if right_to_left:
            d, Z, tform12 = procrustes(xyz2[inliers], xyz1[inliers])
            t = tform12['rotation']
            print(t)
            head1.transform(tform12)
        else:
            d, Z, tform21 = procrustes(xyz1[inliers], xyz2[inliers])
            t = tform21['rotation']
            print(t)

            head2.transform(tform21)
        draw_matches(head1, head2, link.matches, inliers)
        return link

    def icp_transform_from_link(self, link, right_to_left):
        '''

        :param link: the Link object between the two heads that are to be ICP-ed
        :param right_to_left: in which direction is the ICP happening? Left to right or rightto left
        :return: link, while correct head has been transformed
        '''

        if right_to_left:
            self.icp_transform(link.left, link.right,
                               max_iterations=40)
        else:
            self.icp_transform(link.right, link.left,
                               max_iterations=40)
        return link

    def refine_transform_from_link(self, link, right_to_left, angle_over_range=False, cartesian_over_range=False,
                                   filter=None):
        '''
        calls the refine_6D fucntions, passing on all parameters, selecting whether head A moves to head B or the other way round
        '''
        if right_to_left:
            filter, score = refine_6D(self, A=link.left, B=link.right, angle_over_range=angle_over_range,
                                      pos_over_range=cartesian_over_range, filter=filter)
        else:
            filter, score = refine_6D(self, A=link.right, B=link.left, angle_over_range=angle_over_range,
                                      pos_over_range=cartesian_over_range, filter=filter)
        return filter, score

    def all_transforms_from_link(self, link):
        '''
        :param link: The link for which all transforms are to be performed
        :return: the link itslef, unmodified.

        Identifies which head is to be transformed ( the one that is not visible)
        Transforms the head that is not visible and makes it visble (headx.visible = True)

        '''
        head1 = self.heads[self.head_id_from_frame_id(link.right)]
        head2 = self.heads[self.head_id_from_frame_id(link.left)]

        right_to_left = head2.visible and not head1.visible
        if right_to_left:
            self.last_head_id = self.head_id_from_frame_id(link.right)
        else:
            self.last_head_id = self.head_id_from_frame_id(link.left)

        # perform the transformation based on the calculated SIFT matches:
        self.sift_transform_from_link(link, right_to_left)
        # perform the ICP transformation
        self.icp_transform_from_link(link, right_to_left)

        # perform the refine operation
        filter = None
        # first peform a
        filter, score = self.refine_transform_from_link(link, right_to_left, angle_over_range=True,
                                                        cartesian_over_range=True,
                                                        filter=filter)
        filter, score = self.refine_transform_from_link(link, right_to_left, angle_over_range=True,
                                                        cartesian_over_range=True,
                                                        filter=filter)
        last_score = 0
        before_last_score = 0
        while score > last_score or score > before_last_score:
            before_last_score = last_score
            last_score = score
            filter, score = self.refine_transform_from_link(link, right_to_left, filter=filter)
        self.icp_transform_from_link(link, right_to_left)
        head1.visible = True
        head2.visible = True
        return link

    def icp_transform(self, frame1, frame2, r=0.05, max_iterations=1):
        '''
        param:
        r (float): sampling rate for head1
        file_name (string): file name of combined spheres
        '''
        # perform one iteration of icp algorithm
        head1 = self.heads[self.head_id_from_frame_id(frame1)]
        head2 = self.heads[self.head_id_from_frame_id(frame2)]

        # sample both array to the same size
        n_sample = int(head1.xyz.shape[0] * r)
        n_1 = head1.xyz.shape[0]
        n_2 = head2.xyz.shape[0]
        sample_1 = np.random.choice(np.arange(n_1), n_sample)
        sample_2 = np.random.choice(np.arange(n_2), n_sample)
        T, distance, ite = icp.icp(head1.xyz[sample_1], head2.xyz[sample_2], max_iterations=max_iterations)

        # transform head2
        head2.transform_homo(T)
        return

    def create_spheres(self, sparcity=1.0, name=None):
        self.spheres = []
        for head in self.heads:
            if head.visible:
                if sparcity < 1:
                    head.sparsify(sparcity)
                    head.create_vpython_spheres(force_sparce=True)
                else:
                    head.create_vpython_spheres(force_sparce=False)
                self.spheres += head.spheres
        pickle.dump((self.spheres, name), open("pickled_head/head_spheres.p", 'wb'))

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

    def create_png_of_spheres(self, sparcity, name, alpha=0):
        import subprocess
        self.create_spheres(sparcity=sparcity, name=name)
        subprocess.call(["python", "gui.py", "save_only", "alpha", f"{int(alpha)}"])

    def show_spheres(self, sparcity, name=None, alpha=0):
        import subprocess
        self.create_spheres(sparcity=sparcity, name=name)
        subprocess.run(["python", "gui.py", "alpha", f"{int(alpha)}"])

    def create_png_series(self):

        self.reset_all_head_colors()
        for head in self.heads:
            head.visible = False
        sequence = self.heads[0].sequence_id
        for link_idx in range(14):  # iterate through the links between heads
            link = self.links[link_idx]
            head1 = self.heads[self.head_id_from_frame_id(link.right)]
            head2 = self.heads[self.head_id_from_frame_id(link.left)]
            right_to_left = head2.visible and not head1.visible
            if right_to_left:
                self.last_head_id = self.head_id_from_frame_id(link.right)
            else:
                self.last_head_id = self.head_id_from_frame_id(link.left)

            head1.visible = True
            head2.visible = True

            self.reset_all_head_colors()
            self.heads[self.last_head_id].paint([.2, .2, 1])

            if sequence in [2, 4]:
                alpha = -10 - link_idx * 360 / 15
            else:
                alpha = 10 + link_idx * 360 / 15
            self.create_png_of_spheres(sparcity=0.5, name=f"Seq_{sequence}_{link_idx}", alpha=alpha)
        self.reset_all_head_colors()
        link_idx = 14
        if sequence in [2, 4]:
            alpha = -10 - link_idx * 360 / 15
        else:
            alpha = 10 + link_idx * 360 / 15
        self.create_png_of_spheres(sparcity=0.5, name=f"Seq_{sequence}_all", alpha=alpha)

    def create_mesh(self, name):
        def n2v(p):
            return vec(p[0], p[1], p[2])

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

        pickle.dump((self.objects,name) , open(f"pickled_head/head_mesh.p", 'wb'))
        print("completed")
