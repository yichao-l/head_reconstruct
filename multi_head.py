
from Procrustes2 import *
import  HEAD_RECON

class MultiHead():
    def __init__(self):
        self.heads=[]


    @classmethod
    def joined_heads(cls, head1, head2):
        cls.__init__(cls)
        cls.heads.append(head1)
        cls.heads.append(head2)
        cls.join_heads(cls,0,1)
        return cls

    def join_heads(self, index1, index2):
        head1= self.heads[index1]
        head2= self.heads[index2]

        img1, path1 = head1.get_filtered_image()
        img2, path2 = head2.get_filtered_image()
        kp1, des1 = HEAD_RECON.estimate_frame_transform.get_descriptors(path1)
        kp2, des2 = HEAD_RECON.estimate_frame_transform.get_descriptors(path2)
        good_without_list = HEAD_RECON.estimate_frame_transform.get_matched_points(path1,kp1,des1,path2,kp2,des2,0.8)

        cleaned_match = HEAD_RECON.estimate_frame_transform.clean_matches(kp1,path1,kp2,path2,good_without_list)

        head1.reset_colors()
        head2.reset_colors()
        # code below can be used to create
        # head1.paint([1, 0, 0])
        # head2.paint([1,1,0])
        head1.reset_positions()
        head2.reset_positions()
        head1.center()
        head2.center()

        matches = cleaned_match
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        xy1 = np.round(pts1).astype("int").reshape(-1, 2)
        xyindex1 = xy1[:, 1] * 640 + xy1[:, 0]
        xyz1 = head1.xyz_unfiltered[xyindex1]

        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        xy2 = np.round(pts2).astype("int").reshape(-1, 2)
        xyindex2 = xy2[:, 1] * 640 + xy2[:, 0]
        xyz2 = head2.xyz_unfiltered[xyindex2]

        for i in xyindex1:
            pos = np.where(head1.xy_mesh == i)
            head1.rgb[pos] = [0, 1, 0]
        head1.create_vpython_spheres()
        head1.save()

        for i in xyindex2:
            pos = np.where(head2.xy_mesh == i)
            head2.rgb[pos] = [0, 0, 1]
            head2.xyz

        # todo, make list of points unique

        A = head1.xyz_unfiltered[xyindex1][1:] - head1.center_pos
        B = head2.xyz_unfiltered[xyindex2][1:] - head2.center_pos

        # todo, set scaling to false
        d, Z, tform = procrustes(A, B, scaling=True, reflection='best')

        R, c, t = tform['rotation'], tform['scale'], tform['translation']

        head2.transform(R, c, t)
        head1.create_vpython_spheres()
        head2.create_vpython_spheres()
        # todo create spheres as part ofthe mulit-head object
        head1.spheres = head1.spheres + head2.spheres
        head1.save()

