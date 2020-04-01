import cv2
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from procrustes import *
from icp import nearest_neighbor
import os


def get_matches(head1, head2):
    # find K nearest matches ( in terms of descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(head1.des, head2.des, k=2)
    # unfoled the list
    # matches = [val for sublist in matches for val in sublist]
    
    # using one nearest neighbor
    good_without_list = [sublist[0] for sublist in matches]

    # use the ratio test
    # good_without_list=[]
    # for m, n in matches:
    #     if m.distance < 0.9 * n.distance:
    #     # good.append([m])
    #         good_without_list.append(m)
    return good_without_list

def remove_height_variation_from_matches(head1, head2, matches):
    max_variation_y_dimension = 15
    matches = [m for m in matches if
               (abs(head1.kp[m.queryIdx].pt[1] - head2.kp[m.trainIdx].pt[1]) < max_variation_y_dimension)]
    return matches

def get_xyz_from_matches(head1, head2, matches):
    '''
    Get the sets of 3D points of the respective heads, corresponding to the matches.
    :param head1: a head object.
    :param head2: a head object.
    :param matches: a list of DMatch objects.
    :return: xyz1/2 the sets of 3d points of the respective heads, corresponding to the matches.
    '''
    matches = remove_height_variation_from_matches(head1, head2, matches)

    pts1 = np.float32([head1.kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    xy1 = np.round(pts1).astype("int").reshape(-1, 2)
    xyindex1 = xy1[:, 1] * 640 + xy1[:, 0]
    indices1 = [np.argwhere(head1.xy_mesh == ind) for ind in xyindex1]
    filter1 = [len(ind) > 0 for ind in indices1]

    pts2 = np.float32([head2.kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    xy2 = np.round(pts2).astype("int").reshape(-1, 2)
    xyindex2 = xy2[:, 1] * 640 + xy2[:, 0]
    indices2 = [np.argwhere(head2.xy_mesh == ind) for ind in xyindex2]
    filter2 = [len(ind) > 0 for ind in indices2]

    filter = np.asarray(filter1) & np.asarray(filter2)
    indices1 = np.asarray(indices1)[filter]
    indices2 = np.asarray(indices2)[filter]
    xyz1 = np.asarray([head1.xyz[ind[0][0]] for ind in indices1])
    xyz2 = np.asarray([head2.xyz[ind[0][0]] for ind in indices2])
    return xyz1, xyz2, matches


def estimate_transform(head1, head2, matches):
    '''
    Estimate the transform from a set of match objects and two head objects using the RANSAC
    technique and the Procrutes algorithm.
    param head1 (SingleHead): one head as the anchor.
    param head2 (SingleHead): another head to transform from.
    param matches (list(DMatch)): a list of match points from two heads.
    '''
    kp_xyz1, kp_xyz2, matches = get_xyz_from_matches(head1, head2, matches)

    # RANSAC parameters
    max_dist_matches = 0.007
    max_dist_coverage = 0.01

    no_iterations = 5000 # number of ransac iterations.
    no_iterations_all_points = 0 # number of ransac that does coverage calculation.
    min_num_matches = 6 # minimum number of matches for procrustes
    sample_thresh = 0.6 # threshold for sampling match matches
    best_count_coverage = 0 # record the best number of point covered by a transformation
    best_count_matches = 0 # record the best number of maches covered by a tranformation
    best_pct_coverage = -1 # the coverage value that we want to maximize
    best_err_matches = 1000  # the inlier matches that we want to maximize
    best_sample_matches_cvg = []
    best_sample_matches_mchs = []

    # store the points from both frames for coverage calculation
    l = head2.xyz.shape[0]
    filter2 = np.random.random((l)) < 0.1
    xyz1 = head1.xyz
    xyz2 = head2.xyz[filter2]
    temp_best_tform = None
    with tqdm(total=no_iterations) as progressbar:
        # total RANSAC iterations
        for j in range(no_iterations):
            kp_sample = np.random.rand(kp_xyz2.shape[0]) > sample_thresh  # get random sample set
            progressbar.update(1)  # tqdm
            temp_best_count = -1
            if np.sum(kp_sample) >= min_num_matches:  # enough points to unambiguously define transformation?
                _, _, tform = procrustes(kp_xyz1[kp_sample], kp_xyz2[kp_sample])
                R, c, t = tform['rotation'], tform['scale'], tform['translation']
                dist = np.linalg.norm(kp_xyz2.dot(c * R) + t - kp_xyz1, axis=1)
                inliers = dist < max_dist_matches
                
                # if there are more inliers for this transformation.
                if (np.sum(inliers) > best_count_matches):
                    best_sample_matches_mchs = kp_sample.copy()
                    best_count_matches = np.sum(inliers)
                    err = np.sqrt(np.var(dist) / (np.sum(kp_sample) - min_num_matches))
                    best_err_matches = err
                    # update the progress bar
                    progressbar.set_description(
                        f"Head {head1.frame_id} & {head2.frame_id} :cnt:{best_count_matches:.0f} err:{best_err_matches:.4f} cov:{100 * best_pct_coverage :.2f}%")

                # number of ransac that does coverage calculation
                if j < no_iterations_all_points:
                    R, c, t = tform['rotation'], tform['scale'], tform['translation']
                    xyz2_trans = xyz2.dot(c * R) + t
                    distances, indices = nearest_neighbor(xyz2_trans, xyz1)
                    count_all_points = np.sum(distances < max_dist_coverage)
                    # see if the new transformation has a better coverage
                    if count_all_points > best_count_coverage:
                        best_count_coverage = count_all_points
                        best_pct_coverage = best_count_coverage / xyz2.shape[0]
                        best_sample_matches_cvg = kp_sample.copy()
                        progressbar.set_description(
                            f"Head {head1.frame_id} & {head2.frame_id} :cnt:{best_count_matches:.0f} err:{best_err_matches:.4f} cov:{100 * best_pct_coverage :.2f}%")

    return best_sample_matches_cvg, best_pct_coverage, best_sample_matches_mchs, best_err_matches, matches

def draw_matches(head1, head2, matches, inliers):
    '''
    Generate the images with the match points between two frames
    '''
    images_dir="images"
    if not os.path.isdir(images_dir):
        os.mkdir(images_dir)
    # set the background color
    head1.background_color=np.array([0,0,0.99])
    head2.background_color=np.array([0,0,0.99])
    img1 = cv2.cvtColor((head1.get_filtered_image() * 256).astype("uint8"), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor((head2.get_filtered_image() * 256).astype("uint8"), cv2.COLOR_BGR2RGB)
    # parameters for drawing
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(0, 0, 255),
                       matchesMask=inliers.ravel().tolist(),
                       flags=0)
    # generat the images along with the matches indication
    img3 = cv2.drawMatches(img1, head1.kp, img2, head2.kp, matches1to2=matches,
                           outImg=None, **draw_params)
    # label parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (0, 40)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2
    # add text label onto the image
    cv2.putText(img3, f"Sequence {head1.sequence_id}, frames {head1.frame_id} & {head2.frame_id}",
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    # save the image
    file_name=f"Seq_{head1.sequence_id}_frames_SIFT_{head1.frame_id}_{head2.frame_id}.png"
    full_path= os.path.join(images_dir, file_name)
    cv2.imwrite(full_path, img3)


def get_descriptors(img, SIFT_contrastThreshold, SIFT_edgeThreshold, SIFT_sigma):
    '''
    Get a set of SIFT parameters given a color image in an array and three SIFT parameters.
    param:
    img (array): colored image
    The others are all SIFT parameters
    '''
    # scale pixel values the image
    img = img * 256
    img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_BGR2RGB)
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=SIFT_contrastThreshold, edgeThreshold=SIFT_edgeThreshold,
                                       sigma=SIFT_sigma)
    kp, des = sift.detectAndCompute(img, None)
    return kp, des