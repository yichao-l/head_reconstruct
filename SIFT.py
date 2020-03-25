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
    matches = [val
               for sublist in matches
               for val in sublist]

    # matches = [sublist[0] for sublist in matches]
    print(len(matches))
    return matches


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
    max_dist_all_points = 0.01

    no_iterations = 10000
    no_iterations_all_points = 0
    min_num_inliers = 6
    sample_thresh = 0.6

    best_count_all_points = 0
    best_count_matches = 0
    best_coverage_all_points = -1
    best_err_matches = 1000  #
    best_inliers_all_points = []
    best_kp_sample_matches = []

    l = head2.xyz.shape[0]
    filter2 = np.random.random((l)) < 0.1
    xyz1 = head1.xyz
    xyz2 = head2.xyz[filter2]
    temp_best_tform = None
    with tqdm(total=no_iterations) as progressbar:
        for j in range(no_iterations):
            kp_sample = np.random.rand(kp_xyz2.shape[0]) > sample_thresh  # get random sample set
            # kp_sample = np.random.rand(kp_xyz2.shape[0]) > np.random.rand()
            progressbar.update(1)  # tqdm
            temp_best_count = -1
            if np.sum(kp_sample) >= min_num_inliers:  # enough points to unambiguously define transformation?
                _, _, tform = procrustes(kp_xyz1[kp_sample], kp_xyz2[kp_sample])
                R, c, t = tform['rotation'], tform['scale'], tform['translation']
                dist = np.linalg.norm(kp_xyz2.dot(c * R) + t - kp_xyz1, axis=1)
                # last_kp_sample = kp_sample.copy()
                inliers = dist < max_dist_matches
                # print(np.sum(inliers))
                # if np.sum(inliers) > min_num_inliers:
                #     _, _, tform_inliers = procrustes(kp_xyz1[inliers], kp_xyz2[inliers])
                #     R, c, t = tform_inliers['rotation'], tform_inliers['scale'], tform_inliers['translation']
                #     dist = np.linalg.norm(kp_xyz2.dot(c * R) + t - kp_xyz1, axis=1)
                #     err = np.sqrt(np.var(dist) / (np.sum(last_kp_sample) - min_num_inliers))
                # else:
                #     # if np.sum(last_kp_sample) > min_num_inliers and i>0:
                #     #     err = np.sqrt(np.var(dist) / (np.sum(last_kp_sample) - min_num_inliers))
                #     # else:
                #     err = float("NaN")

                if (np.sum(inliers) > best_count_matches):
                    # if (err < best_err_matches) :
                    best_kp_sample_matches = kp_sample.copy()
                    best_count_matches = np.sum(inliers)
                    err = np.sqrt(np.var(dist) / (np.sum(kp_sample) - min_num_inliers))
                    best_err_matches = err
                    # best_tform = tform

                    # if temp_best_count > best_count_matches or (
                    #         temp_best_count == best_count_matches and temp_best_err < best_err_matches):
                    #     best_count_matches = temp_best_count
                    #     best_err_matches = temp_best_err
                    #     best_kp_sample_matches = temp_best_inliers.copy()
                    progressbar.set_description(
                        f"Head {head1.frame_id} & {head2.frame_id} :cnt:{best_count_matches:.0f} err:{best_err_matches:.4f} cov:{100 * best_coverage_all_points :.2f}%")

                if j < no_iterations_all_points:
                    R, c, t = tform['rotation'], tform['scale'], tform[
                        'translation']
                    xyz2_trans = xyz2.dot(c * R) + t
                    distances, indices = nearest_neighbor(xyz2_trans, xyz1)
                    count_all_points = np.sum(distances < max_dist_all_points)
                    if count_all_points > best_count_all_points:
                        best_count_all_points = count_all_points
                        best_coverage_all_points = best_count_all_points / xyz2.shape[0]
                        best_inliers_all_points = kp_sample.copy()
                        progressbar.set_description(
                            f"Head {head1.frame_id} & {head2.frame_id} :cnt:{best_count_matches:.0f} err:{best_err_matches:.4f} cov:{100 * best_coverage_all_points :.2f}%")

            # if not temp_best_tform is None:
            # if j < no_iterations_all_points:
            #     R, c, t = temp_best_tform['rotation'], temp_best_tform['scale'], temp_best_tform['translation']
            #     xyz2_trans = xyz2.dot(c * R) + t
            #     distances, indices = nearest_neighbor(xyz2_trans, xyz1)
            #     count_all_points = np.sum(distances < max_dist_all_points)
            #     if count_all_points > best_count_all_points:
            #         best_count_all_points = count_all_points
            #         best_coverage_all_points = best_count_all_points / xyz2.shape[0]
            #         best_inliers_all_points = temp_best_inliers.copy()
            #         progressbar.set_description(
            #             f"Head {head1.frame_id} & {head2.frame_id} :cnt:{best_count_matches:.0f} err:{best_err_matches:.4f} cov:{100 * best_coverage_all_points :.2f}%")
    return best_inliers_all_points, best_coverage_all_points, best_kp_sample_matches, best_err_matches, matches



def draw_matches(head1, head2, matches, inliers):
    images_dir="images"
    if not os.path.isdir(images_dir):
        os.mkdir(images_dir)
    head1.background_color=np.array([0,0,0.99])
    head2.background_color=np.array([0,0,0.99])
    img1 = cv2.cvtColor((head1.get_filtered_image() * 256).astype("uint8"), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor((head2.get_filtered_image() * 256).astype("uint8"), cv2.COLOR_BGR2RGB)

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(0, 0, 255),
                       matchesMask=inliers.ravel().tolist(),
                       flags=0)

    img3 = cv2.drawMatches(img1, head1.kp, img2, head2.kp, matches1to2=matches,
                           outImg=None, **draw_params)

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (0, 40)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.putText(img3, f"Sequence {head1.sequence_id}, frames {head1.frame_id} & {head2.frame_id}",
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    file_name=f"Seq_{head1.sequence_id}_frames_SIFT_{head1.frame_id}_{head2.frame_id}.png"
    full_path= os.path.join(images_dir, file_name)
    cv2.imwrite(full_path, img3)


def get_descriptors(img, SIFT_contrastThreshold=0.04, SIFT_edgeThreshold=10, SIFT_sigma=4):
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

def get_matched_points(img1, kp1, des1, img2, kp2, des2, ratio=0.7, searching=True):
    '''
    find a set of good matching descriptors given two set of keypoints and descriptors
    '''
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    cv2.imwrite("img1.png", img1)
    # Apply ratio test
    good = []
    good_without_list = []
    # for m, n in matches:
    #     if m.distance < ratio * n.distance:
    #         good.append([m])
    #         good_without_list.append(m)
    good = matches
    good_without_list = matches
    # cv2.drawMatchesKnn expects list of lists as matches.
    if not searching:
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

        # plt.imshow(img3), plt.show()
        cv2.imwrite("des_match.png", img3)
    return good_without_list


def clean_matches(kp1, img1, kp2, img2, matches, min_match=4, searching=True):
    '''
    param:
        matches (list(DMatch)): a list of matching object
    return:
        a list of cleaned matching object after RANSAC
    '''
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # delete matches with huge y differece
    matches = [m for m in matches if (abs(kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1]) < 15)]

    # mean = np.mean([640 - kp1[m.queryIdx].pt[0] + kp2[m.trainIdx].pt[0] for m in matches])
    # matches = [m for m in matches if (abs(640 - kp1[m.queryIdx].pt[0] + kp2[m.trainIdx].pt[0]- mean) < 70)]
    # print([640 - kp1[m.queryIdx].pt[0] + kp2[m.trainIdx].pt[0] for m in matches])
    # print(np.mean([640 - kp1[m.queryIdx].pt[0] + kp2[m.trainIdx].pt[0] for m in matches]))
    MIN_MATCH_COUNT = min_match
    if len(matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10)
        matchesMask = mask.ravel().tolist()

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        if not searching:
            img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d, turn up the match ratio." % (len(matches), MIN_MATCH_COUNT))
        matchesMask = None

    mask = np.array(matchesMask)
    mask = [m > 0 for m in mask.reshape([-1, ])]
    cleaned_matches = np.array(matches)[mask]

    if not searching:
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
        cv2.imwrite("des_match_cleaned.png", img3)
        plt.imshow(img3, 'gray'), plt.show()

    return cleaned_matches

def drawMatches(img1,kp1,img2,kp2,matches):
    '''
    Given two images, kp point list, a set of match points
    Return a combined image with match points marked in green
    '''