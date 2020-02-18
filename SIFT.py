import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook as tqdm
from Procrustes2 import *


def get_matches(head1, head2):
    # find K nearest matches ( in terms of descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(head1.des, head2.des, k=2)
    # unfoled the list
    matches = [val for sublist in matches for val in sublist]
    return matches


def remove_height_variation_from_matches(head1, head2, matches):
    max_variation_y_dimension = 15
    matches = [m for m in matches if
               (abs(head1.kp[m.queryIdx].pt[1] - head2.kp[m.trainIdx].pt[1]) < max_variation_y_dimension)]
    return matches


def get_xyz_from_matches(head1, head2, matches):
    '''

    :param head1: a head object
    :param head2: a head object
    :param matches: amacthes object
    :return: xyz1/2 the sets of 3d points of the respective heads, corresponding to the matches
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


def ransac(head1, head2, matches):
    xyz1, xyz2, matches = get_xyz_from_matches(head1, head2, matches)

    max_dist = 0.015
    best_count = 0
    best_err = 1000
    best_inliers = []
    No_Iterations = 5000
    min_num_inliers = 6

    with tqdm(total=No_Iterations) as progressbar:
        for j in range(No_Iterations):
            inliers = np.random.rand(xyz2.shape[0]) > 0.5
            progressbar.update(1)
            for i in range(20):
                if np.sum(inliers) >= min_num_inliers:
                    try:
                        d, Z, tform = procrustes(xyz1[inliers], xyz2[inliers], scaling=False, reflection='best')
                        R, c, t = tform['rotation'], tform['scale'], tform['translation']
                        dist = np.linalg.norm(xyz2.dot(c * R) + t - xyz1, axis=1)
                        err = np.sqrt(np.var(dist) / (np.sum(inliers) - min_num_inliers))
                        last_inliers = inliers.copy()
                        inliers = dist < max_dist
                        if (np.sum(inliers) >= best_count):
                            # if (err < best_err) and np.sum(inliers)>min_num_inliers:
                            best_err = err
                            progressbar.set_description(
                                f"Head {head1.frame_id} & {head2.frame_id} :best count {best_count:.0f} Error {err:.4f}")
                            best_count = np.sum(inliers)
                            best_err = err
                            best_inliers = inliers.copy()
                            best_tform = tform
                        if np.all(last_inliers == inliers):
                            break
                    except:
                        pass
                else:
                    break

    if best_count < 6:
        raise ValueError('Could not match')
    return best_tform, best_inliers, best_err, matches


def draw_matches(head1, head2, matches, inliers):
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None, flags=2)

    img1 = cv2.cvtColor((head1.get_filtered_image() * 256).astype("uint8"), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor((head2.get_filtered_image() * 256).astype("uint8"), cv2.COLOR_BGR2RGB)

    img3 = cv2.drawMatches(img1, head1.kp, img2, head2.kp, [match for i, match in enumerate(matches) if inliers[i]],
                           None, **draw_params)
    cv2.imwrite("des_match_cleaned.png", img3)


def get_descriptors(img, SIFT_contrastThreshold=0.04, SIFT_edgeThreshold=10, SIFT_sigma=4):
    '''
    param:
    image (array): colored image
    return:
    void
    '''

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
