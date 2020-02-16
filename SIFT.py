import cv2
import matplotlib.pyplot as plt
import numpy as np

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
