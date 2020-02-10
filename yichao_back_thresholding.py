import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes
import sys
np.set_printoptions(threshold=sys.maxsize)

rgb_hand = cv2.imread('./head_2d_image/full_1_1.png')
hsv_hand = cv2.cvtColor(rgb_hand, cv2.COLOR_BGR2HSV)
s = hsv_hand[:,:,1]
cv2.imshow("original",rgb_hand)
# extract the edge and dilate
edge = cv2.Canny(s,150,200)
kernel = np.ones((2,2))
cv2.imshow("edge",edge)
dilation = cv2.dilate(edge,kernel,iterations = 3)
cv2.imshow("dilation",dilation)
closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, np.ones((10,10)))
# dilation2 = cv2.dilate(closing,kernel,iterations = 2) 
# ero = cv2.erode(dilation2,kernel,iterations=3)
cv2.imshow("closing",closing)

# # flood fill
im_floodfill = closing.copy()
im_floodfill = binary_fill_holes(im_floodfill)
im_floodfill = im_floodfill*255

# plt.imshow(im_floodfill)
cv2.imshow("flood fill",im_floodfill)

# erode




# mask = np.zeros_like(rgb_hand)
# mask[:,:,0] = erode
# mask[:,:,1] = erode
# mask[:,:,2] = erode
# fingers = cv2.bitwise_and(rgb_hand,mask)

# # create a green background mask
# green = rgb_hand[:,:,1]
# mask_g = cv2.bitwise_and(green,erode)
# cv2.imshow('mask_g',mask_g)

# # threshold for the new mask
# th = 200
# print(np.max(mask_g))
# mask_g[mask_g>th] = 0
# print(np.max(mask_g))
# mask_g[mask_g>0] = 255
# mask = np.zeros_like(rgb_hand)
# mask[:,:,0] = mask_g
# mask[:,:,1] = mask_g
# mask[:,:,2] = mask_g 
# cv2.imshow('new_mask_g',mask_g)
# masked_finger = cv2.bitwise_and(rgb_hand,mask)

# # labeling and bounding box


# # show the image
# cv2.imshow('filled',masked_finger)
cv2.waitKey(0)
cv2.destroyAllWindows()