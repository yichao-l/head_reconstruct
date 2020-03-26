'''
Top Level Code for AV Coursework
Merges individual point clouds into a single mulithead object
Produces images of the individual merging steps and calculates performance metrics
'''
from single_head import SingleHead
from multi_head import MultiHead
import numpy as np

# Seqence number in the data, range from 1 to 4.
Sequence = 1

# Frame number in the data, range from 1 to 15.
for frame_idx in range(1, 16):  # loop through all frames
    # Read the data from the file, define by Sequence and frame_idx and store the data in a SingeHead object
    head = SingleHead.read_from_file(Sequence,
                                     frame_idx)
    print(frame_idx)
    # apply all filters to remove the unwanted cloud points.
    head.apply_all_filters()
    # save the processed head
    head.save()

# create a list of all the heads, from the save SingleHead objects
list_of_all_heads = [SingleHead.load_from_pickle(Sequence, i) for i in
                     range(1, 16)]  

# create a MultiHead object from the list:
mhead = MultiHead.create_from_heads(list_of_all_heads)
mhead.save()
# calculate teh SIFT points for each SingleHead in the MultiHead object:
mhead.calc_all_sift_keypoints()
# calculate the SIFT transform for each pair of adjacent heads, each pair of adjacnt SingleHeads shares a Link Object:
mhead.calc_all_sift_transforms()

# Set merge method to either A-C
method = 'A'

if method == 'A':
    # corresponds to Method A in the documentation/report.
    mhead.Method_A(sift_transform_method="matches", icp=True, refine_range=False, refine_local=False)
elif method == 'B':
    # Method B
    mhead.Method_B(sift_transform_method="matches", icp=True, refine_range=False, refine_local=False)
elif method == 'C':
    # Method C
    mhead.Method_C(sift_transform_method="matches", icp=True, refine_range=False, refine_local=False)

# create a series of png images for the spheres
mhead.create_png_series()

# calculate the mean eye distance:
mean_eye_dist = mhead.left_eye_deviation()

# Save the MultiHead object for later re-use:
mhead.save()