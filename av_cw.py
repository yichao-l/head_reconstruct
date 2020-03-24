'''
Top Level Code for AV Coursework
Merges individual point clouds into a single mulithea object
Produces images of the individual merging steps and calculates performance metrics
'''

from single_head import SingleHead
from multi_head import MultiHead

Sequence = 1

for frame_idx in range(1, 16):  # loop through all frames
    # read the data from the file, define by Sequence and frame_idx and store the data in a SingeHead object
    head = SingleHead.read_from_file(Sequence,
                                     frame_idx)
    # apply all filters
    head.apply_all_filters()
    # save the processed head
    head.save()

list_of_all_heads = [SingleHead.load_from_pickle(Sequence, i) for i in
                     range(1, 16)]  # create a list of all the heads, from the save SingleHead objects

# create a MultiHead object from the list:
mhead = MultiHead.create_from_heads(list_of_all_heads)
# calculate teh SIFT points for each SingleHead in the MultiHead object:
mhead.calc_all_sift_keypoints()
# calculate the SIFT transform for each pair of adjacent heads, each pair of adjacnt SingleHeads shares a Link Object:
mhead.calc_all_sift_transforms()

for link_idx in range(14):  # iterate through the links between heads
    # calculate and perform all transformations for each link:
    mhead.all_transforms_from_link(mhead.links[link_idx])

# create a series of png images for the spheres
mhead.create_png_series()

# calculate the mean eye distance:
mean_eye_dist = mhead.left_eye_deviation()

# Save the MultiHead object for later re-use:
mhead.save()
