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
    # apply all filters to remove the unwanted cloud points.
    head.apply_all_filters()
    # save the processed head
    head.save()

# create a list of all the heads, from the save SingleHead objects
list_of_all_heads = [SingleHead.load_from_pickle(Sequence, i) for i in
                     range(1, 16)]  

# create a MultiHead object from the list:
mhead = MultiHead.create_from_heads(list_of_all_heads)
# calculate teh SIFT points for each SingleHead in the MultiHead object:
mhead.calc_all_sift_keypoints()
# calculate the SIFT transform for each pair of adjacent heads, each pair of adjacnt SingleHeads shares a Link Object:
mhead.calc_all_sift_transforms()

# Method A
for link_idx in range(14):  # iterate through the links between heads
    # calculate and perform all transformations for each link:
    mhead.all_transforms_from_link(mhead.links[link_idx])

# # Method C
only_first_n = 15
mhead.reset_all_head_positions()
link_index, err = mhead.get_next_unpositioned_link(method="coverage")
positioned_head_count = 0
joined_heads = set()
while (not link_index is None) and (positioned_head_count < only_first_n or only_first_n == -1):
    mhead.links[link_index].print_short()
    joined_heads.add(mhead.links[link_index].left)
    joined_heads.add(mhead.links[link_index].right)
    mhead.heads[mhead.links[link_index].right - 1].background_color = np.asarray([0, 0, 1])
    mhead.heads[mhead.links[link_index].left - 1].background_color = np.asarray([0, 0, 1])
    foo = mhead.all_transforms_from_link(mhead.links[link_index], method="coverage", ICP=True, Refine_Range=False,
                                         Refine_local=False)
    link_index, err = mhead.get_next_unpositioned_link(method="coverage")
    positioned_head_count = max(positioned_head_count + 1, 2)
mhead.left_eye_deviation()

# create a series of png images for the spheres
mhead.create_png_series()

# calculate the mean eye distance:
mean_eye_dist = mhead.left_eye_deviation()

# Save the MultiHead object for later re-use:
mhead.save()
