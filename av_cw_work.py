from single_head import SingleHead
from multi_head import MultiHead
from tqdm.autonotebook import tqdm
import subprocess

# # manually select which sequence to generate
# Sequence = 2
#
# # Frame number in the data, range from 1 to 15.
# for frame_idx in range(1, 16):  # loop through all frames
#     # Read the data from the file, define by Sequence and frame_idx and store the data in a SingeHead object
#     head = SingleHead.read_from_file(Sequence,
#                                      frame_idx)
#     print(frame_idx)
#     # apply all filters to remove the unwanted cloud points.
#     head.apply_all_filters()
#     # save the processed head
#     head.save()
#
# # create a list of all the heads, from the save SingleHead objects
# list_of_all_heads = [SingleHead.load_from_pickle(Sequence, i) for i in
#                      range(1, 16)]
#
# # create a MultiHead object from the list:
# mhead = MultiHead.create_from_heads(list_of_all_heads)
# mhead.save()
#

mhead = MultiHead.load_from_pickle(3, "mhead3_refine")
mhead.create_png_series(name="mhead3_refine", sparcity=0.5)
