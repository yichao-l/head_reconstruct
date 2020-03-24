from single_head import SingleHead
from multi_head import MultiHead
from tqdm.autonotebook import tqdm
import subprocess

# manually select which sequence to generate

Sequence = 2

'''
# generate all single head objects from file 
# '''
# with tqdm(total=15) as progressbar:
#
#     for frame_idx in range(1, 16): # loop through all frames tqdm is here to generate the progress bar
#
#         progressbar.set_description(f"Reading              Head {frame_idx}")
#         progressbar.update(0.1)
#         head = SingleHead.read_from_file(Sequence, frame_idx)
#
#         progressbar.set_description(f"Applying filters for Head {frame_idx}")
#         progressbar.update(0.7)
#         head.apply_all_filters()
#
#         progressbar.set_description(f"Saving               Head {frame_idx}")
#         progressbar.update(0.3)
#         head.save()
#
# list_of_all_heads=[SingleHead.load_from_pickle(Sequence,i) for i in range (1,16)]
# mhead=  MultiHead.create_from_heads(list_of_all_heads)
# mhead.save()


mhead.reset_all_head_positions()
for idx in range(15):
    link = mhead.links[idx]
    link.reset()
    mhead.ransac_from_link(link)

mhead = MultiHead.load_from_pickle(Sequence)
mhead.create_png_series()
