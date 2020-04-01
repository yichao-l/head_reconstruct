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

#
sp = 0.5
#
# mhead = MultiHead.load_from_pickle(1, "mhead1_refine")
# # mhead.create_png_series(name="mhead1_refine", sparcity=sp)
# mhead.reset_all_head_positions()
# mhead.Method_C(sift_transform_method="matches", icp=False, refine_range=False, refine_local=False)
# # mhead.create_png_of_spheres(sp, name="head1_all", alpha=0)
# mhead.create_mesh(name="head_1_mesh")
#
# mhead = MultiHead.load_from_pickle(2, "mhead2_refine")
# # mhead.create_png_series(name="mhead2_refine", sparcity=sp)
# mhead.reset_all_head_positions()
# mhead.Method_C(sift_transform_method="matches", icp=False, refine_range=False, refine_local=False)
# # mhead.create_png_of_spheres(sp, name="head2_all", alpha=0)
# mhead.create_mesh(name="head_2_mesh")
#
#
# mhead = MultiHead.load_from_pickle(4, "mhead4_refine")
# mhead.create_png_series(name="mhead4_refine", sparcity=sp)
# mhead.reset_all_head_positions()
# mhead.Method_C(sift_transform_method="matches", icp=False, refine_range=False, refine_local=False)
# mhead.create_png_of_spheres(sp, name="head4_all", alpha=0)
# mhead.create_mesh(name="head_4_mesh")
# #
#
#
# mhead = MultiHead.load_from_pickle(3, "mhead3_refine")
# # mhead.create_png_series(name="mhead3_refine", sparcity=sp)
# mhead.reset_all_head_positions()
# mhead.Method_C(sift_transform_method="matches", icp=False, refine_range=False, refine_local=False)
# # mhead.create_png_of_spheres(sp, name="head3_all", alpha=0)
# mhead.create_mesh(name="head_3_mesh")
#


mhead = MultiHead.load_from_pickle(3, "mhead2_refine")
mhead.calc_all_sift_transforms()
mhead.save()
mhead = MultiHead.load_from_pickle(3, "mhead3_refine")
mhead.calc_all_sift_transforms()
mhead.save()
mhead = MultiHead.load_from_pickle(3, "mhead4_refine")
mhead.calc_all_sift_transforms()
mhead.save()
mhead = MultiHead.load_from_pickle(3, "mhead1_refine")
mhead.calc_all_sift_transforms()
mhead.save()

# import pandas as pd
# df=pd.DataFrame(columns=["A","A + ICP","A + Refine","C"])
# for Sequence in [1,2,3,4]:
#     mhead = MultiHead.load_from_pickle(3, f"mhead{Sequence}_refine")
#     d_ref = mhead.left_eye_deviation()
#     mhead.reset_all_head_positions()
#     mhead.Method_A(sift_transform_method="matches", icp=False, refine_range=False, refine_local=False)
#     d_A = mhead.left_eye_deviation()
#     mhead.Method_A(sift_transform_method="matches", icp=True, refine_range=False, refine_local=False)
#     d_A_ICP = mhead.left_eye_deviation()
#     mhead.Method_C(sift_transform_method="matches", icp=False, refine_range=False, refine_local=False)
#     d_C = mhead.left_eye_deviation()
#     df=df.append({"A + ICP":d_A_ICP,"A + Refine":d_ref,"A":d_A,"C":d_C}, ignore_index=True)
# print(df)
#
