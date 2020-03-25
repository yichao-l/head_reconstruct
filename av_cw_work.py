from single_head import SingleHead
from multi_head import MultiHead
from tqdm.autonotebook import tqdm
import subprocess

# manually select which sequence to generate

for Sequence in [1, 2, 3, 4]:
    mhead = MultiHead.load_from_pickle(Sequence)
    mhead.calc_all_sift_transforms()
    mhead.Method_A(sift_transform_method="dynamic", icp=True, refine_range=True, refine_local=True)
    mhead.create_png_series(name="A_dynamic_refine", sparcity=0.5)
    mhead.save()
