import os
import numpy as np
from PIL import Image

source_dir1 = "/scratch/bdej/cohanlon/unsupervised/train/depth/"
source_dir2 = "/scratch/bdej/cohanlon/unsupervised/eval/depth/"

min_depth = max_depth = 0
for src in [
    source_dir1 + "million-case",
    source_dir1 + "labeled_sar",
    source_dir1 + "unlabeled_sar",
    source_dir2 + "labeled_sar",
]:
    for filename in os.listdir(src):
        im = Image.open(src + "/" + filename)
        arr = np.array(im)
        min_depth = min(arr.min(), min_depth)
        max_depth = max(arr.max(), max_depth)

print(min_depth, max_depth)
