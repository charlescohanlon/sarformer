import os
from src.data.modality_transforms import CaptionTransform
from collections import deque
import pandas as pd
from tqdm import tqdm

DIR = "/scratch/bdej/cohanlon/data/train/caption/sar"
all_csvs = os.listdir(DIR)
uids = []
for csv in all_csvs:
    path = os.path.join(DIR, csv)
    df = pd.read_csv(path, index_col="UID", dtype=str)
    for uid in df.index:
        uids.append((path, uid))


lengths = deque()
transform = CaptionTransform(shuffle=False)
for x in tqdm(uids):
    caption = transform.load(x)
    tensor = transform.postprocess(caption)
    lengths.append(tensor.shape[-1])

print(pd.Series(lengths).describe())
