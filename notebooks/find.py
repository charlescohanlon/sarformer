import torch
import pandas as pd
from src.data.multimodal_dataset_folder import MultiModalDatasetFolder
from src.data.modality_transforms import UnifiedDataTransform
from src.data.image_augmenter import EmptyAugmenter
from src.data.modality_info import MODALITY_INFO, MODALITY_TRANSFORMS
from tokenizers import Tokenizer
from tqdm import tqdm

image_augmenter = EmptyAugmenter()
# all modalities
transforms = UnifiedDataTransform(
    transforms_dict=MODALITY_TRANSFORMS,
    image_augmenter=image_augmenter,
    resample_mode="bicubic",
    add_sizes=False,
)
data_df = pd.read_csv(
    "/scratch/bdej/cohanlon/data/labeled_sar_data.csv", sep="@", index_col="uid"
)
text_tokenizer = Tokenizer.from_file(
    "/u/cohanlon/sarformer/fourm/utils/tokenizer/trained/tokenizer_inc_nonUS_lower.json"
)
dataset = MultiModalDatasetFolder(
    root="/scratch/bdej/cohanlon/data/train",
    modalities=[
        "tok_rgb@224",
        "tok_depth@224",
        "structured_data",
        "caption",
    ],
    data_df=data_df,
    tokenizer=text_tokenizer,
    max_text_tok_length=516,
    valid_ids=list(data_df.index),
    modality_info=MODALITY_INFO,
    modality_transforms=MODALITY_TRANSFORMS,
    transform=transforms,
    return_path=True,
)
sampler = torch.utils.data.SequentialSampler(dataset)
data_loader = torch.utils.data.DataLoader(
    dataset,
    sampler=sampler,
    batch_size=1,
    num_workers=8,
    pin_memory=True,
    drop_last=False,
)

for x in tqdm(data_loader):
    caption = x["caption"][0][0]  # returns mask so need to index
    if len(caption) != 516:
        print("caption length:", len(caption), "uid:", x["file_name"])
    structured_data = x["structured_data"][0]
    if len(structured_data) != 108:
        print("structured_data length:", len(structured_data), "uid:", x["file_name"])

print(x)
