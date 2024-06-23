import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.v2 import PILToTensor
from transformers import AutoTokenizer
from torch import tensor
from pathlib import Path


class SARDataset(Dataset):
    """
    Any dataset with 4-channel images (RGB + elevation), text,
    and tabular data features. E.g., Million-CASE.\n
    Args:
        root_path: string path to root of dataset
        img_dir: name of the directory containing all images
        csv_file: name of csv file with non-image data
        csv_delimiter: the csv_file delimiter
        max_text_length: the maximum number of text tokens allowed as input
        mm_objective: if true, returns the modality tensors with no target
        transform: must take all modality tensors as input
        target_transform: produces label tensors
    """

    def __init__(
        self,
        root_path,
        img_dir,
        csv_file,
        csv_delimeter,
        max_text_length=512,
        mm_objective=False,
        transform=None,
        target_transform=None,
    ):
        self.root_path = Path(root_path)
        self.data = pd.read_csv(self.root_path / csv_file, sep=csv_delimeter)
        self.img_path = self.root_path / img_dir
        self.max_text_length = max_text_length
        self.mm_objective = mm_objective
        self.transform = transform
        self.target_transform = target_transform

        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        assert self.tokenizer.is_fast

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]

        image_tensor = self.load_topo_img(sample["fid"])  # (4, H, W)

        text_tensor = self.text_to_tensor(sample["prompt"])  # (1, seq_len)

        # NOTE: all tabular features must be floating point numbers in csv_file
        tabular_tensor = tensor(  # (1, num_features)
            [
                sample[name]
                for name in [  # list of tabular feature names
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "2m_dewpoint_temperature",
                    "2m_temperature",
                    "surface_pressure",
                    "total_precipitation",
                    "total_cloud_cover",
                    "low_cloud_cover",
                    "slope_of_subgridscale_orography",
                    "high_vegetation_cover",
                    "surface_net_solar_radiation",
                    "soil_type",
                    "trapping_layer_base_height",
                    "total_column_water_vapour",
                    "skin_temperature",
                    "precipitation_type",
                    "duration",
                    "date",
                    "start_time",
                ]
            ],
            dtype=torch.float32,
        )

        if self.transform is not None:
            image_tensor, text_tensor, tabular_tensor = self.transform(
                image_tensor, text_tensor, tabular_tensor
            )

        if self.mm_objective:
            return image_tensor, text_tensor, tabular_tensor

        target = self.target_transform(sample["coord_offset"], sample["coord_deg"])

        return (
            image_tensor,
            text_tensor,
            tabular_tensor,
            target,
        )

    def load_topo_img(self, fid):
        # produces path string to an image type (NAIP or DEM) with FID fid
        path_to = lambda type: self.img_path / f"{type}_{fid}.tif"

        # NOTE: this is purposely in JPEG format despite the .tif file ending
        nt = PILToTensor()(Image.open(path_to("NAIP"), formats=("JPEG",)))

        dt = PILToTensor()(Image.open(path_to("DEM")), formats=("TIFF",))

        # concats the images to make the result 4-channel
        return torch.cat((nt, dt), dim=0)

    def text_to_tensor(self, text):
        tokenized_input = self.tokenizer(
            text,
            max_length=self.max_text_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
