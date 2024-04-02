import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.v2 import PILToTensor
from tokenizers import Tokenizer, models, pre_tokenizers  # not necessary at the moment
from torch import tensor
from pathlib import Path


class SAR_Dataset(Dataset):
    """
    Any dataset with 4-channel images (RGB + elevation), text,
    and tabular data features. E.g., Million-CASE.\n
    Args:
        root_path: string path to root of dataset
        img_dir: name of the directory containing all images
        csv_file: name of csv file with non-image data
        delimiter: the csv_file delimiter
        mm_objective: if true, returns the modality tensors with no target
        target_transform: (required) produces label tensor(s)
        transform: must take all modality tensors as input
    """

    def __init__(
        self,
        root_path,
        img_dir,
        csv_file,
        delimeter,
        mm_objective=False,
        transform=None,
        target_transform=None,
    ):
        self.root_path = Path(root_path)
        self.data = pd.read_csv(
            self.root_path / csv_file, sep=delimeter, engine="python"
        )
        self.img_path = self.root_path / img_dir
        self.mm_objective = mm_objective
        self.transform = transform
        self.target_transform = target_transform
        self.tokenizer = Tokenizer.from_pretrained("bert-base-cased")

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
                    "start_date",
                    "start_time",
                    "duration",
                    "temp",
                    "feels_like",
                    "pressure",
                    "humidity",
                    "dew_point",
                    "wind_speed",
                    "wind_deg",
                    "wind_gust",
                    "rain_3h",
                    "snow_3h",
                    "clouds_all",
                    "visibility",
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

        # NOTE: purposely in JPEG format despite the .tif file ending
        nt = PILToTensor()(Image.open(path_to("NAIP"), formats=("JPEG",)))
        dt = PILToTensor()(Image.open(path_to("DEM")), formats=("TIFF",))

        # concats the images to make the result 4-channel
        return torch.cat((nt, dt), dim=0)

    def text_to_tensor(self, text):
        encoding = self.tokenizer.encode(text)  # TODO: figure out word embeddings
        return tensor(encoding.ids)
