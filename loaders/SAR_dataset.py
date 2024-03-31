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
    A SAR dataset is any dataset with 4-channel images (RGB + elevation), text,
    and tabular data features. E.g., Million-CASE.\n
    Args:
        root_path: string path to root of dataset
        img_dir: name of the directory containing all images
        csv_file: name of csv file with non-image data
        delimiter: the csv_file delimiter
        target_transform: (required) produces label tensor(s)
        transform: applied to input tensors
    """

    def __init__(
        self,
        root_path,
        img_dir,
        csv_file,
        delimeter,
        target_transform,
        transform=None,
    ):
        self.root_path = Path(root_path)
        self.data = pd.read_csv(
            self.root_path / csv_file, sep=delimeter, engine="python"
        )
        self.img_path = self.root_path / img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.tokenizer = Tokenizer.from_pretrained("bert-base-cased")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]

        image_tensor = self.load_topo_img(sample["FID"])  # (4, 512, 512)

        text_tensor = self.text_to_tensor(sample["Prompt"])  # (TDB)

        # NOTE: all tabular features must be floating point numbers in csv_file
        tabular_tensor = tensor(  # (num_features, 1)
            [
                sample[name]
                for name in [  # list of tabular feature names
                    "Temp",
                    "Feels_like",
                    "Pressure",
                    "Humidity",
                    "Dew_point",
                    "Wind_speed",  # TODO: make sure these align with server.py names
                    "Wind_deg",
                    "Wind_gust",
                    "Rain_3h",
                    "Snow_3h",
                    "Clouds_all",
                    "Visibility",
                    "Date",
                    "Duration",
                ]
            ],
            dtype=torch.float32,
        )

        # NOTE: in a masked modeling objective, transform and target_tranform must be stateful,
        # such that the target_tranform produces a tensor for the portion that is masked by the transform
        if self.transform is not None:
            image_tensor = self.transform(image_tensor, "image")
            text_tensor = self.transform(text_tensor, "text")
            tabular_tensor = self.transform(tabular_tensor, "tabular")

        # TODO: use coord offset and coord degree to produce probability map tensor
        target = self.target_transform(sample["coord_offset"], sample["coord_deg"])

        return (
            {"image": image_tensor, "text": text_tensor, "tabular": tabular_tensor},
            target,
        )

    def load_topo_img(self, fid):
        # produces path string to an image type (NAIP or DEM) with FID fid
        path_to = lambda type: self.img_path / f"{type}_{fid}.tif"

        # NOTE: purposely in JPEG format despite the .tif file ending
        # PILToTensor() does NOT normalize pixel values
        nt = PILToTensor()(Image.open(path_to("NAIP"), formats=("JPEG",)))
        dt = PILToTensor()(Image.open(path_to("DEM")))

        # concats the images to make the result 4-channel
        return torch.cat((nt, dt), dim=0)

    def text_to_tensor(self, text):
        encoding = self.tokenizer.encode(text)  # TODO: figure out word embeddings
        return tensor(encoding.ids)
