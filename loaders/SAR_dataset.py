import asyncio  # may use later
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers  # not necessary at the moment
from torch import tensor
from pathlib import Path


class SAR_Dataset(Dataset):
    """
    A SAR dataset is any dataset with 4-channel images (RGB + elevation), text, and structured data features,
    e.g., Million-CASE.
    """

    def __init__(self, root_path, img_dir, csv_file, delimeter, transform):
        self.root_path = Path(root_path)
        self.data = pd.read_csv(
            self.root_path / csv_file, sep=delimeter, engine="python"
        )
        self.img_path = self.root_path / img_dir
        self.transform = transform
        self.tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]

        # Load and process image asynchronously (potential to speed up data loading in the future)
        # could also make the entire data loading process (the __getitem__ function asynchronous)
        # loop = asyncio.get_event_loop()
        # image = await loop.run_in_executor(None, self.load_image, "Images/NAIP_" + sample['fid'] + ".tif")
        # image = ToTensor()(image)

        # Load and transform image (assuming images are found in the Images/Sample Tiffs directory)
        image_tensor = self.load_topo_img_tensor(sample["FID"])

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        # Extract text
        text = sample["Prompt"]

        # not necessary at the moment
        # text = self.text_to_tensor(text)

        # Extract weather/time data TODO: fix these (add time features as well)
        temp = sample["temp"]
        feels_like = sample["feels_like"]
        pressure = sample["pressure"]
        humidity = sample["humidity"]
        dew_point = sample["dew_point"]
        wind_speed = sample["wind_speed"]
        wind_deg = sample["wind_deg"]
        wind_gust = sample["wind_gust"]
        rain_3h = sample["rain_3h"]
        cloud_count = sample["cloud_count"]
        visibility = sample["visibility"]
        dt = sample["dt"]
        dt_iso = sample["dt_iso"]
        timezone = sample["timezone"]

        # Return data in dictionary format
        return {
            "image": image_tensor,
            "text": text,
            "temp": temp,
            "temp_min": temp_min,
            "temp_max": temp_max,
            "feels_like": feels_like,
            "pressure": pressure,
            "humidity": humidity,
            "dew_point": dew_point,
            "wind_speed": wind_speed,
            "wind_deg": wind_deg,
            "wind_gust": wind_gust,
            "rain_3h": rain_3h,
            "cloud_count": cloud_count,
            "visibility": visibility,
            "dt": dt,
            "dt_iso": dt_iso,
            "timezone": timezone,
        }

    def load_topo_img_tensor(self, fid):
        naip_img = Image.open(self.img_path / f"NAIP_{fid}.tif")
        dem_img = Image.open(self.img_path / f"DEM_{fid}.tif")

        # TODO: need to convert these to tensors and concat them on the right axis

        return image

    # Example usage of the tokenizer
    def text_to_tensor(self, text):
        encoding = self.tokenizer.encode(text)
        tensor = tensor(encoding.ids)
        return tensor
