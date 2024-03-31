from typing import Literal
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from loaders.SAR_dataset import SAR_Dataset


DELIMETER = "\|\~\|"  # regex to define delimeter |~|
CSV_FILE_NAME = "weather_data.csv"
BATCH_SIZE = 32
IMAGE_SIZE = 512


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# torch.backends.cudnn.benchmark = True # cudnn benchmarking only useful when the input size is constant


class MaskedModelingTransform:
    def __init__(self, mask_proportion):
        self.mask_proportion = mask_proportion
        self.masked_image_portion = None
        self.masked_text_portion = None
        self.masked_tabular_portion = None

    def __call__(
        self, x: Tensor, input_type: Literal["image", "text", "tabular", "target"]
    ):
        if input_type == "image":
            _, H, W = x.size
            
            return
        elif input_type == "text":
            pass
        elif input_type == "tabular":
            pass
        else:
            assert (
                self.masked_image_portion != None
                and self.masked_text_portion != None
                and self.masked_tabular_portion != None
            )

            pass


def main():
    pass
    # dataset = SAR_Dataset(
    #     csv_file=CSV_FILE_NAME, delimeter=DELIMETER, transform=valid_transform
    # )
    # data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Visualize result of DataLoader


if __name__ == "__main__":
    main()
