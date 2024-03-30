import asyncio  # may use later
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from loaders.SAR_dataset import Million_CASE

# NOTES:
# unique identifier fid
# NAIP_129389.tif (random number for unique identifier 0 - ? (more than a million))
# pick unique delimeter for csv
# code should work with jpegs (right now tiffs)
# vault/DL4SAR/gen_data/topo
# not including city_name, lat, or long, nothing in weather. Include all in main. Include everything in wind, clouds, rain, snow, visibility
# all above aligned on fid, directly relates to fid. TIF = dataset_fid.pdf

DELIMETER = "\|\~\|"  # regex way to define delimeter as |~|, can be changed later
CSV_FILE_NAME = "weather_data.csv"
BATCH_SIZE = 32
IMAGE_SIZE = 512


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Valid transforms for image (resize necessary for tensor stack)
valid_transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ]
)


def main():
    dataset = Million_CASE(
        csv_file=CSV_FILE_NAME, delimeter=DELIMETER, transform=valid_transform
    )
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Visualize result of DataLoader
    for data in data_loader:
        print("Length of item:", len(data))
        print(data)


if __name__ == "__main__":
    main()
