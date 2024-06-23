from typing import Literal
import torch
import logging

logger = logging.getLogger(__name__)
from torch.utils.data import DataLoader, random_split
import torch
from utils.sar_dataset import SAR_Dataset


DELIMETER = "\|\~\|"  # regex to define delimeter |~|
BATCH_SIZE = 32


def main():
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
    )

    if not torch.cuda.is_available():
        logging.critical("Cuda not found")
        return

    logging.info("Creating dataset")
    million_case = SAR_Dataset(
        root_path="/home/charles/million-case",
        img_dir="topo",
        csv_file="Gen_Data.csv",
        csv_delimeter=DELIMETER,
        mm_objective=True,
    )

    logging.info("Creating loaders")
    train_set, test_set = random_split(million_case, [0.99, 0.01])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)


if __name__ == "__main__":
    main()
