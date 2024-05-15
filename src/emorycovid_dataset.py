import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from torchvision.io import read_image

class EmoryCOVIDDataset(Dataset):
    def __init__(self, csv_file, root_dir, target_col, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the COVID images.
            target_col (string): Target column name (must be one of `binary_target` or `Median`).
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.covid_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.targets = self.covid_frame[target_col].values

    def __len__(self):
        return len(self.covid_frame)

    def __getitem__(self, idx):
        # First column is File Name which has the image file name
        image_path = os.path.join(self.root_dir, self.covid_frame.iloc[idx, 0])
        image = read_image(image_path)  # Loads JPG image as a tensor

        target = self.targets[idx]
        if self.transform:
            image = self.transform(image)

        return {'image': image, 'target': target}

