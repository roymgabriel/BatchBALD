import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class RSNAPneumoniaDataset(Dataset):
    def __init__(self, csv_file, root_dir, target_col, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the DICOM images.
            target_col (string): Target column name (must be one of `Target` or `multiTarget`).
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.pneumonia_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.targets = self.pneumonia_frame[target_col].values

    def __len__(self):
        return len(self.pneumonia_frame)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.pneumonia_frame.iloc[idx, 0] + '.png')
        image = read_image(image_path)  # Loads PNG image as a tensor

        target = self.targets[idx]
        if self.transform:
            image = self.transform(image)

        return {'image': image, 'target': target}

