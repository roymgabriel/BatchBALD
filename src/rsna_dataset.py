import os
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torchvision.io import read_image

class RSNAPneumoniaDataset(Dataset):
    def __init__(self, csv_file, root_dir, target_col, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the DICOM images.
            target_col (string): Target column name (must be one of `Target` or `class`).
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.pneumonia_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.targets = self.pneumonia_frame[target_col].values

        if target_col.lower() == 'class':
            encoder = LabelEncoder()
            self.targets = encoder.fit_transform(self.targets)

    def __len__(self):
        return len(self.pneumonia_frame)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.pneumonia_frame.iloc[idx, 1] + '.png')
        image = read_image(image_path)  # Loads PNG image as a tensor

        target = self.targets[idx]
        if self.transform:
            image = self.transform(image)

        return {'image': image, 'target': target}


# class RSNAPneumoniaDataset(Dataset):
#     def __init__(self, csv_file, root_dir, target_col, transform=None):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the DICOM images.
#             target (string): Target column name (must be one of `Target` or `class`).
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.pneumonia_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform
#         self.targets = self.pneumonia_frame[target_col].values

#         if target_col.lower() not in ['target', 'class']:
#             raise ValueError("Target column must be one of `Target` or `class`!")

#         if target_col.lower() == 'target':
#             label_encoder = LabelEncoder()
#             # Fit label encoder and return encoded labels
#             self.targets = label_encoder.fit_transform(self.targets)


#     def __len__(self):
#         return len(self.pneumonia_frame)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         dicom_path = os.path.join(self.root_dir, self.pneumonia_frame.iloc[idx, 1] + '.dcm')
#         dicom_image = pydicom.dcmread(dicom_path)
#         image = apply_voi_lut(dicom_image.pixel_array, dicom_image)

#         # Convert to a more typical image format (optional step)
#         if dicom_image.PhotometricInterpretation == "MONOCHROME1":
#             image = np.amax(image) - image
#         image = image - np.min(image)
#         image = image / np.max(image)
#         image = (image * 255).astype(np.uint8)

#         # target = self.pneumonia_frame.iloc[idx, :][self.target_col]
#         target = self.targets[idx]
#         if self.transform:
#             image = self.transform(image)

#         return {'image': image, 'target': target}


