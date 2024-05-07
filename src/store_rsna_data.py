import os
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
from PIL import Image

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import rsna_dataset
from torchvision.transforms import functional as F



def convert_dicom_to_png(dicom_directory, output_directory):
    for filename in os.listdir(dicom_directory):
        if filename.endswith(".dcm"):
            dicom_path = os.path.join(dicom_directory, filename)
            image_path = os.path.join(output_directory, filename.replace(".dcm", ".png"))

            dicom_image = pydicom.dcmread(dicom_path)
            image = apply_voi_lut(dicom_image.pixel_array, dicom_image)

            if dicom_image.PhotometricInterpretation == "MONOCHROME1":
                image = np.amax(image) - image

            image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255.0
            image = Image.fromarray(image.astype(np.uint8))
            image.save(image_path)


# Example usage:
# convert_dicom_to_png("./data/RSNA/stage_2_train_images", "./data/RSNA/imgs/")


def calculate_mean_std(dataset):
    """
    Calculates mean and std of RSNA dataset
    """
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    mean = 0.0
    std = 0.0
    for _, images in enumerate(loader):
        image = images['image']
        batch_samples = image.size(0)  # batch size (the last batch can have smaller size!)
        image = image.view(batch_samples, image.size(1), -1)
        mean += image.mean(2).sum(0)
        std += image.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)

    return mean, std




if __name__ == "__main__":
    print(os.getcwd())
    # print(os.listdir('./data/'))
    # print(os.listdir('../data/'))
    root = '/home/adibi/rmg7/'
    rsna_directory = root + "data/RSNA"
    target_col = 'class'
    # transform = transforms.Compose([transforms.ToTensor()])
    rsna_transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy array to PIL Image
        transforms.Resize((256, 256)),  # Resize images to 256x256
        transforms.Grayscale(3),
        transforms.ToTensor(),  # Convert images to Tensor
        # transforms.Normalize(mean=rsna_mean, std=rsna_std)  # Normalization
    ])
    # dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataset = rsna_dataset.RSNAPneumoniaDataset(
            csv_file=rsna_directory + '/panel_data_rsna.csv',
            root_dir=rsna_directory + '/imgs',
            target_col=target_col,
            transform=rsna_transform,    )
    print(dataset)

    mean, std = calculate_mean_std(dataset)
    print(f"Target Col {target_col}")
    print(f'Mean: {mean}')
    print(f'Std Dev: {std}')

