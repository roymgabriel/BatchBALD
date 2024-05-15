import os
from torchvision import transforms
from torch.utils.data import DataLoader
import emorycovid_dataset


def calculate_mean_std(dataset):
    """
    Calculates mean and std of EMORY COVID dataset
    """
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    mean = 0.0
    std = 0.0
    for _, images in enumerate(loader):
        print(images)
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
    covid_directory = root + "data/EMORY_COVID"
    target_col = 'binary_target'
    # transform = transforms.Compose([transforms.ToTensor()])
    emory_transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy array to PIL Image
        transforms.Resize((256, 256)),  # Resize images to 256x256
        transforms.Grayscale(3),
        transforms.ToTensor(),  # Convert images to Tensor
        # transforms.Normalize(mean=rsna_mean, std=rsna_std)  # Normalization
    ])
    # NOTE: change the directory csv file based on task at hand
    dataset = emorycovid_dataset.EmoryCOVIDDataset(
            csv_file=covid_directory + '/mild_dataset.csv',
            root_dir=covid_directory + '/imgs',
            target_col=target_col,
            transform=emory_transform)
    print(dataset)

    mean, std = calculate_mean_std(dataset)
    print(f"Target Col {target_col}")
    print(f'Mean: {mean}')
    print(f'Std Dev: {std}')

