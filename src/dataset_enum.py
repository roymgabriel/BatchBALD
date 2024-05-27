from dataclasses import dataclass
import collections
import enum
import itertools

from torch import optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch.utils.data as data
import torch

from typing import List

import mnist_model
import resnet_model
import emnist_model
import vgg_model
from active_learning_data import ActiveLearningData
from torch_utils import get_balanced_sample_indices
from train_model import train_model
from transformed_dataset import TransformedDataset
import subrange_dataset

#TODO: Need to add RSNA dataset
from rsna_dataset import RSNAPneumoniaDataset
from sklearn.model_selection import train_test_split

# TODO: Need to add NIH Chest XRay Dataset

# TODO: Need to add EMORY COVID Dataset
from emorycovid_dataset import EmoryCOVIDDataset


@dataclass
class ExperimentData:
    active_learning_data: ActiveLearningData
    train_dataset: Dataset
    available_dataset: Dataset
    validation_dataset: Dataset
    test_dataset: Dataset
    initial_samples: List[int]


@dataclass
class DataSource:
    train_dataset: Dataset
    validation_dataset: Dataset = None
    test_dataset: Dataset = None
    shared_transform: object = None
    train_transform: object = None
    scoring_transform: object = None


def get_RSNA(target_col, root="./", train_pct=70, val_pct=10, test_pct=20, seed=9031):
    assert train_pct + val_pct + test_pct == 100, "The sum of the percentages must be 100."

    rsna_directory = root + "data/RSNA"
    rsna_mean = [0.5031, 0.5031, 0.5031]
    rsna_std = [0.2326, 0.2326, 0.2326]

    # # TODO: Check why this data is so much slower than the other datasets
    # # Define transformations for the images
    rsna_transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy array to PIL Image
        transforms.Resize((256, 256)),  # Resize images to 256x256
        transforms.Grayscale(3),
        transforms.ToTensor(),  # Convert images to Tensor
        transforms.Normalize(mean=rsna_mean, std=rsna_std)  # Normalization
    ])

    train_transform = transforms.Compose([
        transforms.RandomOrder([
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))], 1),
            transforms.RandomApply([transforms.RandomAffine(degrees=(-10, 10))], 1),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, scale=(.98, 1.02))], 1),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, shear=(-5, 5))], 1),
            transforms.RandomApply([transforms.RandomHorizontalFlip()], 1),
            ])
        ])


    # train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
    #                                       transforms.RandomHorizontalFlip()])


    # Load all labels
    full_dataset = RSNAPneumoniaDataset(
        csv_file=rsna_directory + '/panel_data_rsna.csv',
        root_dir=rsna_directory + '/imgs',
        target_col=target_col,
        transform=rsna_transform,
    )

    # Calculate split sizes
    test_size = test_pct / 100
    val_size = val_pct / (100 - test_pct) if test_pct < 100 else 0

    # Split dataset into train+val and test
    train_val_indices, test_indices = train_test_split(
        range(len(full_dataset)), test_size=test_size, random_state=seed, stratify=full_dataset.targets
    )

    # Split train+val into train and val if validation is needed
    if val_size > 0:
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_size, random_state=seed,
        )
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    else:
        train_indices = train_val_indices
        val_dataset = None

    # Create dataset views for training, validation (if exists), and testing
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    # # Determine split indices
    # n = len(full_dataset)
    # indices = np.random.RandomState(seed).permutation(n)
    # train_end = int(train_pct / 100 * n)
    # val_end = train_end + int(val_pct / 100 * n)

    # # Use the custom dataset splitting function
    # train_dataset, val_dataset, test_dataset = subrange_dataset.dataset_subset_split(full_dataset, [train_end, val_end])

    return DataSource(
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        test_dataset=test_dataset,
        train_transform=train_transform,
        shared_transform=rsna_transform,
    )


def get_EMORY_COVID(target_col, root="./", train_pct=70, val_pct=10, test_pct=20, seed=9031, data_type='mild_omit'):
    assert train_pct + val_pct + test_pct == 100, "The sum of the percentages must be 100."

    covid_directory = root + "data/EMORY_COVID"
    covid_mean = [0.4841, 0.4841, 0.4841]
    covid_std = [0.2256, 0.2256, 0.2256]

    # Define transformations for the images
    rsna_transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy array to PIL Image
        transforms.Resize((256, 256)),  # Resize images to 256x256
        transforms.Grayscale(3),
        transforms.ToTensor(),  # Convert images to Tensor
        transforms.Normalize(mean=covid_mean, std=covid_std)  # Normalization
    ])

    train_transform = transforms.Compose([
        transforms.RandomOrder([
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))], 1),
            transforms.RandomApply([transforms.RandomAffine(degrees=(-10, 10))], 1),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, scale=(.98, 1.02))], 1),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, shear=(-5, 5))], 1),
            transforms.RandomApply([transforms.RandomHorizontalFlip()], 1),
            ])
        ])


    # train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
    #                                       transforms.RandomHorizontalFlip()])


    # Load all labels
    if data_type == 'mild_omit':
        full_dataset = EmoryCOVIDDataset(
            csv_file=covid_directory + '/mild_dataset.csv',
            root_dir=covid_directory + '/imgs',
            target_col=target_col,
            transform=rsna_transform,
        )
    elif data_type == 'mild_merged':
        full_dataset = EmoryCOVIDDataset(
            # mild_merged_dataset
            csv_file=covid_directory + '/mild_dataset.csv',
            root_dir=covid_directory + '/imgs',
            target_col=target_col,
            transform=rsna_transform,
        )
    else:
        raise ValueError(f'data_type {data_type} parameter is not valid! Must be one of `mild_omit` or `mild_merged`!')

    # Calculate split sizes
    test_size = test_pct / 100
    val_size = val_pct / (100 - test_pct) if test_pct < 100 else 0

    # Split dataset into train+val and test
    train_val_indices, test_indices = train_test_split(
        range(len(full_dataset)), test_size=test_size, stratify=full_dataset.targets, random_state=seed
    )

    # Split train+val into train and val if validation is needed
    if val_size > 0:
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_size, random_state=seed
        )
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    else:
        train_indices = train_val_indices
        val_dataset = None

    # Create dataset views for training, validation (if exists), and testing
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    return DataSource(
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        test_dataset=test_dataset,
        train_transform=train_transform,
        shared_transform=rsna_transform,
    )



def get_CINIC10(root="./"):
    cinic_directory = root + "data/CINIC-10"
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]

    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
    shared_transform = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize(mean=cinic_mean,
                                                                std=cinic_std)])

    train_dataset = datasets.ImageFolder(cinic_directory + '/train')
    validation_dataset = datasets.ImageFolder(cinic_directory + '/valid')

    # Concatenate train and validation set to have more samples.
    merged_train_dataset = torch.utils.data.ConcatDataset([train_dataset, validation_dataset])

    test_dataset = datasets.ImageFolder(cinic_directory + '/test')

    return DataSource(
        train_dataset=merged_train_dataset,
        test_dataset=test_dataset,
        shared_transform=shared_transform,
        train_transform=train_transform,
    )


def get_MNIST():
    # num_classes=10, input_size=28
    transform = transforms.Compose([transforms.Resize((48, 48)),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat the single channel across RGB channels
                                    transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("data", train=False, transform=transform)

    return DataSource(train_dataset=train_dataset, test_dataset=test_dataset)


def get_RepeatedMNIST():
    # num_classes = 10, input_size = 28
    transform = transforms.Compose([transforms.Resize((48, 48)),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat the single channel across RGB channels
                                    transforms.Normalize((0.1307,), (0.3081,))])
    org_train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    train_dataset = data.ConcatDataset([org_train_dataset] * 3)

    test_dataset = datasets.MNIST("data", train=False, transform=transform)
    return DataSource(train_dataset=train_dataset, test_dataset=test_dataset)


class DatasetEnum(enum.Enum):
    mnist = "mnist"
    emnist = "emnist"
    emnist_bymerge = "emnist_bymerge"
    repeated_mnist_w_noise = "repeated_mnist_w_noise"
    repeated_mnist_w_noise2 = "repeated_mnist_w_noise2"
    repeated_mnist_w_noise5 = "repeated_mnist_w_noise5"
    mnist_w_noise = "mnist_w_noise"
    cinic10 = "cinic10"
    rsna_binary = "rsna_binary"
    rsna_multi = "rsna_multi"
    covid_binary = 'covid_binary'
    covid_multi = 'covid_multi'

    def get_data_source(self, seed):
        if self == DatasetEnum.mnist:
            return get_MNIST()
        elif self in (
                DatasetEnum.repeated_mnist_w_noise2,
                DatasetEnum.repeated_mnist_w_noise5,
                DatasetEnum.repeated_mnist_w_noise,
                DatasetEnum.mnist_w_noise,
        ):
            # num_classes=10, input_size=28
            num_repetitions = {
                DatasetEnum.mnist_w_noise: 1,
                DatasetEnum.repeated_mnist_w_noise: 3,
                DatasetEnum.repeated_mnist_w_noise2: 2,
                DatasetEnum.repeated_mnist_w_noise5: 5,
            }[self]

            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            org_train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)

            def apply_noise(idx, sample):
                data, target = sample
                return data + dataset_noise[idx], target

            dataset_noise = torch.empty(
                (len(org_train_dataset) * num_repetitions, 28, 28), dtype=torch.float32
            ).normal_(0.0, 0.1)

            train_dataset = TransformedDataset(
                data.ConcatDataset([org_train_dataset] * num_repetitions), transformer=apply_noise
            )

            test_dataset = datasets.MNIST("data", train=False, transform=transform)

            return DataSource(train_dataset=train_dataset, test_dataset=test_dataset)
        elif self in (DatasetEnum.emnist, DatasetEnum.emnist_bymerge):
            # num_classes=47, input_size=28,
            split = "balanced" if self == DatasetEnum.emnist else "bymerge"
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            train_dataset = datasets.EMNIST("emnist_data", split=split, train=True, download=True,
                                            transform=transform)

            test_dataset = datasets.EMNIST("emnist_data", split=split, train=False, transform=transform)

            """
                Table II contains a summary of the EMNIST datasets and
                indicates which classes contain a validation subset in the
                training set. In these datasets, the last portion of the training
                set, equal in size to the testing set, is set aside as a validation
                set. Additionally, this subset is also balanced such that it
                contains an equal number of samples for each task. If the
                validation set is not to be used, then the training set can be
                used as one contiguous set.
            """
            if self == DatasetEnum.emnist:
                # Balanced contains a test set
                split_index = len(train_dataset) - len(test_dataset)
                train_dataset, validation_dataset = subrange_dataset.dataset_subset_split(train_dataset,
                                                                                          split_index)
            else:
                validation_dataset = None
            return DataSource(
                train_dataset=train_dataset, test_dataset=test_dataset, validation_dataset=validation_dataset
            )
        elif self == DatasetEnum.cinic10:
            return get_CINIC10()
        elif self == DatasetEnum.rsna_binary:
            return get_RSNA('Target', root="./", train_pct=70, val_pct=10, test_pct=20, seed=seed)
        elif self == DatasetEnum.rsna_multi:
            return get_RSNA('class', root="./", train_pct=70, val_pct=10, test_pct=20, seed=seed)
        elif self == DatasetEnum.covid_binary:
            return get_EMORY_COVID('binary_target', root="./", train_pct=70, val_pct=10, test_pct=20, data_type='mild_omit', seed=seed)
        elif self == DatasetEnum.covid_multi:
            return get_EMORY_COVID('Median', root="./",train_pct=70, val_pct=10, test_pct=20, data_type='mild_merged', seed=seed)
        else:
            raise NotImplementedError(f"Unknown dataset {self}!")

    @property
    def num_classes(self):
        if self in (
                DatasetEnum.mnist,
                DatasetEnum.repeated_mnist_w_noise,
                DatasetEnum.repeated_mnist_w_noise2,
                DatasetEnum.repeated_mnist_w_noise5,
                DatasetEnum.mnist_w_noise,
        ):
            return 10
        elif self in (DatasetEnum.emnist, DatasetEnum.emnist_bymerge):
            return 47
        elif self == DatasetEnum.cinic10:
            return 10
        elif self == DatasetEnum.rsna_binary or self == DatasetEnum.covid_binary:
            return 2
        elif self == DatasetEnum.rsna_multi:
            return 3
        elif self == DatasetEnum.covid_multi:
            # 4
            return 3
        else:
            raise NotImplementedError(f"Unknown dataset {self}!")

    def create_bayesian_model(self, device):
        num_classes = self.num_classes
        if self in (
                DatasetEnum.mnist,
                DatasetEnum.repeated_mnist_w_noise,
                DatasetEnum.repeated_mnist_w_noise2,
                DatasetEnum.repeated_mnist_w_noise5,
                DatasetEnum.mnist_w_noise,
        ):
            # NOTE: Changed this to my resnet model
            # return mnist_model.BayesianNet(num_classes=num_classes).to(device)
            # return resnet_model.ResNetBay(num_classes=num_classes, pretrained=True).to(device)
            # return resnet_model_new.resnet18(pretrained=True, num_classes=10).to(device)
            # return vgg_model.vgg16_bn(pretrained=True, num_classes=num_classes).to(device)
            # return vgg_model.vgg16_cinic10_bn(pretrained=True, num_classes=num_classes).to(device)
            return resnet_model.resnet50(pretrained=False, num_classes=num_classes, bn=True).to(device)
        elif self in (DatasetEnum.emnist, DatasetEnum.emnist_bymerge):
            return emnist_model.BayesianNet(num_classes=num_classes).to(device)
        elif self == DatasetEnum.cinic10:
            return resnet_model.resnet50(pretrained=False, num_classes=num_classes, bn=True).to(device)
            # return vgg_model.vgg16_cinic10_bn(pretrained=True, num_classes=num_classes).to(device)
        elif self == DatasetEnum.rsna_binary or self == DatasetEnum.rsna_multi:
            return resnet_model.resnet50(pretrained=False, num_classes=num_classes, bn=True).to(device)
        elif self == DatasetEnum.covid_binary or self == DatasetEnum.covid_multi:
            return resnet_model.resnet50(pretrained=False, num_classes=num_classes, bn=True).to(device)
        else:
            raise NotImplementedError(f"Unknown dataset {self}!")

    def create_optimizer(self, model):
        if self == DatasetEnum.cinic10:
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
        elif self == DatasetEnum.rsna_binary or self == DatasetEnum.rsna_multi:
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
        elif self == DatasetEnum.covid_binary or self == DatasetEnum.covid_multi:
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
        else:
            optimizer = optim.Adam(model.parameters())
        return optimizer

    def create_train_model_extra_args(self, optimizer):
        return {}

    def create_lr_scheduler(self, optimizer):
        # NOTE: Set T_max to a large value so that LR does not reset in between acquisitions
        if self == DatasetEnum.rsna_binary or self == DatasetEnum.rsna_multi \
            or self == DatasetEnum.covid_binary or self == DatasetEnum.covid_multi:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR
            LR_SCHEDULER_PARAMETERS = {'T_max': 30, 'eta_min': 0, 'last_epoch': - 1}
            lr_scheduler = lr_scheduler(optimizer, **LR_SCHEDULER_PARAMETERS)
        else:
            lr_scheduler = None
        return lr_scheduler

    def train_model(
            self,
            train_loader,
            test_loader,
            validation_loader,
            num_inference_samples,
            max_epochs,
            early_stopping_patience,
            desc,
            log_interval,
            device,
            num_classes=num_classes,
            epoch_results_store=None,
    ):
        model = self.create_bayesian_model(device)
        optimizer = self.create_optimizer(model)
        # lr_scheduler = self.create_lr_scheduler(optimizer)
        num_epochs, test_metrics = train_model(
            model,
            optimizer,
            max_epochs,
            early_stopping_patience,
            num_inference_samples,
            test_loader,
            train_loader,
            validation_loader,
            log_interval,
            desc,
            device,
            num_classes=num_classes,
            epoch_results_store=epoch_results_store,
            # lr_scheduler=lr_scheduler,
            num_lr_epochs=1,
            **self.create_train_model_extra_args(optimizer),
        )
        return model, num_epochs, test_metrics


def get_experiment_data(
        data_source,
        num_classes,
        initial_samples,
        reduced_dataset,
        samples_per_class,
        validation_set_size,
        balanced_test_set,
        balanced_validation_set,
        store,
):
    train_dataset, test_dataset, validation_dataset = (
        data_source.train_dataset,
        data_source.test_dataset,
        data_source.validation_dataset,
    )

    active_learning_data = ActiveLearningData(train_dataset)
    if initial_samples is None:
        initial_samples = list(
            itertools.chain.from_iterable(
                get_balanced_sample_indices(
                    get_targets(train_dataset), num_classes=num_classes, n_per_digit=samples_per_class
                ).values()
            )
        )

    # Split off the validation dataset after acquiring the initial samples.
    active_learning_data.acquire(initial_samples)

    if validation_dataset is None:
        print("Acquiring validation set from training set.")
        if not validation_set_size:
            validation_set_size = len(test_dataset)

        if not balanced_validation_set:
            validation_dataset = active_learning_data.extract_dataset(validation_set_size)
        else:
            print("Using a balanced validation set")
            validation_dataset = active_learning_data.extract_dataset_from_indices(
                balance_dataset_by_repeating(
                    active_learning_data.available_dataset, num_classes, validation_set_size, upsample=False
                )
            )
    else:
        if validation_set_size == 0:
            print("Using provided validation set.")
            validation_set_size = len(validation_dataset)
        if validation_set_size < len(validation_dataset):
            print("Shrinking provided validation set.")
            if not balanced_validation_set:
                validation_dataset = data.Subset(
                    validation_dataset, torch.randperm(len(validation_dataset))[:validation_set_size].tolist()
                )
                store['Using a balanced validation set:'] = False
            else:
                print("Using a balanced validation set")
                validation_dataset = data.Subset(
                    validation_dataset,
                    balance_dataset_by_repeating(validation_dataset, num_classes, validation_set_size, upsample=False),
                )
                store['Using a balanced validation set:'] = True

    if balanced_test_set:
        print("Using a balanced test set")
        print("Distribution of original test set classes:")
        classes = get_target_bins(test_dataset)
        print(classes)

        test_dataset = data.Subset(
            test_dataset, balance_dataset_by_repeating(test_dataset, num_classes, len(test_dataset))
        )

        store['USING BALANCED DATASET'] = True
        store['Distribution of original test set classes:'] = classes

    if reduced_dataset:
        # Let's assume we won't use more than 1000 elements for our validation set.
        active_learning_data.extract_dataset(len(train_dataset) - max(len(train_dataset) // 20, 5000))
        test_dataset = subrange_dataset.SubrangeDataset(test_dataset, 0, max(len(test_dataset) // 10, 5000))
        if validation_dataset:
            validation_dataset = subrange_dataset.SubrangeDataset(validation_dataset, 0,
                                                                  len(validation_dataset) // 10)
        print("USING REDUCED DATASET!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        store['USING REDUCED DATASET'] = True

    show_class_frequencies = True
    if show_class_frequencies:
        print("Distribution of training set classes:")
        classes = get_target_bins(train_dataset)
        print(classes)
        store['Distribution of training set classes:'] = classes

        print("Distribution of validation set classes:")
        classes = get_target_bins(validation_dataset)
        print(classes)
        store['Distribution of validation set classes:'] = classes

        print("Distribution of test set classes:")
        classes = get_target_bins(test_dataset)
        print(classes)
        store['Distribution of test set classes:'] = classes

        print("Distribution of pool classes:")
        classes = get_target_bins(active_learning_data.available_dataset)
        print(classes)
        store['Distribution of pool classes:'] = classes

        print("Distribution of active set classes:")
        classes = get_target_bins(active_learning_data.active_dataset)
        print(classes)
        store['Distribution of active set classes:'] = classes

    print(f"Dataset info:")
    print(f"\t{len(active_learning_data.active_dataset)} active samples")
    print(f"\t{len(active_learning_data.available_dataset)} available samples")
    print(f"\t{len(validation_dataset)} validation samples")
    print(f"\t{len(test_dataset)} test samples")

    # store results in log file
    store['active samples'] = len(active_learning_data.active_dataset)
    store['available samples'] = len(active_learning_data.available_dataset)
    store['validation samples'] = len(validation_dataset)
    store['test samples'] = len(test_dataset)

    if data_source.shared_transform is not None or data_source.train_transform is not None:
        train_dataset = TransformedDataset(
            active_learning_data.active_dataset,
            vision_transformer=compose_transformers([data_source.train_transform, data_source.shared_transform]),
        )
    else:
        train_dataset = active_learning_data.active_dataset

    if data_source.shared_transform is not None or data_source.scoring_transform is not None:
        available_dataset = TransformedDataset(
            active_learning_data.available_dataset,
            vision_transformer=compose_transformers([data_source.scoring_transform, data_source.shared_transform]),
        )
    else:
        available_dataset = active_learning_data.available_dataset

    if data_source.shared_transform is not None:
        test_dataset = TransformedDataset(test_dataset, vision_transformer=data_source.shared_transform)
        validation_dataset = TransformedDataset(validation_dataset, vision_transformer=data_source.shared_transform)

    return ExperimentData(
        active_learning_data=active_learning_data,
        train_dataset=train_dataset,
        available_dataset=available_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        initial_samples=initial_samples,
    ), store


def compose_transformers(iterable):
    iterable = list(filter(None, iterable))
    if len(iterable) == 0:
        return None
    if len(iterable) == 1:
        return iterable[0]
    return transforms.Compose(iterable)


# TODO: move to utils?
def get_target_bins(dataset):
    classes = collections.Counter(int(target) for target in get_targets(dataset))
    return classes


# TODO: move to utils?
def balance_dataset_by_repeating(dataset, num_classes, target_size, upsample=True):
    balanced_samples_indices = get_balanced_sample_indices(get_targets(dataset), num_classes, len(dataset)).values()

    if upsample:
        num_samples_per_class = max(
            max(len(samples_per_class) for samples_per_class in balanced_samples_indices),
            target_size // num_classes
        )
    else:
        num_samples_per_class = min(
            max(len(samples_per_class) for samples_per_class in balanced_samples_indices),
            target_size // num_classes
        )

    def sample_indices(indices, total_length):
        return (torch.randperm(total_length) % len(indices)).tolist()

    balanced_samples_indices = list(
        itertools.chain.from_iterable(
            [
                [samples_per_class[i] for i in sample_indices(samples_per_class, num_samples_per_class)]
                for samples_per_class in balanced_samples_indices
            ]
        )
    )

    print(
        f"Resampled dataset ({len(dataset)} samples) to a balanced set of {len(balanced_samples_indices)} samples!")

    return balanced_samples_indices


# TODO: move to utils?
def get_targets(dataset):
    """Get the targets of a dataset without any target target transforms(!)."""
    if isinstance(dataset, TransformedDataset):
        return get_targets(dataset.dataset)
    if isinstance(dataset, data.Subset):
        targets = get_targets(dataset.dataset)
        return torch.as_tensor(targets)[dataset.indices]
    if isinstance(dataset, data.ConcatDataset):
        return torch.cat([get_targets(sub_dataset) for sub_dataset in dataset.datasets])

    if isinstance(
            dataset, (datasets.MNIST, datasets.ImageFolder,)
    ):
        return torch.as_tensor(dataset.targets)
    if isinstance(dataset, datasets.SVHN):
        return dataset.labels
    if isinstance(dataset, RSNAPneumoniaDataset):
        return torch.as_tensor(dataset.targets)
    if isinstance(dataset, EmoryCOVIDDataset):
        return torch.as_tensor(dataset.targets)

    raise NotImplementedError(f"Unknown dataset {dataset}!")
