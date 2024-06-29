import argparse
import functools
import itertools
import multiprocessing
import torch

import torch.utils.data as data

from blackhc import laaos

# NOTE(blackhc): get the directory right (oh well)
import blackhc.notebook

import torch_utils
from dataset_enum import DatasetEnum, get_experiment_data, get_targets, train_model
from random_fixed_length_sampler import RandomFixedLengthSampler
from active_learning_data import ActiveLearningData

import prettyprinter as pp
import logging
import sys

import os
from torch_utils import is_main_process
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group


# def ddp_setup():
#     init_process_group(backend="nccl")
#     torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

def create_experiment_config_argparser(parser):
    # Training settings
    parser.add_argument("--batch_size", type=int, default=64, help="input batch size for training")
    parser.add_argument("--scoring_batch_size", type=int, default=256, help="input batch size for scoring")
    parser.add_argument("--test_batch_size", type=int, default=256, help="input batch size for testing")
    parser.add_argument("--validation_set_size", type=int, default=128, help="validation set size")
    parser.add_argument(
        "--early_stopping_patience", type=int, default=5, help="# patience epochs for early stopping per iteration"
    )
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs to train")
    parser.add_argument("--epoch_samples", type=int, default=5056, help="number of epochs to train")
    parser.add_argument("--quickquick", action="store_true", default=False, help="uses a very reduced dataset")
    parser.add_argument(
        "--balanced_validation_set",
        action="store_true",
        default=False,
        help="uses a balanced validation set (instead of randomly picked)"
        "(and if no validation set is provided by the dataset)",
    )
    parser.add_argument("--num_inference_samples", type=int, default=5, help="number of samples for inference")
    parser.add_argument("--no_cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument(
        "--name", type=str, default="results", help="name for the results file (name of the experiment)"
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--train_dataset_limit",
        type=int,
        default=0,
        help="how much of the training set to use for training after splitting off the validation set (0 for all)",
    )
    parser.add_argument(
        "--balanced_training_set",
        action="store_true",
        default=False,
        help="uses a balanced training set (instead of randomly picked)"
        "(and if no validation set is provided by the dataset)",
    )
    parser.add_argument(
        "--balanced_test_set",
        action="store_true",
        default=False,
        help="force balances the test set---use with CAUTION!",
    )
    parser.add_argument(
        "--log_interval", type=int, default=10, help="how many batches to wait before logging training status"
    )
    parser.add_argument(
        "--dataset",
        type=DatasetEnum,
        default=DatasetEnum.mnist,
        help=f"dataset to use (options: {[f.name for f in DatasetEnum]})",
    )
    return parser

def main():
    # ddp_setup()

    parser = argparse.ArgumentParser(
        description="Pure training loop without AL",
        formatter_class=functools.partial(argparse.ArgumentDefaultsHelpFormatter, width=120),
    )
    parser.add_argument('--local_rank', type=int, default=0)

    parser = create_experiment_config_argparser(parser)

    args = parser.parse_args()

    store = laaos.create_file_store(
        args.name,
        suffix="",
        truncate=False,
        type_handlers=(blackhc.laaos.StrEnumHandler(), blackhc.laaos.ToReprHandler()),
    )

    if is_main_process():
        store["args"] = args.__dict__
        store["cmdline"] = sys.argv[:]

        print("|".join(sys.argv))
        print(args.__dict__)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    if is_main_process():
        print(f"Using {device} for computations")

    kwargs = {"num_workers": multiprocessing.cpu_count() - 1, "pin_memory": True} if use_cuda else {}

    dataset: DatasetEnum = args.dataset

    data_source = dataset.get_data_source(seed=args.seed)

    reduced_train_length = args.train_dataset_limit

    experiment_data, store = get_experiment_data(
        data_source,
        dataset.num_classes,
        None,
        False,
        0,
        args.validation_set_size,
        args.balanced_test_set,
        args.balanced_validation_set,
        store=store
    )

    if not reduced_train_length:
        reduced_train_length = len(experiment_data.available_dataset)

    if is_main_process():
        print(f"Training with reduced dataset of {reduced_train_length} data points")

    if not args.balanced_training_set:
        experiment_data.active_learning_data.acquire(
            experiment_data.active_learning_data.get_random_available_indices(reduced_train_length)
        )
    else:
        if is_main_process():
            print("Using a balanced training set.")
        num_samples_per_class = reduced_train_length // dataset.num_classes
        experiment_data.active_learning_data.acquire(
            list(
                itertools.chain.from_iterable(
                    torch_utils.get_balanced_sample_indices(
                        get_targets(experiment_data.available_dataset), dataset.num_classes, num_samples_per_class
                    ).values()
                )
            )
        )

    if len(experiment_data.train_dataset) < args.epoch_samples:
        sampler = RandomFixedLengthSampler(experiment_data.train_dataset, args.epoch_samples)
    else:
        sampler = data.RandomSampler(experiment_data.train_dataset)

    test_loader = torch.utils.data.DataLoader(
        experiment_data.test_dataset, batch_size=args.test_batch_size, shuffle=False,
        # sampler=DistributedSampler(experiment_data.test_dataset),
        **kwargs
    )
    train_loader = torch.utils.data.DataLoader(
        experiment_data.train_dataset, sampler=sampler,
        batch_size=args.batch_size, **kwargs
    )

    validation_loader = torch.utils.data.DataLoader(
        experiment_data.validation_dataset, batch_size=args.test_batch_size, shuffle=False,
        # sampler=DistributedSampler(experiment_data.validation_dataset),
        **kwargs
    )

    def desc(name):
        return lambda engine: "%s" % name

    dataset.train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        validation_loader=validation_loader,
        num_inference_samples=args.num_inference_samples,
        max_epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        desc=desc,
        log_interval=args.log_interval,
        device=device,
        num_classes=dataset.num_classes,
        epoch_results_store=store,
        # gpu_count=torch.cuda.device_count(),
        gpu_count=1,
        # gpu_id=int(os.environ["LOCAL_RANK"]),
        gpu_id=0,
    )

    # destroy_process_group()
    # print("DONE")


if __name__ == "__main__":
    import os
    # Set NCCL_P2P_DISABLE environment variable
    # os.environ['NCCL_P2P_DISABLE'] = '1'

    # Set OMP_NUM_THREADS to a desired value
    # os.environ["OMP_NUM_THREADS"] = "1"

    #!/usr/bin/env python3
    main()
