import ignite
from torch import optim as optim
from torch.nn import functional as F
from torch import nn
import torch

import ignite_restoring_score_guard
from ignite_progress_bar import ignite_progress_bar
from ignite_utils import epoch_chain, chain, log_epoch_results, store_epoch_results, store_iteration_results
from sampler_model import SamplerModel, NoDropoutModel
from typing import NamedTuple

from metrics_utils import *
from torch.profiler import profile, record_function, ProfilerActivity
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


def is_main_process():
    return int(os.environ.get("LOCAL_RANK", 0)) == 0

class TrainModelResult(NamedTuple):
    num_epochs: int
    test_metrics: dict


def build_metrics(num_classes, test_dtype=None):
    # NOTE: COVID WEIGHTS ONLY
    if num_classes == 2:
        # for binary classes
        class_weights = torch.tensor([1/0.140578, 1/0.859422], dtype=torch.float)
    elif num_classes == 4:
        # Uncomment this for multi (mild included)
        class_weights = torch.tensor([1/0.34765625, 1/0.3031851 , 1/0.25120192, 1/0.09795673], dtype=torch.float)
    elif num_classes == 3:
        # Use this for multi (mild omitted)
        class_weights = torch.tensor([1/0.498922, 1/0.360500 , 1/0.140578], dtype=torch.float)

    # # NOTE: RSNA WEIGHTS ONLY
    # if num_classes == 2:
    #     # for binary classes
    #     class_weights = torch.tensor([1/0.683892, 1/0.316108], dtype=torch.float)
    # elif num_classes == 3:
    #     # Use this for multi
    #     class_weights = torch.tensor([1/0.316108, 1/0.391074 , 1/0.292818], dtype=torch.float)

    class_weights = class_weights / class_weights.sum()  # Normalize weights

    return {
        "accuracy": Accuracy(),
        # "nll": Loss(F.nll_loss),
        "nll": Loss(WeightedNLLLoss(weight=class_weights, test_dtype=test_dtype)),
        "f1": F1Score(),
        "precision": Precision(average=True),
        "recall": Recall(average=True),
        "ROC_AUC": ROC_AUC(num_classes=num_classes),
        "PRC_AUC": PRC_AUC(num_classes=num_classes),
        "specificity": Specificity(num_classes=num_classes),
    }


def train_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    max_epochs,
    early_stopping_patience,
    num_inference_samples,
    test_loader,
    train_loader,
    validation_loader,
    log_interval,
    desc,
    device,
    num_classes,
    gpu_count,
    gpu_id=int(os.environ["LOCAL_RANK"]),
    lr_scheduler: optim.lr_scheduler._LRScheduler = None,
    num_lr_epochs=0,
    epoch_results_store=None,
) -> TrainModelResult:
    if gpu_count > 1:
        # model = nn.DataParallel(model)
        model = model.to(gpu_id)
        model = DDP(model, device_ids=[gpu_id])
        if is_main_process():
            print('Model is using', torch.cuda.device_count(), 'GPUs.')
    else:
        print('Using a single GPU or CPU.')

    # Ensure the model is on the correct device
    # model = model.to(device)

    test_sampler = SamplerModel(model.module if gpu_count > 1 else model, k=min(num_inference_samples, 100)).to(gpu_id)
    validation_sampler = NoDropoutModel(model.module if gpu_count > 1 else model).to(gpu_id)
    training_sampler = SamplerModel(model.module if gpu_count > 1 else model, k=1).to(gpu_id)


    # NOTE: COVID WEIGHTS ONLY
    if num_classes == 2:
        # for binary classes
        class_weights = torch.tensor([1/0.140578, 1/0.859422], dtype=torch.float)
    elif num_classes == 4:
        # Uncomment this for multi (mild included)
        class_weights = torch.tensor([1/0.34765625, 1/0.3031851 , 1/0.25120192, 1/0.09795673], dtype=torch.float)
    elif num_classes == 3:
        # Use this for multi (mild omitted)
        class_weights = torch.tensor([1/0.498922, 1/0.360500 , 1/0.140578], dtype=torch.float)

    # NOTE: RSNA WEIGHTS ONLY
    # if num_classes == 2:
    #     # for binary classes
    #     class_weights = torch.tensor([1/0.683892, 1/0.316108], dtype=torch.float)
    # elif num_classes == 3:
    #     # Use this for multi
    #     class_weights = torch.tensor([1/0.316108, 1/0.391074 , 1/0.292818], dtype=torch.float)

    class_weights = class_weights / class_weights.sum()  # Normalize weights


    # # Start profiling
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #              schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    #              on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profiler'),
    #              record_shapes=True,
    #              profile_memory=True,
    #              with_stack=True) as prof:

    if gpu_count > 1:
        gradient_accumulation_steps = 4
    else:
        gradient_accumulation_steps = 1

    trainer = ignite.engine.create_supervised_trainer(training_sampler, optimizer,
                                                      WeightedNLLLoss(weight=class_weights),
                                                      device=device,
                                                      gradient_accumulation_steps=gradient_accumulation_steps)
    validation_evaluator = ignite.engine.create_supervised_evaluator(
        validation_sampler, metrics=build_metrics(num_classes=num_classes), device=device
    )

    def out_of_patience():
        # NOTE: the learning rate scheduler will change during epoch training and reset to original value after new
        # acquisition batch is added. We need to adjust this somehow that it either continues to decrease till end of acquisition
        # or it resets but to a smaller value than initial LR value. So if we started at 1e-3 and then scheduled. Then we acquired
        # a new batch of data, the new LR should start at < 1e-3.
        nonlocal num_lr_epochs
        if num_lr_epochs <= 0 or lr_scheduler is None:
            trainer.terminate()
        else:
            lr_scheduler.step()
            restoring_score_guard.patience = int(restoring_score_guard.patience * 1.5 + 0.5)
            if is_main_process():
                print(f"New LRs: {[group['lr'] for group in optimizer.param_groups]}")
                print(f"num_lr_epochs: {num_lr_epochs}")
            num_lr_epochs -= 1

    if lr_scheduler is not None and is_main_process():
        print(f"LRs: {[group['lr'] for group in optimizer.param_groups]}")
        print(f"num_lr_epochs: {num_lr_epochs}")

    restoring_score_guard = ignite_restoring_score_guard.RestoringScoreGuard(
        patience=early_stopping_patience,
        score_function=lambda engine: engine.state.metrics["accuracy"],
        out_of_patience_callback=out_of_patience,
        module=model.module if gpu_count > 1 else model,
        optimizer=optimizer,
        training_engine=trainer,
        validation_engine=validation_evaluator,
    )

    if test_loader is not None:

        test_evaluator = ignite.engine.create_supervised_evaluator(test_sampler,
                                                                metrics=build_metrics(num_classes=num_classes, test_dtype='test'),
                                                                device=device)
        ignite_progress_bar(test_evaluator, desc("Test Eval"), log_interval)
        chain(trainer, test_evaluator, test_loader)
        log_epoch_results(test_evaluator, "Test", trainer)

    ignite_progress_bar(trainer, desc("Training"), log_interval)
    ignite_progress_bar(validation_evaluator, desc("Validation Eval"), log_interval)

    # NOTE(blackhc): don't run a full test eval after every epoch.
    # epoch_chain(trainer, test_evaluator, test_loader)

    epoch_chain(trainer, validation_evaluator, validation_loader)

    log_epoch_results(validation_evaluator, "Validation", trainer)

    if epoch_results_store is not None:
        epoch_results_store["validations"] = []
        epoch_results_store["losses"] = []
        store_epoch_results(validation_evaluator, epoch_results_store["validations"])
        store_iteration_results(trainer, epoch_results_store["losses"], log_interval=2)

        if test_loader is not None:
            store_epoch_results(test_evaluator, epoch_results_store, name="test")

    if len(train_loader.dataset) > 0:
        trainer.run(train_loader, max_epochs)
    else:
        test_evaluator.run(test_loader)

    num_epochs = trainer.state.epoch if trainer.state else 0

    test_metrics = None
    if test_loader is not None:
        test_metrics = test_evaluator.state.metrics

    return TrainModelResult(num_epochs, test_metrics)
