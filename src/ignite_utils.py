import collections
import os
import ignite
from torch.utils import data as data
from torch_utils import is_main_process

def epoch_chain(engine: ignite.engine.Engine, then_engine: ignite.engine.Engine, dataloader: data.DataLoader):
    @engine.on(ignite.engine.Events.EPOCH_COMPLETED)
    def on_complete(_):
        then_engine.run(dataloader)


def chain(engine: ignite.engine.Engine, then_engine: ignite.engine.Engine, dataloader: data.DataLoader):
    @engine.on(ignite.engine.Events.COMPLETED)
    def on_complete(_):
        then_engine.run(dataloader)


def log_epoch_results(engine: ignite.engine.Engine, name, trainer: ignite.engine.Engine):
    @engine.on(ignite.engine.Events.COMPLETED)
    def log(_):
        if is_main_process():
            metrics = engine.state.metrics
            avg_accuracy = metrics["accuracy"]
            avg_nll = metrics["nll"]
            avg_f1 = metrics["f1"]
            avg_precision = metrics["precision"]
            avg_recall = metrics["recall"]
            roc_auc = metrics["ROC_AUC"]
            prc_auc = metrics["PRC_AUC"]
            specificity = metrics["specificity"]
            print(
                f"{name} Results - Epoch: {trainer.state.epoch if trainer.state else 0}  "
                f"Avg accuracy: {avg_accuracy*100:.2f}%, "
                f"Avg loss: {avg_nll:.2f}, "
                f"Avg F1: {avg_f1*100:.2f}%, "
                f"Avg Precision: {avg_precision*100:.2f}%, "
                f"Avg Recall: {avg_recall*100:.2f}%, "
                f"ROC AUC: {roc_auc*100:.2f}%, "
                f"PRC AUC: {prc_auc*100:.2f}% ",
                f"Specificity: {specificity*100:.2f}%"
            )

def log_results(engine: ignite.engine.Engine, name):
    @engine.on(ignite.engine.Events.COMPLETED)
    def log(_):
        if is_main_process():
            metrics = engine.state.metrics
            avg_accuracy = metrics["accuracy"]
            avg_nll = metrics["nll"]
            avg_f1 = metrics["f1"]
            avg_precision = metrics["precision"]
            avg_recall = metrics["recall"]
            roc_auc = metrics["ROC_AUC"]
            prc_auc = metrics["PRC_AUC"]
            specificity = metrics["specificity"]

            print(
                f"{name} Results - "
                f"Avg accuracy: {avg_accuracy*100:.2f}%, "
                f"Avg loss: {avg_nll:.2f}, "
                f"Avg F1: {avg_f1*100:.2f}%, "
                f"Avg Precision: {avg_precision*100:.2f}%, "
                f"Avg Recall: {avg_recall*100:.2f}%, "
                f"ROC AUC: {roc_auc*100:.2f}%, "
                f"PRC AUC: {prc_auc*100:.2f}% ",
                f"Specificity: {specificity*100:.2f}%"
            )



def store_iteration_results(engine: ignite.engine.Engine, store_object, log_interval=1000):
    @engine.on(ignite.engine.Events.ITERATION_COMPLETED)
    def log(_):
        if engine.state.iteration % log_interval == 0:
            store_object.append(engine.state.output)


def store_epoch_results(engine: ignite.engine.Engine, store_object, name=None):
    @engine.on(ignite.engine.Events.EPOCH_COMPLETED)
    def log(_):
        metrics = engine.state.metrics
        if isinstance(store_object, collections.abc.MutableSequence):
            store_object.append(metrics)
        else:
            store_object[name] = metrics
