from ignite.metrics import Accuracy, Loss, Precision, Recall, Metric, ConfusionMatrix
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

import torch.nn as nn
import numpy as np
import torch

class WeightedNLLLoss(nn.Module):
    def __init__(self, weight=None, test_dtype=None):
        super(WeightedNLLLoss, self).__init__()
        if test_dtype:
            # for testing eval we use type double
            self.weight = weight.cuda().double()
        else:
            self.weight = weight.cuda()
        self.nll_loss = nn.NLLLoss(weight=self.weight)

    def forward(self, y_pred, y):
        return self.nll_loss(y_pred, y)

class F1Score(Metric):
    def __init__(self, output_transform=lambda x: x):
        self.precision = Precision(output_transform=output_transform, average=False)
        self.recall = Recall(output_transform=output_transform, average=False)
        super(F1Score, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self.precision.reset()
        self.recall.reset()

    @reinit__is_reduced
    def update(self, output):
        self.precision.update(output)
        self.recall.update(output)

    @sync_all_reduce()
    def compute(self):
        p = self.precision.compute()
        r = self.recall.compute()
        return (2 * (p * r) / (p + r)).nanmean()

class PRC_AUC(Metric):
    def __init__(self, num_classes: int, output_transform=lambda x: x):
        super(PRC_AUC, self).__init__(output_transform=output_transform)
        self._precision = []
        self._recall = []
        self.num_classes = num_classes
    @reinit__is_reduced
    def reset(self):
        self._precision = []
        self._recall = []
        super(PRC_AUC, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        # ROY: Assuming y has shape (n_samples,) and y_pred has shape (n_samples, n_classes) with probabilities
        # we binarize the labels to make sure the sklearn package runs successfully
        y_binary = label_binarize(y.cpu().numpy(), classes=np.arange(self.num_classes))
        if self.num_classes == 2:
            # only use the logits (or softmax probabilities .softmax(1)) of the positive class hence [:, 1]
            precision, recall, _ = precision_recall_curve(y_binary.ravel(), y_pred.cpu().numpy()[:, 1])
        else:
            # get micro precision recall scores
            precision, recall, _ = precision_recall_curve(y_binary.ravel(), y_pred.cpu().numpy().ravel())
        self._precision.append(precision)
        self._recall.append(recall)

    @sync_all_reduce()
    def compute(self):
        # Flatten the list of precisions and recalls
        precisions = np.concatenate(self._precision)
        recalls = np.concatenate(self._recall)
        # Sort all FPRs and corresponding TPRs
        sorted_indexes = np.argsort(recalls)
        recalls = recalls[sorted_indexes]
        precisions = precisions[sorted_indexes]
        # Use sklearn's auc function to calculate the area under the curve
        return auc(recalls, precisions)

class ROC_AUC(Metric):
    def __init__(self, num_classes: int, output_transform=lambda x: x):
        super(ROC_AUC, self).__init__(output_transform=output_transform)
        self.num_classes = num_classes
        self._fpr = []
        self._tpr = []

    @reinit__is_reduced
    def reset(self):
        self._fpr = []
        self._tpr = []
        super(ROC_AUC, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        # Convert y to a binary format needed for multiclass ROC computation
        y_binary = label_binarize(y.cpu().numpy(), classes=np.arange(self.num_classes))
        # Compute ROC curve data for each class and micro-average
        if self.num_classes == 2:
            fpr, tpr, _ = roc_curve(y_binary, y_pred.cpu().numpy()[:, 1])
        else:
            fpr, tpr, _ = roc_curve(y_binary.ravel(), y_pred.cpu().numpy().ravel())
        self._fpr.append(fpr)
        self._tpr.append(tpr)

    @sync_all_reduce()
    def compute(self):
        # Concatenate all FPRs and TPRs
        all_fpr = np.concatenate(self._fpr)
        all_tpr = np.concatenate(self._tpr)
        # Sort all FPRs and corresponding TPRs
        sorted_indexes = np.argsort(all_fpr)
        all_fpr = all_fpr[sorted_indexes]
        all_tpr = all_tpr[sorted_indexes]
        # Calculate AUC using sklearn's auc function
        return auc(all_fpr, all_tpr)


class Specificity(Metric):
    def __init__(self, num_classes: int, output_transform=lambda x: x):
        self.confusion_matrix = ConfusionMatrix(num_classes=num_classes)
        self.num_classes = num_classes
        super(Specificity, self).__init__(output_transform)

    @reinit__is_reduced
    def reset(self):
        self.confusion_matrix.reset()

    @reinit__is_reduced
    def update(self, output):
        self.confusion_matrix.update(output)

    @sync_all_reduce()
    def compute(self):
        cm = self.confusion_matrix.compute().cpu().numpy()
        specificities = []
        for i in range(self.num_classes):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificities.append(specificity)
        return torch.tensor(specificities).mean().item()
