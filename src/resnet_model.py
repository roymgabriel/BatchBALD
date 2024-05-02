import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from mc_dropout import BayesianModule, MCDropout, MCDropout2d


class ResNetBay(BayesianModule):
    """
    Implements ResNet with MC Dropout...inspired by https://github.com/BlackHC/batchbald_redux/blob/master/03_consistent_mc_dropout.ipynb

    if `dropout_type` is set to '', then it is just a regular ResNet model
    elif set to 'last', then only last FC will include MCDropout
    elif set to 'all', then not only last FC will include MCDropout but also after every block in every layer
    a MCDropout2d will be placed. See example model output by `print(model)` after using example below.

    Example Usage:
    # Create model with dropout at the last layer using ResNet18
    >>> model_last = ResNetBay(num_classes=3, dropout_type='last', resnet_model='resnet18')
    >>> model_all = ResNetBay(num_classes=3, dropout_type='all', resnet_model='resnet50')

    >>> # model_last.to('cuda:0')
    >>> # model_all.to('cuda:0')

    >>> # Example input tensor
    >>> input_tensor = torch.randn(16, 3, 224, 224) # Standard size for ResNet

    # Forward pass
    >>> model_last.train()  # Ensure to set to train mode to enable dropout functionality
    >>> output_last = model_last.forward(input_tensor, k=20)  # Assuming k is defined in the forward method

    >>> model_all.train()
    >>> output_all = model_all.forward(input_tensor, k=20)

    >>> print(output_last.shape, output_all.shape)  # Output shapes

    >>> print("MC DROPOUT IN LAST LAYER ONLY:")
    >>> print(output_last.mean(dim=1))
    >>> print(output_last.std(dim=1))
    >>> print()
    >>> print("MC DROPOUT IN ALL LAYERS INCLUDING LAST LAYER:")
    >>> print(output_all.mean(dim=1))
    >>> print(output_all.std(dim=1))


    """
    def __init__(self, num_classes=10, pretrained=True, dropout_type='last'):
        super().__init__(num_classes)

        # Select ResNet model
        # if cfg.RESNET_MODEL == 'resnet18':
        #     self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if cfg.MODEL_WEIGHTS == 'DEFAULT' else None)
        # elif cfg.RESNET_MODEL == 'resnet50':
        #     self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if cfg.MODEL_WEIGHTS == 'DEFAULT' else None)
        # else:
        #     raise ValueError("Unsupported ResNet model type!")

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # change the number of input channels to 1 instead of 3
        # TODO: Check if this resets weights?
        # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify the classifier layer
        # num_ftrs = self.model.fc.in_features
        # self.model.fc = nn.Linear(num_ftrs, num_classes)



    def deterministic_forward_impl(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def mc_forward_impl(self, x: torch.Tensor):
        # Depending on the dropout_type, add dropout layers
        if self.dropout_type == 'last':
            self.model.fc = nn.Sequential(
                MCDropout(),
                nn.Linear(num_ftrs, num_classes)
            )
            # Pass in last fully connected layer with dropout then lineary
            # (0): MCDropout(p=0.5)
            # (1): Linear(in_features=512, out_features=10, bias=True)
            return F.log_softmax(self.fc(x), dim=1)
        elif self.dropout_type == 'all':
            # Insert dropout after every block
            for name, child in self.model.named_children():
                if isinstance(child, nn.Sequential):
                    for block_name, block in child.named_children():
                        setattr(child, block_name, nn.Sequential(
                            block,
                            MCDropout2d()
                        ))
            # Add dropout before the final fully connected layer
            self.model.fc = nn.Sequential(
                MCDropout(),
                nn.Linear(num_ftrs, num_classes)
            )

            # pass in all layers with new dropouts
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            x = x.view(x.size(0), -1)
            return x

        else:
            raise NotImplementedError("only handles last and all")




