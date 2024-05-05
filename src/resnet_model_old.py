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
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__(num_classes)


        # Select ResNet model
        # if cfg.RESNET_MODEL == 'resnet18':
        #     self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if cfg.MODEL_WEIGHTS == 'DEFAULT' else None)
        # elif cfg.RESNET_MODEL == 'resnet50':
        #     self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if cfg.MODEL_WEIGHTS == 'DEFAULT' else None)
        # else:
        #     raise ValueError("Unsupported ResNet model type!")

        # self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.model = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)

        # change the number of input channels to 1 instead of 3
        # TODO: Check if this resets weights?
        # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify the classifier layer
        # self.num_ftrs = self.model.fc.in_features
        # self.model.fc = nn.Linear(num_ftrs, num_classes)

        # Depending on the dropout_type, add dropout layers

            # self.model.fc = nn.Sequential(
            #     MCDropout(),
            #     nn.Linear(self.num_ftrs, self.num_classes)
            # ).to(self.device)

        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model.classifier = nn.Sequential(

            # nn.Linear(512 * 7 * 7, 4096),
                nn.Linear(512 * 1 * 1, 4096),
                nn.ReLU(True),
                MCDropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                MCDropout(),
                nn.Linear(4096, num_classes),
            )

            # # Pass in last fully connected layer with dropout then lineary
            # # (0): MCDropout(p=0.5)
            # # (1): Linear(in_features=512, out_features=10, bias=True)
            # return F.log_softmax(self.model.fc(x), dim=1)
        # elif self.dropout_type == 'all':
        #     # Insert dropout after every block
        #     for name, child in self.model.named_children():
        #         if isinstance(child, nn.Sequential):
        #             for block_name, block in child.named_children():
        #                 setattr(child, block_name, nn.Sequential(
        #                     block,
        #                     MCDropout2d()
        #                 ))
        #     # Add dropout before the final fully connected layer
        #     self.model.fc = nn.Sequential(
        #         MCDropout(),
        #         nn.Linear(self.num_ftrs, num_classes)
        #     ).to(self.device)



    def deterministic_forward_impl(self, x: torch.Tensor):
        # # input shape is torch.Size([32, 3, 48, 48])
        # x = self.model.conv1(x)
        # # torch.Size([32, 64, 24, 24])
        # x = self.model.bn1(x)
        # # torch.Size([32, 64, 24, 24])
        # x = self.model.relu(x)
        # # torch.Size([32, 64, 24, 24])
        # x = self.model.maxpool(x)
        # # torch.Size([32, 64, 12, 12])

        # x = self.model.layer1(x)
        # # # torch.Size([32, 64, 12, 12])
        # x = self.model.layer2(x)
        # # torch.Size([32, 128, 6, 6])
        # x = self.model.layer3(x)
        # # torch.Size([32, 256, 3, 3])
        # x = self.model.layer4(x)
        # # torch.Size([32, 512, 2, 2])
        # x = self.model.avgpool(x)
        # # torch.Size([32, 512, 1, 1])
        # x = torch.flatten(x, 1)
        # # torch.Size([32, 512])
        # # NOTE: we ignore last layer since this is where dropout would occur
        # # x = self.model.fc(x)
        # # # torch.Size([32, num_classes])

        # return x
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def mc_forward_impl(self, x: torch.Tensor):
        x = self.model.classifier(x)
        return F.log_softmax(x, dim=1)




