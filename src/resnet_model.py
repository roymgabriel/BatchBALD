import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torch import Tensor
import mc_dropout

__all__ = ["ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(mc_dropout.BayesianModule):
    def __init__(self, block, layers, num_classes=1000, init_weights=True, bn=False):
        super().__init__(num_classes)
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if bn:
            self.fc = nn.Sequential(
                mc_dropout.MCDropout(),
                nn.Linear(512 * block.expansion, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(True),
                nn.Linear(512, num_classes),
            )
        else:
            self.fc = nn.Sequential(
                mc_dropout.MCDropout(),
                nn.Linear(512 * block.expansion, num_classes),
            )


        if init_weights:
            self.apply(self.initialize_weights)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


    def deterministic_forward_impl(self, x: Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def mc_forward_impl(self, x: Tensor):
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


    @staticmethod
    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

resnet_cfgs = {
    '18': [2, 2, 2, 2],     # ResNet-18
    '34': [3, 4, 6, 3],     # ResNet-34
    '50': [3, 4, 6, 3],     # ResNet-50, uses Bottleneck
    '101': [3, 4, 23, 3],   # ResNet-101, uses Bottleneck
    '152': [3, 8, 36, 3],   # ResNet-152, uses Bottleneck
}

def _resnet(arch, block, pretrained, progress, **kwargs):
    if pretrained:
        kwargs["init_weights"] = False
    model = ResNet(block, resnet_cfgs[arch], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[f'resnet{arch}'], progress=progress)

        # Remove weights related to the fully connected layer
        state_dict.pop('fc.weight', None)
        state_dict.pop('fc.bias', None)

        model.load_state_dict(state_dict, strict=False)

    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model."""
    return _resnet('18', BasicBlock, pretrained, progress, **kwargs)

def resnet34(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model."""
    return _resnet('34', BasicBlock, pretrained, progress, **kwargs)

def resnet50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model."""
    return _resnet('50', Bottleneck, pretrained, progress, **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model."""
    return _resnet('101', Bottleneck, pretrained, progress, **kwargs)

def resnet152(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-152 model."""
    return _resnet('152', Bottleneck, pretrained, progress, **kwargs)
