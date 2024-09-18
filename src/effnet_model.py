import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torch import Tensor
import mc_dropout

__all__ = ["EffNet", "effnet_b0", "effnet_b1", "effnet_b2", "effnet_b3", "effnet_b4"]

model_urls = {
    'effnet_b0': 'https://download.pytorch.org/models/efficientnet_b0_rwightman-bdbadd66.pth',
    # Additional model URLs can be added here
}

class MBConvBlock(nn.Module):
    # Basic MBConv block for EfficientNet
    def __init__(self, in_channels, out_channels, expansion_factor, stride, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        mid_channels = in_channels * expansion_factor

        # Expansion phase (inverted bottleneck)
        self.expand_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False) if expansion_factor != 1 else None
        self.bn0 = nn.BatchNorm2d(mid_channels)

        # Depthwise convolution phase
        self.depthwise_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=mid_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # Squeeze and Excitation layer, if required, can be added here

        # Output phase
        self.project_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip_connection = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        identity = x
        if self.expand_conv:
            x = F.relu6(self.bn0(self.expand_conv(x)), inplace=True)

        x = F.relu6(self.bn1(self.depthwise_conv(x)), inplace=True)
        x = self.bn2(self.project_conv(x))

        if self.skip_connection:
            x += identity

        return x

# Assuming MBConvBlock is defined as previously described

class EffNet(mc_dropout.BayesianModule):
    def __init__(self, cfg, num_classes=1000, init_weights=True):
        super().__init__(num_classes)
        self.cfg = cfg  # Configuration for each block
        self.features = self._make_layers(MBConvBlock, cfg)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)  # dropout rate might change depending on the model
        self.fc = nn.Linear(cfg[-1][1], num_classes)  # The output dimension of the last block

        if init_weights:
            self.apply(self.initialize_weights)

    def _make_layers(self, block, cfg):
        layers = []
        for exp_size, out_channels, num_blocks, stride in cfg:
            for _ in range(num_blocks):
                stride = stride if _ == 0 else 1
                layers.append(block(self.in_channels, out_channels, exp_size, stride))
                self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    @staticmethod
    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

# Configuration for each variant
# (expansion_factor, out_channels, num_blocks, stride)
effnet_configs = {
    'b0': [(1, 16, 1, 1), (6, 24, 2, 2), (6, 40, 2, 2), (6, 80, 3, 2),
           (6, 112, 3, 1), (6, 192, 4, 2), (6, 320, 1, 2)],
    # Add other configurations for b1, b2, ..., b7
}

def effnet(version, num_classes=1000, pretrained=False, **kwargs):
    """Constructs an EfficientNet model."""
    model = EffNet(effnet_configs[version], num_classes=num_classes, **kwargs)
    if pretrained:
        model_url = f'https://download.pytorch.org/models/efficientnet_{version}_rwightman.pth'
        state_dict = load_state_dict_from_url(model_url)
        model.load_state_dict(state_dict)
    return model

# # Example to create an EfficientNet-B0 model
# model_b0 = effnet('b0', num_classes=1000, pretrained=False)
