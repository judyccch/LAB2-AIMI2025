import torch.nn as nn


# TODO implement EEGNet model
"""
class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        pass

    def forward(self, x):
        pass
"""


class EEGNet(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.25,
                 activation='elu', elu_alpha=1.0,
                 temporal_filters=16, spatial_filters=32,
                 bn_momentum=0.1):
        super().__init__()

        # activation function
        if activation.lower() == 'elu':
            act_layer = nn.ELU(alpha=elu_alpha, inplace=True)
        elif activation.lower() == 'leakyrelu':
            act_layer = nn.LeakyReLU(0.1, inplace=True)
        elif activation.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif activation.lower() == 'gelu':
            act_layer = nn.GELU()
        elif activation.lower() == 'mish':
            act_layer = nn.Mish()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # 第一層 temporal conv
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, temporal_filters, kernel_size=(1, 51), padding=(0, 25), bias=False),
            nn.BatchNorm2d(temporal_filters, eps=1e-5, momentum=bn_momentum)
        )

        # 第二層 depthwise spatial conv
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(temporal_filters, spatial_filters, kernel_size=(2, 1),
                      groups=temporal_filters, bias=False),
            nn.BatchNorm2d(spatial_filters, eps=1e-5, momentum=bn_momentum),
            act_layer,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=dropout_rate)
        )

        # 第三層 separable conv
        self.separableConv = nn.Sequential(
            nn.Conv2d(spatial_filters, spatial_filters, kernel_size=(1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(spatial_filters, eps=1e-5, momentum=bn_momentum),
            act_layer,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p=dropout_rate)
        )

        # 輸出分類層
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(736, 2)
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.classify(x)
        return x





# (Optional) implement DeepConvNet model
"""
class DeepConvNet(nn.Module):
    def __init__(self):
        super(DeepConvNet, self).__init__()
        pass

    def forward(self, x):
        pass
"""

class DeepConvNet(nn.Module):
    def __init__(self, num_classes: int = 2, C: int = 2, T: int = 750,
                 activation='elu', elu_alpha=1.0, dropout_rate=0.5, bn_momentum=0.1):
        super().__init__()

        # Activation function
        if activation.lower() == 'elu':
            act_layer = nn.ELU(alpha=elu_alpha, inplace=True)
        elif activation.lower() == 'leakyrelu':
            act_layer = nn.LeakyReLU(0.1, inplace=True)
        elif activation.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif activation.lower() == 'gelu':
            act_layer = nn.GELU()
        elif activation.lower() == 'mish':
            act_layer = nn.Mish()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), bias=False),
            nn.Conv2d(25, 25, kernel_size=(C, 1), bias=False),
            nn.BatchNorm2d(25, eps=1e-5, momentum=bn_momentum),
            act_layer,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout_rate),
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), bias=False),
            nn.BatchNorm2d(50, eps=1e-5, momentum=bn_momentum),
            act_layer,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout_rate),
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), bias=False),
            nn.BatchNorm2d(100, eps=1e-5, momentum=bn_momentum),
            act_layer,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout_rate),
        )

        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), bias=False),
            nn.BatchNorm2d(200, eps=1e-5, momentum=bn_momentum),
            act_layer,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout_rate),
        )

        # Feature size 推導 (C=2, T=750)
        in_features = 200 * 1 * 43

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, num_classes, bias=True)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x