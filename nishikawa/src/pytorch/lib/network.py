### ライブラリのインポート
import torch
import torch.nn as nn


### AlexNet
def AlexNet(in_channels=3, out_channels=4096):
    """
    AlexNetのネットワーク
    
    Parameters
    ----------
    in_channels : int, default 3
        インプット画像のチャネル数
    out_channels : int, default 4096
        分類するクラスの数

    Returns
    -------
    AlexNetのネットワーク
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=55, kernel_size=11, stride=4, padding=0),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        nn.Conv2d(in_channels=55, out_channels=256, kernel_size=5, stride=1, padding=2),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        nn.Dropout2d(),
        nn.AdaptiveAvgPool2d((6, 6)),
        nn.Flatten(),
        nn.Linear(in_features=6*6*256, out_features=4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(in_features=4096, out_features=4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(in_features=4096, out_features=out_channels),
        nn.Softmax(dim=0)
    )

### VGG16
def VGG16(in_channels=3, out_channels=1000):
    """
    VGG16のネットワーク
    
    Parameters
    ----------
    in_channels : int, default 3
        インプット画像のチャネル数
    out_channels : int, default 1000
        分類するクラスの数

    Returns
    -------
    VGG16のネットワーク
    """
    return nn.Sequential(
        # Block1
        nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        # Block2
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        # Block3
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        # Block4
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        # Block5
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        nn.Dropout(p=0.5),
        # fc
        nn.AdaptiveAvgPool2d((7, 7)),
        nn.Flatten(),
        nn.Linear(in_features=7*7*512, out_features=4096),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=4096),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=out_channels),
        nn.Softmax(dim=0)
    )

### VGG19
def VGG19(in_channels=3, out_channels=1000):
    """
    VGG16のネットワーク
    
    Parameters
    ----------
    in_channels : int, default 3
        インプット画像のチャネル数
    out_channels : int, default 1000
        分類するクラスの数

    Returns
    -------
    VGG16のネットワーク
    """
    return nn.Sequential(
        # Block1
        nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        # Block2
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        # Block3
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
        nn.ReLU(True),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        # Block4
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
        nn.ReLU(True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        # Block5
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
        nn.ReLU(True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        nn.Dropout(p=0.5),
        # fc
        nn.AdaptiveAvgPool2d((7, 7)),
        nn.Flatten(),
        nn.Linear(in_features=7*7*512, out_features=4096),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=4096),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=out_channels),
        nn.Softmax(dim=0)
    )
