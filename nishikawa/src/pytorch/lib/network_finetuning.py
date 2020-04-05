import torch
import torchvision
from torchvision import models
import torch.nn as nn

### VGG16
def VGG16(class_size):
    """
    VGG16のネットワーク
    
    Parameters
    ----------
    class_size : int
        分類するクラスの数

    Returns
    -------
    AlexNetのネットワーク
    """
    use_pretrained = False
    net = models.vgg16(pretrained=use_pretrained)

    #分類するクラス数を指定する
    net.classifier[6] = nn.Linear(in_features=4096, out_features=class_size)

    #print(net)

    for name, param in net.named_parameters():
        param.requires_grad = True

    return net

### ResNet 50
def ResNet50(class_size):
    """
    ResNet50のネットワーク
    
    Parameters
    ----------
    class_size : int
        分類するクラスの数

    Returns
    -------
    ResNet50のネットワーク
    """
    use_pretrained = False
    net = models.resnet50(pretrained=use_pretrained)

    #分類するクラス数を指定する
    net.fc = nn.Linear(in_features=2048, out_features=class_size)

    #print(net)

    for name, param in net.named_parameters():
        param.requires_grad = True

    return net

### ResNet 152
def ResNet152(class_size):
    """
    ResNet50のネットワーク
    
    Parameters
    ----------
    class_size : int
        分類するクラスの数

    Returns
    -------
    ResNet50のネットワーク
    """
    use_pretrained = False
    net = models.resnet152(pretrained=use_pretrained)

    #分類するクラス数を指定する
    net.fc = nn.Linear(in_features=2048, out_features=class_size)

    print(net)

    for name, param in net.named_parameters():
        param.requires_grad = True

    return net

#### Inception v3
def InceptionV3(class_size):
    """
    InceptionV3のネットワーク
    
    Parameters
    ----------
    class_size : int
        分類するクラスの数

    Returns
    -------
    InceptionV3のネットワーク
    """
    
    """ Inception v3
      Be careful, expects (299,299) sized images and has auxiliary output
    """
    use_pretrained = False
    net = models.inception_v3(pretrained=use_pretrained)

    #分類するクラス数を指定する
    net.AuxLogits.fc = nn.Linear(768, class_size)
    net.fc = nn.Linear(2048, class_size)

    #print(net)

    for name, param in net.named_parameters():
        param.requires_grad = True

    return net