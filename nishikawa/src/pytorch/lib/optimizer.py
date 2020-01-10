### ライブラリのインポート
import torch
import torch.optim as optim
import torch.nn as nn

### ソフトマックス交差エントロピ誤差
def CrossEntropyLoss():
    """
    ソフトマックス交差エントロピ誤差関数
    
    Returns
    -------
    ソフトマックス交差エントロピ誤差関数
    """
    return nn.CrossEntropyLoss()


### 確率的勾配降下法(SGD)
def SGD(net, lr=0.001, momentum=0.9):
    """
    確率的勾配降下法(SGD)
    
    Parameters
    ----------
    net : nn.Module
        学習対象のネットワーク
    lr : float
        学習率(learning rate)
    momentum : float
        momentum factor

    Returns
    -------
    確率的勾配降下法(SGD)
    """
    return optim.SGD(net.parameters(), lr=lr, momentum=momentum)

### Adam
def Adam(net, lr=0.001):
    """
    勾配降下法(Adam)
    
    Parameters
    ----------
    net : nn.Module
        学習対象のネットワーク
    lr : float
        学習率(learning rate)

    Returns
    -------
    Adaptive Moment Estimation
    """
    return optim.Adam(net.parameters(), lr=lr)
