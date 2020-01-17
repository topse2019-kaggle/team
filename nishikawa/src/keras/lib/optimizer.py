### ライブラリのインポート
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau

### 参考サイト
"""
https://keras.io/ja/optimizers/
"""


### 確率的勾配降下法(SGD)
def SGD(lr=0.001, momentum=0.9):
    """
    確率的勾配降下法(SGD)
    
    Parameters
    ----------
    lr : float, default 0.001
        学習率(learning rate)
    momentum : float, default 0.9
        momentum factor

    Returns
    -------
    確率的勾配降下法(SGD)
    """
    return optimizers.SGD(lr=lr, momentum=momentum)

### Adam
def Adam(lr=0.001):
    """
    勾配降下法(Adam)
    
    Parameters
    ----------
    lr : float, default 0.001
        学習率(learning rate)

    Returns
    -------
    Adaptive Moment Estimation
    """
    return optimizers.Adam(lr=lr)
