### ライブラリのインポート
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

### AlexNet
def AlexNet(input_shape=(224,224,3), out_channels=4096):
    """
    AlexNetのネットワーク
    
    Parameters
    ----------
    input_shape : タプル, default (224,224,3)
        インプット画像の情報
    out_channels : int, default 4096
        分類するクラスの数

    Returns
    -------
    AlexNetのネットワーク
    """
    return Sequential([
        Conv2D(filters=55, kernel_size=11, strides=4, padding='valid', activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(3,3), strides=2, padding='valid'),
        Conv2D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3,3), strides=2, padding='valid'),
        Conv2D(filters=384, kernel_size=3, strides=1, padding='same', activation='relu'),
        Conv2D(filters=254, kernel_size=3, strides=1, padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3,3), strides=2, padding='valid'),
        Dropout(rate=0.5),
        Flatten(),
        Dense(units=4096, activation = "relu"),
        Dense(units=4096, activation = "relu"),
        Dropout(rate=0.5),
        Dense(units=out_channels, activation = "softmax")
    ])

### VGG16
def VGG16(input_shape=(224,224,3), out_channels=4096):
    """
    VGG16のネットワーク
    
    Parameters
    ----------
    input_shape : タプル, default (224,224,3)
        インプット画像の情報
    out_channels : int, default 4096
        分類するクラスの数

    Returns
    -------
    VGG16のネットワーク
    """
    return Sequential([
        # Block1
        Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=input_shape),
        Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'),
        # Block2
        Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'),
        Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'),
        # Block3
        Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'),
        Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'),
        Conv2D(filters=256, kernel_size=1, strides=1, padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'),
        # Block4
        Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=1, strides=1, padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'),
        # Block5
        Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=1, strides=1, padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'),
        Dropout(rate=0.5),
        # fc
        Flatten(),
        Dense(units=4096, activation = "relu"),
        Dropout(rate=0.5),
        Dense(units=4096, activation = "relu"),
        Dropout(rate=0.5),
        Dense(units=out_channels, activation = "softmax")
    ])