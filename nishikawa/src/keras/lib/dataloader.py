### ライブラリのインポート
from keras.preprocessing.image import ImageDataGenerator
from . import dataloader_support as support
import numpy as np
from keras.utils.np_utils import to_categorical

### 学習用のgenerator
def train_generator(root, num_classes, resize=224):
    """
    学習データ用のデータとジェネレータを取得する。
    
    Parameters
    ----------
    root : string
        学習対象となる画像が格納されているディレクトリのroot
    num_classes : int
        分類する数
    resize : int
        学習対象の画像のサイズ

    Returns
    -------
    (画像データ, ラベルデータ, データジェネレータ)
    """
    return data_loader(root=root, num_classes=num_classes, data_type="train", resize=resize)

### 評価用のgenerator
def test_generator(root, num_classes, resize=224):
    """
    評価データ用のデータとジェネレータを取得する。
    
    Parameters
    ----------
    root : string
        評価対象となる画像が格納されているディレクトリのroot
    num_classes : int
        分類する数
    resize : int
        評価対象の画像のサイズ

    Returns
    -------
    (画像データ, ラベルデータ, データジェネレータ)
    """
    return data_loader(root=root, num_classes=num_classes, data_type="val", resize=resize)

### DataLoader
def data_loader(root, num_classes, data_type="train", resize=224):
    """
    Parameters
    ----------
    root : string
        学習対象となる画像が格納されているディレクトリのroot
    num_classes : int
        分類する数
    data_type : string, default train
        train or val
    resize : int
        学習対象の画像のサイズ

    Returns
    -------
    (画像データ, ラベルデータ, データジェネレータ)
    """
    # ラベル名を取得
    classes, class_to_idx = support._find_classes(root)
    # イメージファイルのパスを取得
    images = support.make_dataset(root, class_to_idx)
    # generatorの設定
    if data_type == "train":
        generator = data_generator(normalization=False)
    else :
        generator = data_generator(normalization=False)
    
    labels = []
    datas = []
    for image, clazz in images:
        labels.append(clazz)
        datas.append(np.asarray(support.pil_loader(image).resize((resize,resize))))
    return np.array(datas).reshape(-1,resize,resize,3), to_categorical(y=np.array(labels), num_classes=num_classes), generator

### DataSet用のtransform(DataGenerator)
def data_generator(normalization=True, flip_available=False):
    """
    Parameters
    ----------
    normalization : boolean, default True
        正規化の実施有無
    flip_available : boolean, default False
        回転可否

    Returns
    -------
    データジェネレータ
    """
    ### TODO(resize, random erasing)
    return ImageDataGenerator(
        featurewise_std_normalization=True,
        horizontal_flip=False,
        vertical_flip=False)