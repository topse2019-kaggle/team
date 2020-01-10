### 各種ライブラリをインポート
import torch
import torchvision
import torchvision.transforms as transforms

### 参考サイト
"""
Pytorch公式サイト
https://pytorch.org/docs/stable/torchvision/datasets.html
"""

### 学習用のtransform
def train_transform(resize=224, flip_available=False, normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5)):
    """
    学習用のtransform
    
    Parameters
    ----------
    flip_available : boolean
        flip可能(True) or flip不可(False)
    normalize_mean : sequence
        各チャネルの平均値
    normalize_std : sequence
        各チャネルの標準偏差

    Returns
    -------
    学習用のtransform
    """
    return transform(resize=resize, flip_available=flip_available, train=True, normalize_mean=normalize_mean, normalize_std=normalize_std)

### 評価用のtransform
def val_transform(resize=224, flip_available=False, normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5)):
    """
    評価用のtransform
    
    Parameters
    ----------
    resize : int
        画像サイズ
    flip_available : boolean
        flip可能(True) or flip不可(False)
    normalize_mean : sequence
        各チャネルの平均値
    normalize_std : sequence
        各チャネルの標準偏差

    Returns
    -------
    評価用のtransform
    """
    return transform(resize=resize, flip_available=flip_available, train=False, normalize_mean=normalize_mean, normalize_std=normalize_std)

### DataSet用のtransform
def transform(resize=224, flip_available=False, train=True, normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5)):
    """
    DataSet用のtransform
    
    Parameters
    ----------
    resize : int
        画像サイズ
    flip_available : boolean
        flip可能(True) or flip不可(False)
    train : boolean
        学習用(True) or 評価用(False)
    normalize_mean : sequence
        各チャネルの平均値
    normalize_std : sequence
        各チャネルの標準偏差

    Returns
    -------
    DataSet用のtransform
    """
    transform_list = []
    
    # 学習用
    if(train):
        # 回転可能
        if(flip_available):
            transform_list.append(transforms.RandomHorizontalFlip())
            transform_list.append(transforms.RandomVerticalFlip())

    # 共通
    transform_list.append(transforms.Resize((resize, resize)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(normalize_mean, normalize_std))
    
    # 学習用
    if(train):
        transform_list.append(transforms.RandomErasing(value='random'))
    
    # transformの定義一覧
    return transforms.Compose(transform_list)


### データローダ
def DataLoader(train_dataset, test_dataset, batch_size=10, suffle=False, num_workers=2):
    """
    データローダ
    
    Parameters
    ----------
    train_dataset : DataSet
        学習用DataSet
    test_dataset : DataSet
        評価用DataSet

    Returns
    -------
    DataLoaderのセット(train, val)
    """
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=suffle, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=suffle, num_workers=num_workers)
    return { "train" : train_loader, "val" : test_loader }

