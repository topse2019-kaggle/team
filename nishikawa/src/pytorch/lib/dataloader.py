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

### simple-transform
def simple_transform(resize=224):
    """
    Simpleなtransform
    
    Parameters
    ----------
    resize : int
        画像サイズ

    Returns
    -------
    DataSet用のtransform
    """
    return transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=False)
    ])

### パターン検証用のtransform
def pattern_transform(resize=224, HorizontalFlip=True, VerticalFlip=True, Rotation=True, Perspective=True, Crop=True, Erasing=True):
    """
    DataSet用のtransform
    
    Parameters
    ----------
    resize : int
        画像サイズ
    HorizontalFlip : boolean, default True
        RandomHorizontalFlipの実行可否
    VerticalFlip : boolean, default True
        RandomVerticalFlipの実行可否
    Rotation : boolean, default True
        RandomRotationの実行可否
    Perspective : boolean, default True
        RandomPerspectiveの実行可否
    Crop : boolean, default True
        RandomResizedCropの実行可否
    Erasing : boolean, default True
        RandomErasingの実行可否

    Returns
    -------
    DataSet用のtransform
    """
    transform_list = []
    
    # Resize
    transform_list.append(transforms.Resize((resize, resize)))
    
    ### 回転系
    # HorizontalFlip
    if HorizontalFlip:
        transform_list.append(torchvision.transforms.RandomHorizontalFlip(p=0.5))
    # VerticalFlip
    if VerticalFlip:
        transform_list.append(torchvision.transforms.RandomVerticalFlip(p=0.5))
    # Rotation (degrees=180)
    if Rotation:
        transform_list.append(torchvision.transforms.RandomRotation(degrees=180, resample=False, expand=False, center=None))

    ### 遠近
    # Perspective
    if Perspective:
        transform_list.append(torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3))
    
    ### Crop系
    # ResizedCrop
    if Crop:
        transform_list.append(torchvision.transforms.RandomResizedCrop(size=resize, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2))
    
    # ToTensor
    transform_list.append(transforms.ToTensor())
    # Erasing (value=random)
    if Erasing:
        transform_list.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random', inplace=False))
    # Normalize (mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=False))
    
    #print(transform_list)
    # transformの定義一覧
    return transforms.Compose(transform_list)

### データローダ
def DataLoader(train_dataset, test_dataset, batch_size=10, shuffle=False, num_workers=2):
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return { "train" : train_loader, "val" : test_loader }