### ライブラリのインポート
import os
import sys
from PIL import Image

### 参考サイト
"""
pytorch DatasetFolder
https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
"""

### 対象とするファイルの拡張子の一覧
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

### 許可している拡張子かのチェック
def is_image_file(filename):
    """
    学習/評価データとして、許可している拡張子かのチェック
    
    Parameters
    ----------
    filename : string
        ファイル名
    extensions : List[string]
        拡張子の一覧

    Returns
    -------
    True(対象データ) / False(対象外データ)
    """
    return filename.lower().endswith(IMG_EXTENSIONS)

### ラベル名の取得
def _find_classes(dir):
    """
   ラベル名を取得する
    
    Parameters
    ----------
    dir : string
        学習/評価データが含まれるディレクトリのroot

    Returns
    -------
    ([ラベルの一覧], [(ラベル, index)])
    """
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

### 学習/評価データの取得
def make_dataset(dir, class_to_idx, extensions=IMG_EXTENSIONS):
    """
   ラベル名を取得する
    
    Parameters
    ----------
    dir : string
        学習/評価データが含まれるディレクトリのroot
    class_to_idx : Tupple
        ([ラベルの一覧], [(ラベル, index)])
    extensions : Tupple
        対象とする拡張子の一覧

    Returns
    -------
    images : (ファイルパス, ラベルのindex)
    """
    images = []
    dir = os.path.expanduser(dir)
    
    ### 画像ファイルのデータを取得
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_image_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

### 画像ファイルの読み込み
def pil_loader(path):
    """
   画像ファイルの読み込み
    
    Parameters
    ----------
    path : string
        読み込み対象のファイル名

    Returns
    -------
    画像データ
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')