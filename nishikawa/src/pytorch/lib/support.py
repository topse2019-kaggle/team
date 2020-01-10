### ライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt
import torchvision


### 画像の表示
def show_image(image):
    """
    画像を表示します。
    
    Parameters
    ----------
    image : array
        変換対象の画像データ(3次元配列)

    Returns
    -------
    (画像の表示)
    
    """
    image = image / 2 + 0.5
    npimg = image.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

### 複数の画像を並べ表示する。
def show_images(iterator, img_num):
    """
    指定した数の画像をイテレータからデータを取得して表示します。
    
    Parameters
    ----------
    iterator : (images, label)を持つイテレータ
        画像データを持つイテレータ
    img_num : int
        表示する画像の数

    Returns
    -------
    (画像の表示)
    
    """
    images, labels = iterator.next()
    #画像を表示
    show_image(torchvision.utils.make_grid(images[:img_num]))
    # ラベルを表示
    print(' '.join('%5s' % class_names[labels[j]] for j in range(img_num)))