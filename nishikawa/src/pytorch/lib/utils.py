### ライブラリのインポート
from lib import network_finetuning as network
from lib import architecture
from lib import dataloader as dl
from lib import utils_transform as utils_t
from lib import optimizer
from lib import trainer

import torchvision

import torch
import numpy as np
import datetime

### random_seed一覧
random_seed_list = [
    0,1,2,3,4,5,6,7,8,9
]

### transfrom一覧(対象Data Autmentationのみ)
transform_list_include = [
    utils_t.trainsform_none(),
    utils_t.trainsform_horizontal_flip(),
    utils_t.trainsform_vertical_flip(),
    utils_t.trainsform_rotation(),
    utils_t.trainsform_perspective(),
    utils_t.trainsform_crop(),
    utils_t.trainsform_erasing()
]
### transform一覧(対象Data Augmentationを除く)
transform_list_exclude = [
    utils_t.trainsform_all(),
    utils_t.exclude_horizontal_flip(),
    utils_t.exclude_vertical_flip(),
    utils_t.exclude_rotation(),
    utils_t.exclude_perspective(),
    utils_t.exclude_random_crop(),
    utils_t.exclude_random_erasing()
]

### 
def evaluation_start(class_size, batch_size, train_path, val_path, test_path, random_seed_list=random_seed_list, net=None, transform_list=transform_list_include, multiGPU=False, epoch_count=10):
    """
    パターン評価の開始

    Parameters
    ----------
    class_size : int
        クラス数
    batch_size : int
        ミニバッチサイズ
    train_path : string
        学習対象のデータ・セットへのパス
    val_path : string
        評価対象のデータ・セットへのパス
    test_path : string, default False
        テスト対象のデータ・セットへのパス
    random_seed_list : list(int), default transform_list_include
        乱数シードの一覧
    net :net : nn.Module, defalt None
        CNNのネットワーク(デフォルト：ResNet50)
    transform_list : list(transoform), default transform_list_include
        実行対象のData Augmentationの一覧
    epoch_count : int, default 10
        実行回数
    """
    
    # ネットワークをバックアップ(繰り返し使うため)
    net_bk = net
    
    print("処理を開始。{}".format(datetime.datetime.now()))
    
    ### 指定したシードの数分実行
    for seed in random_seed_list:
        print("乱数シード={}".format(seed))
        ### 指定したtransformの数分実行
        
        # ランダムシードを固定
        set_random_seed(random_seed=seed)
        
        for transform in transform_list:
            print("transform={}".format(transform))
            ### 各種初期化

            # アーキテクチャを初期化
            if net_bk == None:
                net = network.ResNet50(class_size=class_size)
            else :
                net = net_bk
        
            # DataLoaderを初期化
            train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
            val_dataset = torchvision.datasets.ImageFolder(root=val_path, transform=utils_t.trainsform_none())
            data_loader = dl.DataLoader(train_dataset, val_dataset, batch_size=batch_size, shuffle=True)
            
            ### アーキテクチャの初期化
            model = architecture.CNN_Architecture(net)
            
            ### 学習開始
            model.train(data_loader, epoch_count=epoch_count, multiGPU=multiGPU)
            
            ### テスト
            test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=utils_t.trainsform_none()) 
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
            model.predict(test_loader)
            
    print("処理を終了しました。{}".format(datetime.datetime.now()))

### ランダムシードの固定            
def set_random_seed(random_seed=0):
    """
    ランダムシードの固定

    Parameters
    ----------
    random_seed : int, default 0
        乱数シード
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
