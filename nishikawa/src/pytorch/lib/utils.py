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

### Data Augmentationマッピング
data_aug_dict = {
    # none, all
    "trainsform_none" : utils_t.trainsform_none(),
    "trainsform_all" : utils_t.trainsform_all(),
    # include
    "trainsform_horizontal_flip" : utils_t.trainsform_horizontal_flip(),
    "trainsform_vertical_flip" : utils_t.trainsform_vertical_flip(),
    "trainsform_rotation" : utils_t.trainsform_rotation(),
    "trainsform_perspective" : utils_t.trainsform_perspective(),
    "trainsform_crop" : utils_t.trainsform_crop(),
    "trainsform_erasing" : utils_t.trainsform_erasing(),
    # exclude
    "exclude_horizontal_flip" : utils_t.exclude_horizontal_flip(),
    "exclude_vertical_flip" : utils_t.exclude_vertical_flip(),
    "exclude_rotation" : utils_t.exclude_rotation(),
    "exclude_perspective" : utils_t.exclude_perspective(),
    "exclude_crop" : utils_t.exclude_crop(),
    "exclude_erasing" : utils_t.exclude_erasing()
}
### 実行対象Data Augmentation一覧(include)
transform_list_include = [
    "trainsform_none",
    "trainsform_horizontal_flip",
    "trainsform_vertical_flip",
    "trainsform_rotation",
    "trainsform_perspective",
    "trainsform_crop",
    "trainsform_erasing"
]
### 実行対象Data Augmentation一覧(exclude)
transform_list_exclude = [
    "trainsform_all",
    "exclude_horizontal_flip",
    "exclude_vertical_flip",
    "exclude_rotation",
    "exclude_perspective",
    "exclude_random_crop",
    "exclude_random_erasing"
]

### 
def evaluation_start(class_size, batch_size, train_path, val_path, test_path, save_root, random_seed_list=random_seed_list, net=None, transform_list=transform_list_include, multiGPU=False, epoch_count=10):
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
    test_path : string
        テスト対象のデータ・セットへのパス
    save_path : string
        パラメータ保存先のパス(ルート)
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
        
        # テスト用のData Loaderの準備
        test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=utils_t.trainsform_none()) 
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
            
        for transform_str in transform_list:
            ### 各種初期化

            # アーキテクチャを初期化
            if net_bk == None:
                net = network.ResNet50(class_size=class_size)
            else :
                net = net_bk
        
            # DataLoaderを初期化
            transform = data_aug_dict[transform_str]
            print("transform={}".format(transform))
            train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
            val_dataset = torchvision.datasets.ImageFolder(root=val_path, transform=utils_t.trainsform_none())
            data_loader = dl.DataLoader(train_dataset, val_dataset, batch_size=batch_size, shuffle=True)
            
            ### アーキテクチャの初期化
            model = architecture.CNN_Architecture(net)
            
            ### 学習開始
            #model.train(data_loader, epoch_count=epoch_count, multiGPU=multiGPU, random_seed=seed, save_root=save_root, save_path=transform_str, test_loader=test_loader)
            model.train(data_loader, epoch_count=epoch_count, multiGPU=multiGPU, random_seed=seed, save_root=save_root, save_path=transform_str)
            
            ### テスト          
            # 学習済みパラメータのパス
            dir, loss_file, acc_file = get_save_path(save_root, transform_str, seed)
            
            device = model.device
            print('loss値最小による検証')
            trainer.__predict_load_param(param_path=loss_file, test_loader=test_loader, device=device)
            print('acc値最大による検証')
            trainer.__predict_load_param(param_path=acc_file, test_loader=test_loader, device=device)
            
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

### 学習済みパラメータが保存されているディレクトリ、ファイルのパスを取得する
def get_save_path(save_root, save_path, random_seed):
    """
    学習済みパラメータが保存されているディレクトリ、ファイルのパスを取得する

    Parameters
    ----------
    save_root : string
        学習済みパラメータが保存されているパスのルート
    save_path : string
        学習済みパラメータが保存されているパス(ルート以下)
    random_seed : int
        乱数シード
        
    Returns
    -------
    学習済みパラメータが保存されているディレクトとファイル
    """
    dir = save_root + "/seed_" + str(random_seed) + "/"
    loss_file = dir + save_path + "_loss" + ".pth"
    acc_file = dir + save_path + "_acc" + ".pth"
    return dir, loss_file, acc_file
    