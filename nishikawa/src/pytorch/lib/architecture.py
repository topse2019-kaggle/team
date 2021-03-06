### ライブラリのインポート
import torch
from . import trainer
from . import optimizer as optim


### 畳み込みネットワーク
class CNN_Architecture():
    """
    CNNのモデルに対する処理を共通化する。

    Attributes
    ----------
    net : nn.Module
        CNNのネットワーク(学習済みモデル)
    in_channels : int, default 3
        インプット画像のチャネル数
    out_channels : int, default 4096
        分類するクラスの数
    epoch_count : int, default 10
        学習回数
    data_loader : Dict
        DataLoaderのセット(train, val)
    criterion : Loss functions
        誤差関数
    optimizer : torch.optim
        最適化関数
    """
    
    # コンストラクタ
    def __init__(self, net):
        """
        コンストラクタ

        Parameters
        ----------
        net : nn.Module
            CNNのネットワーク
        """
        self.net = net
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 学習
    def train(self, data_loader, criterion=None, optimizer=None, epoch_count=10, is_inception=False, multiGPU=False, cuda="cuda:0", visdom_port=8097, random_seed=None, save_root=None, save_path=None, test_loader=None):
        """
        学習済みモデルに対するテストデータを使用した精度の評価

        Parameters
        ----------
        data_loader : Dict
            DataLoaderのセット(train, val)
        criterion : Loss functions default None
            誤差関数
        optimizer : torch.optim default None
            最適化関数
        epoch_count : int, default 10
            学習回数
        is_inception : boolean, default False
            inceptionネットワークの有無
        multiGPU : boolean, default False
            複数のGPUを使用する場合, True
        cuda : string, default cuda:0
            使用するGPUを指定する場合に使用
        visdom_port : int, defult 8097
            visdomで使用するポート番号を指定
        random_seed : int, default None
            (検証用) 乱数シードの値
        save_root : string, defualt None
            (検証用)パラメータ保存先のパス(ルート)
        save_path : string, defualt None
            (検証用)パラメータ保存先のパス(ルート以下)
        test_loader : DataLoader, default None
            (検証用)テスト用のデータローダ
            
        Returns
        -------
        net : nn.Module
            学習済みのネットワーク    
        """
        if torch.cuda.is_available() and multiGPU==False:
            self.device = cuda
        
        # default値をセット
        if(criterion is None):
            criterion = optim.CrossEntropyLoss()
        if(optimizer is None):
            optimizer = optim.Adam(self.net)
        
        self.net = trainer.train(self.net, data_loader, criterion, optimizer, epoch_count, device=self.device, multiGPU=multiGPU, is_inception=is_inception, visdom_port=visdom_port, random_seed=random_seed, save_root=save_root, save_path=save_path, test_loader=test_loader)
    
    # 評価
    def predict(self, test_loader):
        """
        学習済みモデルに対するテストデータを使用した精度の評価

        Parameters
        ----------
        test_loader : DataLoader
            テストデータを含むDataLoader

        Returns
        -------
        (モデルの精度)

        """
        trainer.predict(self.net, test_loader, self.device)
          
    # モデルの保存
    def save(self, path):
        """
        学習済みモデルの保存

        Parameters
        ----------
        path : string
            保存先のpath

        Returns
        -------
        (モデルの保存)

        """
        torch.save(self.net.state_dict(), path)