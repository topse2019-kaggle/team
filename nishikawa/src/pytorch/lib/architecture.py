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
    def train(self, data_loader, criterion=None, optimizer=None, epoch_count=10, is_inception=False):
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
            
        Returns
        -------
        net : nn.Module
            学習済みのネットワーク    
        """
        # default値をセット
        if(criterion is None):
            criterion = optim.CrossEntropyLoss()
        if(optimizer is None):
            optimizer = optim.Adam(self.net)
        
        self.net = trainer.train(self.net, data_loader, criterion, optimizer, epoch_count, device=self.device, multiGPU=False, is_inception=is_inception)
    
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