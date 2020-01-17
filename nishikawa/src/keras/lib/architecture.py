### ライブラリのインポート
from . import trainer
from lib import dataloader

### 畳み込みネットワーク
class CNN_Architecture():
    """
    CNNのモデルに対する処理を共通化する。
    
    Attributes
    ----------
    net : models.Model
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
    
    ### コンストラクタ
    def __init__(self, net):
        """
        コンストラクタ

        Parameters
        ----------
        net : nn.Module
            CNNのネットワーク
        """
        self.net = net

    ### 学習
    def train(self, data_loader, batch_size=10, loss="categorical_crossentropy", optimizer=None, epoch_count=10):
        """
        学習済みモデルに対するテストデータを使用した精度の評価

        Parameters
        ----------
        data_loader : Dict
            DataLoaderのセット(train, val)
        batch_size : int, default 10
            ミニバッチサイズ
        loss : string, default categorical_crossentropy
            誤差関数
        optimizer : keras.optimizers, default None(Adam)
            最適化関数
        epoch_count : int, default 10
            学習回数
            
        Returns
        -------
        Historyオブジェクト  
        """
        return trainer.train(self.net, data_loader=data_loader, batch_size=batch_size, loss=loss, optimizer=optimizer, epoch_count=epoch_count)

    def predict(self, test_dataset):
        """
        Parameters
        ----------
        test_dataset : numpy
            評価対象のデータセット
        datagenerator : ImageDataGenerator
            評価対象のデータセットの変換generator

        Returns
        -------
        (モデルの精度)
        """
        return trainer.predict(self.net, test_dataset, dataloader.transform(normalization=False))