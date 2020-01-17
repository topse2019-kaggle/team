### ライブラリのインポート
from keras import callbacks
import visdom
import numpy as np

### 学習率の自動調整
def ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=5, mode='auto', epsilon=0.0001, cooldown=0):
    """
    学習率を自動調整する。
    評価値の改善が止まった時に学習率を減らす。
    
    Parameters
    ----------
    monitor : string, default val_acc
        監視対象
    factor : float, default 0.1
        学習の削減率
    patience : int
        学習率削減までのepoch数
    mode : string, default auto
        監視対象(増減)
    epsilon : float
        改善判定用の閾値
    cooldown : int, default 0
        学習率削減後の再開までのepoch数

    Returns
    -------
    ReduceLROnPlateau
    """
    return callbacks.ReduceLROnPlateau(monitor=monitor, factor=factor, patience=patience, verbose=1, mode=mode, epsilon=epsilon, cooldown=cooldown)

### 学習結果の自動出力
class History_Visdom(callbacks.Callback):
    
    ### コンストラクタ
    def __init__(self):
        """
        コンストラクタ
        Visdomのインスタンスを作成する。

        """
        self.viz = visdom.Visdom()
        
    ### epoch終了時の処理
    def on_epoch_end(self, epoch, logs={}):
        """
        epoch終了時に、学習率を表示する

        Parameters
        ----------
        epoch : int
            epoch
        logs : callback methods
            学習状況

        Returns
        -------
        学習率の表示
        """
        self.viz.line(X=np.array([epoch]), Y=np.array([logs.get('loss')]), win='train_test', name='train_loss', update='append')
        self.viz.line(X=np.array([epoch]), Y=np.array([logs.get('acc')]), win='train_test', name='train_acc', update='append')
        self.viz.line(X=np.array([epoch]), Y=np.array([logs.get('val_loss')]), win='train_test', name='test_loss', update='append')
        self.viz.line(X=np.array([epoch]), Y=np.array([logs.get('val_acc')]), win='train_test', name='test_acc', update='append')