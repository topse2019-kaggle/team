### ライブラリのインポート
import torch
import torch.optim
import datetime
from visdom import Visdom
import numpy as np

### 学習
def train(net, data_loader, criterion, optimizer, epoch_count=10, device="cpu", multiGPU=False):
    """
    学習済みモデルに対するテストデータを使用した精度の評価
    
    Parameters
    ----------
    net : nn.Module
        学習対象のネットワーク
    data_loader : Dict
        DataLoaderのセット(train, val)
    criterion : Loss functions
        誤差関数
    optimizer : torch.optim
        最適化関数
    epoch_count : int, default 10
        学習回数
    device : string default cpu
        学習デバイス(cpu / cuda)
        
    Returns
    -------
    net : nn.Module
        学習済みのネットワーク    
    """
    
    # デバイスの設定
    print(device)
    net = net.to(device)
    if(multiGPU):
        print("multiGPU")
        net = torch.nn.DataParallel(net)
    
    # グラフモジュールの初期化
    viz = Visdom()
    
    # 指定した回数分学習を行う
    for epoch in range(epoch_count):
        print('----------')
        print('Epoch {}/{} {}'.format(epoch+1, epoch_count, datetime.datetime.now()))

        # 学習と評価を交互に実行する
        for phase in ['train', 'val']:
            # モードの切り替え
            if phase == 'train':
                net.train()
            else:
                net.eval()

            # 学習結果の初期化
            epoch_loss = 0.0
            epoch_corrects = 0

            if (epoch == 0) and (phase == 'train'):
                continue

            # DataLoaderで取得したデータ分繰り返し学習する
            for i, data in enumerate(data_loader[phase], 0):

                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 最適化関数の設定
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # 学習モード時の処理
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    # 学習結果の更新
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            # 学習結果の更新
            epoch_loss = epoch_loss / len(data_loader[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(data_loader[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            if phase == 'train':
                viz.line(X=np.array([epoch]), Y=np.array([epoch_loss]), win='train_test', name='train_loss', update='append')
                viz.line(X=np.array([epoch]), Y=np.array([epoch_acc.item()]), win='train_test', name='train_acc', update='append')
            else :
                viz.line(X=np.array([epoch]), Y=np.array([epoch_loss]), win='train_test', name='test_loss', update='append')
                viz.line(X=np.array([epoch]), Y=np.array([epoch_acc.item()]), win='train_test', name='test_acc', update='append')
        
    print('Finished Training')
    return net


### テストデータを使用した評価
def predict(model, test_loader, device="cpu"):
    """
    学習済みモデルに対するテストデータを使用した精度の評価
    
    Parameters
    ----------
    model : nn.Module
        学習済みモデル
    test_loader : DataLoader
        テストデータを含むDataLoader

    Returns
    -------
    (モデルの精度)
    
    """
    #検証
    count_when_correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            test_data, teacher_labels = data
            test_data = test_data.to(device)
            teacher_labels = teacher_labels.to(device)
            
            results = model(test_data)
            _, predicted = torch.max(results.data, 1)
            total += teacher_labels.size(0)
            count_when_correct += (predicted == teacher_labels).sum().item()

    print('検証画像に対しての正解率： %d %%' % (100 * count_when_correct / total))