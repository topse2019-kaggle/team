### ライブラリのインポート
from . import optimizer as opt
from . import callback

### 学習
def train(net, data_loader, batch_size=10, loss="categorical_crossentropy", optimizer=None, epoch_count=10):
    """
    Parameters
    ----------
    net : keras.models.Model
        学習対象のネットワーク
    data_loader : Dict
        'train' : (X_train, Y_train, DataGenerator),
        'val' : (X_test, Y_test, DataGenerator)
    batch_size : int, default 10
        ミニバッチのサイズ
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
    ### 最適関数の初期化
    if optimizer == None:
        optimizer = opt.Adam()
    
    ### DataGerenratorのフィッティング
    (X_train, Y_train, train_gererator) = data_loader['train']
    (X_test, Y_test, test_gererator) = data_loader['val']
    train_gererator.fit(X_train)
    test_gererator.fit(X_test)
    
    ### DataAugumentationの実行
    train_dataloader = train_gererator.flow(x=X_train, y=Y_train, batch_size=batch_size)
    test_dataloader = test_gererator.flow(x=X_test, y=Y_test, batch_size=batch_size)
    
    ### コンパイル
    net.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    
    ### 学習
    return net.fit_generator(
        generator=train_dataloader, 
        steps_per_epoch=len(X_train)//batch_size, 
        epochs=epoch_count, 
        verbose=2, 
        callbacks=[callback.ReduceLROnPlateau(),callback.History_Visdom()],
        validation_data=test_dataloader,
        validation_steps=len(X_test)//batch_size
    )

### 評価
def predict(model, test_dataset, datagenerator, batch_size=10):
    """
    Parameters
    ----------
    model : keras.models.Model
        学習済みのモデル
    test_dataset : numpy
        評価対象のデータセット
    datagenerator : ImageDataGenerator
        評価対象のデータセットの変換generator

    Returns
    -------
    (モデルの精度)
    """
    ### データの準備
    (X_test, Y_test) = test_dataset
    test_generator = datagenerator.fit(X_test)
    test_loader = test_generator.flow(x=X_test, y=Y_test, batch_size=batch_size)
    
    ### データの評価
    predict = model.predict_generator(generator=test_loader, steps=len(X_test)//batch_size)
    
    ### 評価結果の出力
    acc_count = len(np.where((a == b) == 1))
    
    print('検証画像に対しての正解率： %d %%' % (100 * acc_count / len(X_test)))
    