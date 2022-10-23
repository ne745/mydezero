if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dezero
import dezero.functions as F
from dezero import DataLoader
from dezero import optimizers
from dezero.models import MLP


def main():
    # ハイパーパラメータ
    max_epoch = 5
    batch_size = 100
    hidden_size = 1000

    # データの準備
    train_set = dezero.datasets.MNIST(train=True)
    test_set = dezero.datasets.MNIST(train=False)
    train_loader = DataLoader(train_set, batch_size)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    model = MLP((hidden_size, 10), activation=F.relu)
    optimizer = optimizers.SGD().setup(model)

    log = []
    for epoch in range(max_epoch):
        sum_loss, sum_acc = 0, 0

        for batch_x, batch_t in train_loader:
            # 勾配の計算 / パラメータの更新
            y = model(batch_x)
            loss = F.softmax_cross_entropy(y, batch_t)
            acc = F.accuracy(y, batch_t)
            model.cleargrads()
            loss.backward()
            optimizer.update()

            sum_loss += float(loss.data) * len(batch_t)
            sum_acc += float(acc.data) * len(batch_t)

        # 学習経過を表示
        train_avg_loss = sum_loss / len(train_set)
        train_avg_acc = sum_acc / len(train_set)
        print(f'epoch {epoch + 1}')
        print(
            f'train loss: {train_avg_loss:.4f}, '
            f'accuracy: {train_avg_acc:.4f}')

        sum_loss, sum_acc = 0, 0
        with dezero.no_grad():
            for batch_x, batch_t in test_loader:
                y = model(batch_x)
                loss = F.softmax_cross_entropy(y, batch_t)
                acc = F.accuracy(y, batch_t)
                sum_loss += float(loss.data) * len(batch_t)
                sum_acc += float(acc.data) * len(batch_t)

        test_avg_loss = sum_loss / len(test_set)
        test_avg_acc = sum_acc / len(test_set)
        print(f'test loss: {test_avg_loss:.4f}, accuracy: {test_avg_acc:.4f}')

        log.append({
            'train_loss': train_avg_loss,
            'train_acc': train_avg_acc,
            'test_loss': test_avg_loss,
            'test_acc': test_avg_acc,
        })

    # 損失関数の学習経過を描画
    df = pd.DataFrame(log)
    ep = np.arange(max_epoch)
    plt.plot(ep, df['train_loss'], label='train')
    plt.plot(ep, df['test_loss'], label='test')
    plt.legend()
    plt.savefig('loss.jpg')
    plt.clf()

    plt.plot(ep, df['train_acc'], label='train')
    plt.plot(ep, df['test_acc'], label='test')
    plt.legend()
    plt.savefig('accuracy.jpg')
    plt.clf()


if __name__ == '__main__':
    main()
