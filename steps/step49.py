if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import math

import matplotlib.pyplot as plt
import numpy as np

import dezero
import dezero.functions as F
from dezero import optimizers
from dezero.models import MLP


def main():
    # ハイパーパラメータ
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    lr = 1.0

    # データの準備 / モデル・オプティマイザーの生成
    train_set = dezero.datasets.Spiral(train=True)
    model = MLP((hidden_size, 3))
    optimizer = optimizers.SGD(lr).setup(model)

    data_size = len(train_set)
    max_iter = math.ceil(data_size / batch_size)

    losses = []
    for epoch in range(max_epoch):
        # データセットのシャッフル
        index = np.random.permutation(data_size)
        sum_loss = 0

        for i in range(max_iter):
            # ミニバッチの取り出し
            batch_index = index[i * batch_size:(i + 1) * batch_size]
            batch = [train_set[i] for i in batch_index]
            batch_xy = np.array([example[0] for example in batch])
            batch_t = np.array([example[1] for example in batch])

            # 勾配の計算 / パラメータの更新
            y = model(batch_xy)
            loss = F.softmax_cross_entropy(y, batch_t)
            model.cleargrads()
            loss.backward()
            optimizer.update()

            sum_loss += float(loss.data) * len(batch_t)

        # 学習経過を表示
        avg_loss = sum_loss / data_size
        print(f'epoch {epoch + 1}: loss {avg_loss:.2}')
        losses.append(avg_loss)

    # 損失関数の学習経過を描画
    e = np.arange(max_epoch)
    plt.plot(e, losses)
    plt.savefig('loss.jpg')
    plt.clf()

    # 学習結果を描画
    h = 0.001
    x_min, y_min = train_set[0][0]
    x_max, y_max = train_set[0][0]
    for (x, y), _ in train_set:
        x_min = min(x, x_min)
        x_max = max(x, x_max)
        y_min = min(y, y_min)
        y_max = max(y, y_max)
    x_min, x_max = x_min - .1, x_max + .1
    y_min, y_max = y_min - .1, y_max + .1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h))
    X = np.c_[xx.ravel(), yy.ravel()]

    with dezero.no_grad():
        score = model(X)
    predict_cls = np.argmax(score.data, axis=1)
    Z = predict_cls.reshape(xx.shape)
    plt.contourf(xx, yy, Z)

    # データを描画
    markers = ['o', 'x', '^']
    colors = ['orange', 'blue', 'green']
    for (x, y), c in train_set:
        plt.scatter(x, y, s=40, marker=markers[c], c=colors[c])
    plt.savefig('result.jpg')


if __name__ == '__main__':
    main()
