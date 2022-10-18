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
    xy, t = dezero.datasets.get_spiral(train=True)
    model = MLP((hidden_size, 3))
    optimizer = optimizers.SGD(lr).setup(model)

    data_size = len(xy)
    max_iter = math.ceil(data_size / batch_size)

    losses = []
    for epoch in range(max_epoch):
        # データセットのシャッフル
        index = np.random.permutation(data_size)
        sum_loss = 0

        for i in range(max_iter):
            # ミニバッチの生成
            batch_index = index[i * batch_size:(i + 1) * batch_size]
            batch_x = xy[batch_index]
            batch_t = t[batch_index]

            # 勾配の計算 / パラメータの更新
            y = model(batch_x)
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
    x_min, x_max = xy[:, 0].min() - .1, xy[:, 0].max() + .1
    y_min, y_max = xy[:, 1].min() - .1, xy[:, 1].max() + .1
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
    for (x, y), c in zip(xy, t):
        plt.scatter(x, y, s=40, marker=markers[c], c=colors[c])
    plt.savefig('result.jpg')


if __name__ == '__main__':
    main()
