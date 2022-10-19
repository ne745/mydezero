if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import dezero


def main():
    # データの準備 / モデル・オプティマイザーの生成
    train_set = dezero.datasets.Spiral(train=True)
    print(train_set[0])
    print(len(train_set))


if __name__ == '__main__':
    main()
