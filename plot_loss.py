import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def main(args):
    print(args.path)

    df = pd.read_json(os.path.join(args.path, 'None.log.json'), lines=True)
    print(df)

    plt.figure(figsize=(12, 8))
    plt.plot(df['epoch'], df['loss'], label='loss')
    plt.plot(df['epoch'], df['loss_cls'], label='loss_cls')
    plt.plot(df['epoch'], df['loss_bbox'], label='loss_bbox')
    y_max = np.mean(df['loss']) + 2 * np.std(df['loss'])
    y_min = min(min(df['loss_cls']), min(df['loss_bbox']))
    plt.ylim(0.9 * y_min, 1.1 * y_max)
    plt.yscale('log')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(args.path, 'loss.png'))


def argparse():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default='./results/MMRotate_test/')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argparse()
    main(args)
