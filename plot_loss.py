import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def main(args):
    print(args.data)

    df = pd.read_json(args.data, lines=True)
    print(df)

    plt.figure(figsize=(12, 8))
    plt.plot(df['epoch'], df['loss'], label='loss')
    plt.plot(df['epoch'], df['loss_cls'], label='loss_cls')
    plt.plot(df['epoch'], df['loss_bbox'], label='loss_bbox')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(args.output)


def argparse():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data', type=str,
                        default='./results/MMRotate_test/None.log.json')
    parser.add_argument('--output', type=str,
                        default='./results/MMRotate_test/loss.png')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argparse()
    main(args)