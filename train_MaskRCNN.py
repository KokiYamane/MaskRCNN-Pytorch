import os
from typing import Tuple
import wandb

import torch
import torchvision.transforms as T

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import sys
sys.path.append('.')
from SegmentationDataset import SegmentationDataset
from Trainer import Tranier
from plot_results import plot_segmentation_masks
from data_augmentation import AddGaussianNoise
from MaskRCNNModel import get_model_instance_segmentation


class MaskRCNNTrainer(Tranier):
    def __init__(
        self,
        data_path: str,
        out_dir: str,
        batch_size: int,
        learning_rate: float,
        wandb_flag: bool,
        gpu: list = [0],
        early_stopping_count: int = 1e10,
        num_workers: int = 1,
    ):
        self.out_dir = out_dir

        # plot results
        self.valid_images = []
        self.valid_outputs = []
        self.fig_segment_masks = plt.figure(figsize=(20, 20))

        transforms = T.Compose([
            T.ToTensor(),
            AddGaussianNoise(std=0.1),
            T.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
            ),
            T.RandomErasing(),
            # GridMask(),
        ])

        dataset = SegmentationDataset(
            data_path,
            transforms=transforms,
        )
        # valid_dataset = SegmentationDataset(
        #     data_path,
        #     get_transform(train=False),
        # )
        torch.manual_seed(1)
        indices = torch.randperm(len(dataset)).tolist()
        N = int(len(dataset) * 0.1)
        train_dataset = torch.utils.data.Subset(dataset, indices[:-N])
        valid_dataset = torch.utils.data.Subset(dataset, indices[-N:])

        def collate_fn(batch):
            return tuple(zip(*batch))

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            # drop_last=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            # drop_last=True,
        )

        print('train data num:', len(train_dataset))
        print('valid data num:', len(valid_dataset))

        num_classes = dataset.num_classes
        model = get_model_instance_segmentation(num_classes=num_classes)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=0.005,
            momentum=0.9,
            weight_decay=0.0005,
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.1,
        )

        super().__init__(
            train_loader=train_loader,
            valid_loader=valid_loader,
            model=model,
            calc_loss=self.calc_loss,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            out_dir=out_dir,
            wandb_flag=wandb_flag,
            gpu=gpu,
            early_stopping_count=early_stopping_count,
        )

        if wandb_flag:
            wandb.init(
                project='MaskRCNN',
                name=os.path.basename(out_dir),
            )
            config = wandb.config
            config.data_path = data_path
            config.batch_size = batch_size
            config.learning_rate = learning_rate
            config.train_data_num = len(train_dataset)
            config.valid_data_num = len(valid_dataset)
            wandb.watch(model)

    def calc_loss(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        valid: bool = False,
    ) -> torch.Tensor:
        images, targets = batch

        images = [image.to(self.device) for image in images]
        targets = [
            {k: v.to(self.device) for k, v in target.items()}
            for target in targets
        ]
        # print(torch.sum(targets[0]['masks']))

        if not valid:
            losses = self.model(images, targets)
            total_loss = sum(losses.values())
            return total_loss
        else:
            outputs = self.model(images)

            images_np = [image.to('cpu').detach().numpy() for image in images]
            output_np = [
                {k: v.to('cpu').detach().numpy() for k, v in output.items()}
                for output in outputs
            ]
            self.valid_images.extend(images_np)
            self.valid_outputs.extend(output_np)

            return torch.zeros(1)

    def plot_results(self, epoch: int):
        if epoch % 100 == 0 or (epoch % 1 == 0 and epoch <= 100):
            plot_segmentation_masks(
                self.fig_segment_masks,
                self.valid_images,
                self.valid_outputs,
                epoch=epoch,
            )
            self.fig_segment_masks.savefig(
                os.path.join(self.out_dir, 'segment_masks.png'),
            )

            if self.wandb_flag:
                wandb.log({
                    'epoch': epoch,
                    'segment_masks': wandb.Image(self.fig_segment_masks),
                })

        self.valid_images = []
        self.valid_outputs = []

    def train(self, n_epochs: int):
        return super().train(n_epochs, callback=self.plot_results)


def main(args):
    MaskRCNNTrainer(
        data_path=args.data,
        out_dir=args.output,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        wandb_flag=args.wandb,
        gpu=args.gpu,
        num_workers=args.num_workers,
    ).train(args.epoch)


def argparse():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data', type=str,
                        default='../datasets/PennFudanPed/')
    parser.add_argument('--output', type=str, default='./results/test/')
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--early_stopping', type=int, default=1e10)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--gpu', default='0',
                        type=lambda x: list(map(int, x.split(','))))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argparse()
    main(args)
