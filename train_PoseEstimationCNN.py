import os
from typing import Tuple
import wandb
import math
import numpy as np
import cv2

import torch

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import sys
sys.path.append('.')
from PoseEstimationDataset import PoseEstimationDataset
from Trainer import Tranier
from PoseEstimationCNN import PoseEstimationCNN


class PoseEstimationCNNTrainer(Tranier):
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
        self.valid_targets = []
        self.valid_outputs = []
        self.fig_pose_estimation = plt.figure(figsize=(20, 10))

        train_dataset = PoseEstimationDataset(
            data_path,
        )
        valid_dataset = PoseEstimationDataset(
            data_path,
        )
        torch.manual_seed(1)
        indices = torch.randperm(len(train_dataset)).tolist()
        N = int(len(valid_dataset) * 0.2)
        train_dataset = torch.utils.data.Subset(train_dataset, indices[:-N])
        valid_dataset = torch.utils.data.Subset(valid_dataset, indices[-N:])

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

        model = PoseEstimationCNN()
        # self.loss_fn = torch.nn.MSELoss()
        self.loss_fn = torch.nn.SmoothL1Loss()

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
                project='PoseEstimation',
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

        images = torch.stack(images).to(self.device)
        targets = torch.stack(targets).to(self.device)

        # if not valid:
        #     images += torch.randn_like(images) * 0.1

        pred = self.model(images)
        loss = self.loss_fn(pred, targets)

        # if valid:
        self.valid_images.extend(images.cpu().detach().numpy())
        self.valid_targets.extend(targets.cpu().detach().numpy())
        self.valid_outputs.extend(pred.cpu().detach().numpy())

        return loss

    def plot_results(self, epoch: int):
        if epoch % 100 == 0 or (epoch % 1 == 0 and epoch <= 100):
            self.plot_pose_estimation(
                self.fig_pose_estimation,
                self.valid_images,
                self.valid_targets,
                self.valid_outputs,
                epoch=epoch,
            )
            self.fig_pose_estimation.savefig(
                os.path.join(self.out_dir, 'pose_estimation.png'),
            )

            if self.wandb_flag:
                wandb.log({
                    'epoch': epoch,
                    'pose_estimation': wandb.Image(self.fig_pose_estimation),
                })

        self.valid_images = []
        self.valid_targets = []
        self.valid_outputs = []

    def plot_pose_estimation(self, fig, images, targets, outputs, epoch=0):
        fig.clf()
        # row, col = 5, 10
        # row, col = 1, 5
        # col = 5
        col = 10
        row = math.ceil(len(images) / col)
        for i in range(len(images)):
            ax = fig.add_subplot(row, col, i + 1)
            image = images[i].transpose(1, 2, 0)
            image = (255 * image).astype(np.uint8)
            image = cv2.UMat(image)
            image = image.get()
            ax.imshow(image)
            ax.axis('off')

            H, W, _ = image.shape

            target_angle = targets[i]
            target_dx = H / 4 * np.cos(target_angle)
            target_dy = H / 4 * np.sin(target_angle)
            ax.arrow(
                x=W / 2,
                y=H / 2,
                dx=target_dx,
                dy=target_dy,
                width=0.1,
                head_width=0.05,
                head_length=0.2,
                length_includes_head=True,
                color='red',
            )

            output_angle = outputs[i]
            output_dx = H / 4 * np.cos(output_angle)
            output_dy = H / 4 * np.sin(output_angle)
            ax.arrow(
                x=W / 2,
                y=H / 2,
                dx=output_dx,
                dy=output_dy,
                width=0.1,
                head_width=0.05,
                head_length=0.2,
                length_includes_head=True,
                color='blue',
            )

        fig.suptitle('{} epoch'.format(epoch))
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

    def train(self, n_epochs: int):
        return super().train(
            n_epochs,
            callback=self.plot_results,
        )


def main(args):
    PoseEstimationCNNTrainer(
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

    def tp(x):
        return list(map(int, x.split(',')))

    parser.add_argument('--gpu', type=tp, default='0')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argparse()
    main(args)
