import os
from typing import Tuple
import numpy as np
import cv2
import wandb
import math

import torch
# import torch.nn as nn

import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import sys
sys.path.append('.')
from PennFudanDataset import PennFudanDataset
from Trainer import Tranier


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
        self.fig_segment_masks = plt.figure(figsize=(20, 10))

        def get_transform(train):
            transforms = []
            transforms.append(T.ToTensor())
            # if train:
            #     transforms.append(T.RandomHorizontalFlip(0.5))
            return T.Compose(transforms)

        train_dataset = PennFudanDataset(
            data_path,
            get_transform(train=True),
        )
        valid_dataset = PennFudanDataset(
            data_path,
            get_transform(train=False),
        )
        torch.manual_seed(1)
        indices = torch.randperm(len(train_dataset)).tolist()
        N = int(len(valid_dataset) * 0.1)
        train_dataset = torch.utils.data.Subset(train_dataset, indices[:-N])
        valid_dataset = torch.utils.data.Subset(valid_dataset, indices[-N:])

        def collate_fn(batch):
            return tuple(zip(*batch))

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            # num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            # drop_last=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            # num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        print('train data num:', len(train_dataset))
        print('valid data num:', len(valid_dataset))

        model = self.get_model_instance_segmentation(num_classes=2)

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

    def get_model_instance_segmentation(self, num_classes):
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=True)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes,
        )

        return model

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
            self.plot_segmentation_masks(
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

    def plot_segmentation_masks(self, fig, images, outputs, epoch=0):
        fig.clf()
        # row, col = 5, 10
        # row, col = 1, 5
        col = 5
        row = math.ceil(len(images) / 5)
        for i, (image, output) in enumerate(zip(images, outputs)):
            ax = fig.add_subplot(row, col, i + 1)
            image = image.transpose(1, 2, 0)
            image = (255 * image).astype(np.uint8)
            image = cv2.UMat(image)
            masks = output['masks']
            scores = output['scores']
            for mask, score in zip(masks, scores):
                if score < 0.75:
                    continue

                mask = mask.transpose(1, 2, 0)
                mask = (255 * mask).astype(np.uint8)
                mask = cv2.GaussianBlur(mask, (5, 5), 0)
                ret, mask = cv2.threshold(
                    mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours, hierarchy = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours) == 0:
                    continue

                contour = max(contours, key=lambda x: cv2.contourArea(x))
                cv2.drawContours(
                    image,
                    [contour],
                    -1,
                    # color=(0, 255, 0),
                    # thickness=10,
                    color=(0, int(255 * score), 0),
                    thickness=int(10 * score),
                )
            image = image.get()
            ax.imshow(image)
            ax.axis('off')

        fig.suptitle('{} epoch'.format(epoch))
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

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

    def tp(x):
        return list(map(int, x.split(',')))

    parser.add_argument('--gpu', type=tp, default='0')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argparse()
    main(args)
