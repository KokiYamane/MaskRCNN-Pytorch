from torch import nn
# import torch.nn.functional as F
import torch
import numpy as np


class PoseEstimationCNN(nn.Module):
    def __init__(self, channels=[3, 64, 128, 256, 1]):
        super().__init__()

        conv_list = []
        conv_list.append(nn.Dropout(0.01))
        for i in range(len(channels) - 1):
            conv_list.append(
                nn.Conv2d(
                    channels[i],
                    channels[i + 1],
                    # kernel_size=4,
                    # stride=2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode='replicate',
                )
            )
            conv_list.append(nn.BatchNorm2d(channels[i + 1]))
            conv_list.append(nn.ReLU())
            conv_list.append(nn.AvgPool2d(2, stride=2))
        conv_list.append(nn.Flatten())
        self.conv = nn.Sequential(*conv_list)

        H, W = 128, 128
        H_feature = H // 2**(len(channels) - 1)
        W_feature = W // 2**(len(channels) - 1)
        feature_dim = channels[-1] * H_feature * W_feature
        self.dense = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        # self.dense = nn.Linear(feature_dim, 1)

    def forward(self, x):
        x = self.conv(x)
        output = self.dense(x)
        output = torch.tanh(output)
        output = output * np.pi
        # print(x.shape)
        # output = F.adaptive_avg_pool2d(x, (1, 1))
        # output = output.flatten()
        # print(output.shape)
        output = output.squeeze()
        return output
