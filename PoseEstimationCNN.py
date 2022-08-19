from torch import nn
import torch.nn.functional as F


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
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    padding_mode='replicate',
                )
            )
            conv_list.append(nn.BatchNorm2d(channels[i + 1]))
            conv_list.append(nn.ReLU())
        # conv_list.append(nn.Flatten())
        self.conv = nn.Sequential(*conv_list)

    def forward(self, x):
        x = self.conv(x)
        # print(x.shape)
        output = F.adaptive_avg_pool2d(x, (1, 1))
        output = output.flatten()
        # print(output.shape)
        return output
