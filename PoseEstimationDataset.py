import os
import glob
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np


class PoseEstimationDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize((128, 128)),
        ])
        self.label_paths = glob.glob(os.path.join(root, 'pose', '*'))

        # self.labels = []
        # for label_path in self.label_paths:
        #     label = self._load_label(label_path)
        #     label = torch.tensor(label, dtype=torch.float32)
        #     self.labels.append(label)

    def __getitem__(self, idx):
        label_path = self.label_paths[idx]
        label_filename = os.path.splitext(
            os.path.basename(label_path))[0]
        image_path = os.path.join(
            self.root, 'segmentations', f'{label_filename}.png')
        image = Image.open(image_path)
        image = self.transforms(image)
        # print(image.shape)

        label = self._load_label(label_path)
        label = torch.tensor(label, dtype=torch.float32)

        return image, label

    def __len__(self):
        return len(self.label_paths)

    def _load_label(self, label_path):
        label = np.loadtxt(label_path, dtype='object')
        # x1, y1 = label[:2].astype(np.int)
        # x2, y2 = label[6:8].astype(np.int)
        x1, y1, x2, y2 = label[:4].astype(np.int)
        # print(x1, y1, x2, y2)
        label = np.arctan2(y2 - y1, x2 - x1)
        return label


def main():
    dataset = PoseEstimationDataset(root='./data/FOOMA/FOOMA_green')
    print(len(dataset))
    print(dataset[0])
    pass


if __name__ == '__main__':
    main()
