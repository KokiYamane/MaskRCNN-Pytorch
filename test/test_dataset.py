import unittest
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
import torch
import torchvision.transforms as T

import sys
sys.path.append('.')
sys.path.append('..')
from PennFudanDataset import PennFudanDataset


class TestImageDataset(unittest.TestCase):
    def test_dataset(self):
        print('\n========== test dataset ==========')
        # datafolder = '../data/test_dataset'
        datafolder = '../data/FOOMA'

        def get_transform(train):
            transforms = []
            transforms.append(T.ToTensor())
            if train:
                transforms.append(T.RandomHorizontalFlip(0.5))
            return T.Compose(transforms)

        dataset = PennFudanDataset(
            datafolder,
            get_transform(train=True),
        )

        print('data length:', len(dataset))

        def collate_fn(batch):
            return tuple(zip(*batch))

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=10,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=0,
            drop_last=True,
        )
        # for _ in range(3):
        start = time.time()
        for i, (images, targets) in enumerate(tqdm(dataloader)):
            pass
            print('#', i)
            print('image shape:', images[0].shape)
            print('box shape:', targets[0]['boxes'].shape)
            print('mask shape:', targets[0]['masks'].shape)
        end = time.time()
        print('elapsed time:', end - start)

        # plot data
        fig = plt.figure(figsize=(10, 10))
        row = len(images)
        col = max([len(target['masks']) for target in targets])
        for i, (image, target) in enumerate(zip(images, targets)):
            ax = fig.add_subplot(row, col, col * i + 1)
            image = image.numpy().transpose(1, 2, 0)
            ax.imshow(image)
            ax.axis('off')

            for j, mask in enumerate(target['masks']):
                ax = fig.add_subplot(row, col, col * i + 2 + j)
                mask = mask.numpy()
                ax.imshow(mask, cmap='gray')
                ax.axis('off')
        fig.savefig('../results/test_dataset.png')


if __name__ == "__main__":
    unittest.main()
