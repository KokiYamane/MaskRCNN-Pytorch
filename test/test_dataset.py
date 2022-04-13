import unittest
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
        datafolder = '../data/test_dataset'

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

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=256,
            shuffle=True,
        )
        for _ in range(3):
            start = time.time()
            for i, (images, targets) in enumerate(tqdm(dataloader)):
                pass
                print('#', i)
                print('image shape:', images.shape)
                print(targets['masks'].shape)
            end = time.time()
            print('elapsed time:', end - start)


if __name__ == "__main__":
    unittest.main()
