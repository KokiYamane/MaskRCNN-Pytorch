import os
import numpy as np
import torch
from PIL import Image
import glob


class InstanceSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned

        self.image_paths = []
        self.mask_paths = []
        self.label = []
        classes = sorted(glob.glob(os.path.join(root, '*')))
        for label, path in enumerate(classes):
            print(f'loading data from: {path}')
            image_paths = sorted(
                glob.glob(os.path.join(path, 'originals', '*')))
            mask_paths = sorted(
                glob.glob(os.path.join(path, 'instance_segmentations', '*')))
            self.image_paths.extend(image_paths)
            self.mask_paths.extend(mask_paths)
            self.label.extend([label] * len(image_paths))

    def __getitem__(self, idx):
        # load images and masks
        # img_path = os.path.join(
        #     self.root, 'originals', self.imgs[idx])
        # mask_path = os.path.join(
        #     self.root, 'instance_segmentations', self.masks[idx])
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        img = Image.open(image_path).convert('RGB')
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        # print('t max:', np.max(masks))

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        # labels = torch.ones((num_objs,), dtype=torch.int64)
        labels = self.label[idx] * torch.ones((num_objs,), dtype=torch.int64)

        if self.transforms is not None:
            img = self.transforms(img)
            masks = self.transforms(masks)

        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        return img, target

    def __len__(self):
        return len(self.image_paths)
