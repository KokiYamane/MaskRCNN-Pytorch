import os
import torch
import torchvision.transforms as T
from PIL import Image
import glob
from tqdm import tqdm

import sys
sys.path.append('.')
from train_MaskRCNN import get_model_instance_segmentation


def main(args):
    model = get_model_instance_segmentation(num_classes=2)
    model = model.cuda()
    model.load_state_dict(torch.load(args.model))
    model.eval()

    image_paths = glob.glob(os.path.join(args.data, 'originals', '*'))

    transform = T.Compose([
        T.ToTensor(),
    ])

    for image_path in tqdm(image_paths):
        # print(image_path)
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        image = image.cuda()
        # print(image.shape)
        result = model(image.unsqueeze(0))
        # print(result)
        for i in range(len(result[0]['boxes'])):
            x1, y1, x2, y2 = result[0]['boxes'][i].int()
            image_part = image[:, y1:y2, x1:x2]
            mask = result[0]['masks'][i][:, y1:y2, x1:x2]
            image_part = torch.mul(image_part, mask)
            image_part = T.ToPILImage()(image_part)
            image_part.save(os.path.join(args.data, 'segmentations',
                            '%s_%d.png' % (os.path.basename(image_path), i)))


def argparse():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data', type=str,
                        default='../datasets/PennFudanPed/')
    parser.add_argument('--model', type=str)
    parser.add_argument('--output', type=str, default='./results/test/')
    parser.add_argument('--gpu', default='0',
                        type=lambda x: list(map(int, x.split(','))))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argparse()
    main(args)
