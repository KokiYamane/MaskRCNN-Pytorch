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
    # os.makedirs(os.path.join(args.data, 'segmentations'))
    device = torch.device(
        f'cuda:{args.gpu[0]}'if torch.cuda.is_available() else 'cpu')

    model = get_model_instance_segmentation(num_classes=2)
    print(model)
    model = model.to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    print(model)

    image_paths = glob.glob(os.path.join(args.data, 'originals', '*'))
    print(image_paths)

    # transform = T.Compose([
    #     T.ToTensor(),
    # ])
    output_path = os.path.join(args.data, 'segmentations')
    os.makedirs(output_path, exist_ok=True)

    for image_path in tqdm(image_paths):
        # print(image_path)
        image = Image.open(image_path).convert('RGB')
        image = T.ToTensor()(image)
        # print(image)
        image = image.to(device)
        # print(image.shape)
        result = model(image.unsqueeze(0))
        # print(result)
        for i in range(len(result[0]['boxes'])):
            x1, y1, x2, y2 = result[0]['boxes'][i].int()
            image_part = image[:, y1:y2, x1:x2]
            # mask = result[0]['masks'][i][:, y1:y2, x1:x2]
            # image_part = torch.mul(image_part, mask)
            image_part = T.ToPILImage()(image_part)
            image_path = os.path.splitext(image_path)[0]
            image_part.save(
                os.path.join(
                    output_path,
                    '%s_%d.png' % (os.path.basename(image_path), i),
                ))


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
