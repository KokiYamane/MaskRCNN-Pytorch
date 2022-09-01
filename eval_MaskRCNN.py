import torchvision
import torch
from PIL import Image
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
from plot_results import plot_segmentation_masks
from MaskRCNNModel import get_model_instance_segmentation


def main(args):
    # pass
    model = get_model_instance_segmentation(num_classes=5)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    model.to('cuda')

    image = Image.open(args.image).convert('RGB')
    print(image)
    image = torchvision.transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    image = image.to('cuda')
    print(image.shape)
    result = model(image)

    print(result)

    images_np = [image.to('cpu').detach().numpy() for image in image]
    output_np = [
        {k: v.to('cpu').detach().numpy() for k, v in output.items()}
        for output in result
    ]
    fig = plt.figure(figsize=(10, 10))
    plot_segmentation_masks(fig, images_np, output_np)
    fig.savefig('./results/eval.png')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='./results/test/')
    parser.add_argument('--image', type=str, default='./results/test/')
    args = parser.parse_args()
    main(args)
