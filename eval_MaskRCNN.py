import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
from PIL import Image
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
from plot_results import plot_segmentation_masks

def get_model_instance_segmentation(num_classes):
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
