import os
import urllib.request

import mmcv
from mmrotate.models import build_detector
from mmcv.runner import load_checkpoint


def get_model_MMRotate(
    config_path: str,
    checkpoint_path: str,
    device: str = 'cuda',
):
    MMRotateModel_path = 'MMRotateModels/'

    # download model params if it doesn't exist
    checkpoint_save_path = os.path.join(MMRotateModel_path, checkpoint_path)
    if not os.path.exists(checkpoint_save_path):
        base_url = 'https://download.openmmlab.com/mmrotate/v0.1.0/'
        download_path = base_url + checkpoint_path
        print(f'Downloading checkpoint from {download_path}')

        # make directory if it doesn't exist
        dirname = os.path.dirname(checkpoint_save_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        urllib.request.urlretrieve(download_path, checkpoint_save_path)

    print('load config from local path:', config_path)
    config = mmcv.Config.fromfile(config_path)
    config.model.pretrained = None
    model = build_detector(config.model)

    checkpoint = load_checkpoint(
        model, checkpoint_save_path, map_location=device)
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.cfg = config

    return model


def main():
    config_path = '../mmrotate/configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py'
    checkpoint_path = 'rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90/rotated_retinanet_obb_r50_fpn_1x_dota_le90-c0097bc4.pth'

    model = get_model_MMRotate(config_path, checkpoint_path)
    # print(model)
    print('Classes:', model.CLASSES)


if __name__ == '__main__':
    main()
