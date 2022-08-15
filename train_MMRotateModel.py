import os
import time

from mmrotate.datasets.builder import ROTATED_DATASETS
from mmrotate.datasets.dota import DOTADataset
from mmdet.datasets import PIPELINES
import mmcv
import mmdet
import mmdet.apis

import torch
import torchvision.transforms as T

config_path = '../mmrotate/configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py'
checkpoint_path = './MMRotateModels/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90/rotated_retinanet_obb_r50_fpn_1x_dota_le90-c0097bc4.pth'


@ROTATED_DATASETS.register_module()
class TinyDataset(DOTADataset):
    # CLASSES = ('ship',)
    CLASSES = ('class',)


@PIPELINES.register_module()
class MyTransform:
    """Add your transform

    Args:
        p (float): Probability of shifts. Default 0.5.
    """

    def __init__(self, p=0.5):
        self.p = p

        self.transforms = T.Compose([
            T.ToTensor(),
            T.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                # hue=0.5,
            ),
            # T.ToPILImage(),
        ])

    def __call__(self, results):
        # if random.random() > self.p:
        #     results['dummy'] = True
        # return results
        # print(results)
        results['img'] = self.transforms(results['img'])

        # to int
        results['img'] = 255 * results['img']
        results['img'] = results['img'].permute(1, 2, 0)
        results['img'] = results['img'].detach().numpy().astype(dtype='uint8')
        # results['img'] = T.ToPILImage()(results['img'])

        return results


def main(args):
    config = mmcv.Config.fromfile(config_path)

    # data_path = './data/ssdd_tiny/'
    data_path = args.data
    dateset_type = 'TinyDataset'

    config.dataset_type = dateset_type
    config.data_root = data_path

    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
        dict(type='RRandomFlip', flip_ratio=0.5),
        # dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='MyTransform', p=0.2),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    ]

    # train
    config.data.train.type = dateset_type
    config.data.train.data_root = data_path
    config.data.train.ann_file = 'train'
    config.data.train.img_prefix = 'images'
    config.data.train.pipeline = train_pipeline

    # test
    # config.data.test.type = dateset_type
    # config.data.test.data_root = data_path
    # config.data.test.ann_file = 'val'
    # config.data.test.img_prefix = 'images'

    # val
    config.data.val.type = dateset_type
    config.data.val.data_root = data_path
    config.data.val.ann_file = 'val'
    config.data.val.img_prefix = 'images'

    # class num
    # config.model.roi_head.bbox_head[0].num_classes = 1
    # config.model.roi_head.bbox_head[1].num_classes = 1

    # model path
    if args.pretrained:
        config.load_from = checkpoint_path

    # output path
    config.work_dir = args.output

    # learning rate
    config.optimizer.lr = args.learning_rate
    # epoch
    config.runner.max_epochs = args.epoch

    # seed
    SEED = 12
    config.seed = SEED
    mmdet.apis.set_random_seed(SEED, deterministic=False)

    # device setting
    # config.device = f'cuda:{args.gpu[0]}'
    config.device = 'cuda'
    config.gpu_ids = args.gpu

    # log setting
    config.log_config.interval = 10
    config.log_config.hooks = [
        dict(type='TextLoggerHook'),
        # dict(
        #     type='MMDetWandbHook',
        #     init_kwargs={'project': 'mmrotate'},
        #     interval=10,
        #     log_checkpoint=True,
        #     log_checkpoint_metadata=True,
        #     num_eval_images=100,
        #     bbox_score_thr=0.3,
        # )
    ]

    # make dataset
    datasets = [mmdet.datasets.build_dataset(config.data.train)]
    # config.workflow = [('train', 1), ('val', 1)]
    # if len(config.workflow) == 2:
    #     datasets.append(mmdet.datasets.build_dataset(config.data.val))

    # make model
    model_for_train = mmdet.models.build_detector(
        config.model,
        train_cfg=config.get('train_cfg'),
        test_cfg=config.get('test_cfg'),
    )

    # classes setting
    model_for_train.CLASSES = datasets[0].CLASSES

    # make output dir
    mmcv.mkdir_or_exist(os.path.abspath(config.work_dir))

    print(config.pretty_text)

    # train
    start = time.time()
    mmdet.apis.train_detector(
        model_for_train,
        datasets,
        config,
        distributed=False,
        # validate=True,
        validate=False,
    )
    end = time.time()
    print(f'Training time: {end - start:%2f} sec')


def argparse():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/ssdd_tiny/')
    parser.add_argument('--output', type=str,
                        default='./results/MMRotate_test/')
    parser.add_argument('--epoch', type=int, default=10000)
    # parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--gpu', default='0',
                        type=lambda x: list(map(int, x.split(','))))
    parser.add_argument('--pretrained', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argparse()
    main(args)
