#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: imagenet-resnet.py

import argparse
import os

from tensorpack import logger, QueueInput
from tensorpack.models import *
from tensorpack.callbacks import *
from tensorpack.train import (
     AutoResumeTrainConfig, SyncMultiGPUTrainerReplicated, launch_train_with_config)
from tensorpack.dataflow import FakeData
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.utils.gpu import get_num_gpu

from imagenet_utils import (
    fbresnet_augmentor, get_imagenet_dataflow, ImageNetModel,
    eval_on_ILSVRC12)
from resnet_model import (
    preresnet_group, preresnet_basicblock, preresnet_bottleneck,
    resnet_group, resnet_basicblock, resnet_bottleneck, se_resnet_bottleneck,
    resnet_backbone)
from randaug_utils import randaug_augmentor


class Model(ImageNetModel):
    def __init__(self, depth, mode, batch_size, unlabeled_batch_mult, uda_weight):
        if mode == 'se':
            assert depth >= 50

        self.uda_weight = uda_weight
        self.batch_size = batch_size
        self.unl_mult = unlabeled_batch_mult
        self.mode = mode
        basicblock = preresnet_basicblock if mode == 'preact' else resnet_basicblock
        bottleneck = {
            'resnet': resnet_bottleneck,
            'preact': preresnet_bottleneck,
            'se': se_resnet_bottleneck}[mode]
        self.num_blocks, self.block_func = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck),
            152: ([3, 8, 36, 3], bottleneck)
        }[depth]

    def get_logits(self, image, bn_fn=BatchNorm, bnrelu_fn=BNReLU):
        with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format=self.data_format):
            return resnet_backbone(
                image, self.num_blocks,
                preresnet_group if self.mode == 'preact' else resnet_group, self.block_func,
                bn_fn, bnrelu_fn
            )


def get_data(name, batch, unlabeled_batch_mult):
    isTrain = name == 'train'
    augmentors_l = fbresnet_augmentor(isTrain, resize=True)
    augmentors_u1 = fbresnet_augmentor(isTrain, resize=False)
    augmentors_u2 = randaug_augmentor(isTrain, resize=False)
    #augmentors_u2 = fbresnet_augmentor(isTrain, resize=False)
    return get_imagenet_dataflow(
        args.data, name, batch, unlabeled_batch_mult, 
        augmentors_l, augmentors_u1, augmentors_u2, balanced=True
    )


def get_config(model, fake=False, unlabeled_batch_mult=7):
    nr_tower = max(get_num_gpu(), 1)
    assert args.batch % nr_tower == 0
    batch = args.batch // nr_tower

    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch))
    if batch < 32 or batch > 64:
        logger.warn("Batch size per tower not in [32, 64]. This probably will lead to worse accuracy than reported.")
    if fake:
        data = QueueInput(FakeData(
            [[batch, 224, 224, 3], [batch],[batch, 224, 224, 3], [batch]], 1000, random=False, dtype='uint8'))
        callbacks = []
    else:
        data = QueueInput(get_data('train', batch, unlabeled_batch_mult))
        START_LR = 0.3
        BASE_LR = START_LR * (args. batch / 256.)
        callbacks = [
            ModelSaver(),
            EstimatedTimeLeft(),
            ScheduledHyperParamSetter(
                'step_learning_rate',
                [(0, min(START_LR, BASE_LR)), (6, BASE_LR * 1e-1), (12, BASE_LR * 1e-2), (16, BASE_LR * 1e-3)]
            ), 
            #ScheduledHyperParamSetter('step_learning_rate', [(0, min(START_LR, BASE_LR))])
        ]
        if BASE_LR > START_LR:
            callbacks.append(
                ScheduledHyperParamSetter(
                    'step_learning_rate', [(0, START_LR), (5, BASE_LR)], interp='linear')
            )

        infs = [ClassificationError('wrong-top1', 'val-error-top1'),
                ClassificationError('wrong-top5', 'val-error-top5')
        ]
        dataset_val = get_data('val', batch, unlabeled_batch_mult)
        if nr_tower == 1:
            # single-GPU inference with queue prefetch
            callbacks.append(InferenceRunner(QueueInput(dataset_val), infs))
        else:
            # multi-GPU inference (with mandatory queue prefetch)
            callbacks.append(DataParallelInferenceRunner(
                dataset_val, infs, list(range(nr_tower))))

    return AutoResumeTrainConfig(
        model=model,
        data=data,
        callbacks=callbacks,
        steps_per_epoch=100 if args.fake else 1280000 // args.batch,
        max_epoch=18,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use. Default to use all available ones')
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--load', help='load a model for training or evaluation')
    parser.add_argument('--fake', help='use FakeData to debug or benchmark this model', action='store_true')
    parser.add_argument('--data-format', help='image data format',
                        default='NCHW', choices=['NCHW', 'NHWC'])
    parser.add_argument('-d', '--depth', help='ResNet depth',
                        type=int, default=50, choices=[18, 34, 50, 101, 152])
    parser.add_argument('--eval', type=int, default=0, choices=[0, 1])
    parser.add_argument('--batch', default=256, type=int,
                        help="total batch size. "
                        "Note that it's best to keep per-GPU batch size in [32, 64] to obtain the best accuracy."
                        "Pretrained models listed in README were trained with batch=32x8.")
    parser.add_argument('--unlabeled_batch_mult', type=int, default=1,
                        help='The batch size of unlabeled data equals args.batch * args.unlabeled_batch_mult.')
    parser.add_argument('--weight', type=float, default=1, help='Unsupervised loss weight.')
    parser.add_argument('--mode', choices=['resnet', 'preact', 'se'],
                        help='variants of resnet to use', default='resnet')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = Model(args.depth, args.mode, args.batch, args.unlabeled_batch_mult, args.weight)
    model.data_format = args.data_format
    if bool(args.eval):
        batch = 128    # something that can run on one gpu
        ds = get_data('val', batch, args.unlabeled_batch_mult)
        eval_on_ILSVRC12(model, get_model_loader(args.load), ds)
    else:
        if args.fake:
            logger.set_logger_dir(os.path.join('train_log', 'tmp'), 'd')
        else:
            _logger_dir = os.path.join('train_log', 'uda-temp0.4-thresh0.5-w{}-rampup1-samecrop-15augmentors-unlmult{}-imagenet-{}-d{}-batch{}'\
                .format(args.weight, args.unlabeled_batch_mult, args.mode, args.depth, args.batch))
            logger.set_logger_dir(_logger_dir)
            os.system('cp imagenet_utils.py {}'.format(_logger_dir))
            os.system('cp adanet-resnet.py {}'.format(_logger_dir))
            os.system('cp resnet_model.py {}'.format(_logger_dir))
            os.system('cp unlabeled_utils.py {}'.format(_logger_dir))
            os.system('cp randaug_utils.py {}'.format(_logger_dir))

        config = get_config(model, fake=args.fake, unlabeled_batch_mult=args.unlabeled_batch_mult)
        if args.load:
            config.session_init = get_model_loader(args.load)
        trainer = SyncMultiGPUTrainerReplicated(max(get_num_gpu(), 1))
        launch_train_with_config(config, trainer)
