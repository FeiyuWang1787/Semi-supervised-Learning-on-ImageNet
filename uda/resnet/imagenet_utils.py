# -*- coding: utf-8 -*-
# File: imagenet_utils.py


import functools
import cv2
import numpy as np
import tqdm
import multiprocessing
import tensorflow as tf
from abc import abstractmethod

from tensorpack import *
from tensorpack import ModelDesc
from tensorpack.input_source import QueueInput, StagingInput
from tensorpack.dataflow import (
    JoinData, imgaug, dataset, AugmentImageComponent, 
    PrefetchDataZMQ, BatchData, MultiThreadMapData)
from tensorpack.predict import PredictConfig, FeedfreePredictor
from tensorpack.utils.stats import RatioCounter
from tensorpack.models import regularize_cost
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.common import get_global_step_var
from tensorpack.utils import logger
import ilsvrcsemi
#from flip_gradient import flip_gradient
from unlabeled_utils import AugmentImageComponentTwice


class GoogleNetResize(imgaug.ImageAugmentor):
    """
    crop 8%~100% of the original image
    See `Going Deeper with Convolutions` by Google.
    """
    def __init__(self, crop_area_fraction=0.08,
                 aspect_ratio_low=0.75, aspect_ratio_high=1.333,
                 target_shape=224
        ):
        self._init(locals())

    def _augment(self, img, _):
        h, w = img.shape[:2]
        area = h * w
        for _ in range(10):
            targetArea = self.rng.uniform(self.crop_area_fraction, 1.0) * area
            aspectR = self.rng.uniform(self.aspect_ratio_low, self.aspect_ratio_high)
            ww = int(np.sqrt(targetArea * aspectR) + 0.5)
            hh = int(np.sqrt(targetArea / aspectR) + 0.5)
            if self.rng.uniform() < 0.5:
                ww, hh = hh, ww
            if hh <= h and ww <= w:
                x1 = 0 if w == ww else self.rng.randint(0, w - ww)
                y1 = 0 if h == hh else self.rng.randint(0, h - hh)
                out = img[y1:y1 + hh, x1:x1 + ww]
                out = cv2.resize(out, (self.target_shape, self.target_shape), interpolation=cv2.INTER_CUBIC)
                return out
        out = imgaug.ResizeShortestEdge(self.target_shape, interp=cv2.INTER_CUBIC).augment(img)
        out = imgaug.CenterCrop(self.target_shape).augment(out)
        return out


def fbresnet_augmentor(isTrain, resize):
    """
    Augmentor used in fb.resnet.torch, for BGR images in range [0,255].
    """
    if isTrain:
        augmentors = [GoogleNetResize()] if resize else []
        augmentors += [
            #GoogleNetResize(),
            # It's OK to remove the following augs if your CPU is not fast enough.
            # Removing brightness/contrast/saturation does not have a significant effect on accuracy.
            # Removing lighting leads to a tiny drop in accuracy.
            imgaug.RandomOrderAug(
                [# We removed the following augmentation
                 #imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 #imgaug.Contrast((0.6, 1.4), clip=False),
                 #imgaug.Saturation(0.4, rgb=False),
                 #rgb-bgr conversion for the constants copied from fb.resnet.torch
                 #imgaug.Lighting(0.1,
                 #                eigval=np.asarray(
                 #                    [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                 #                eigvec=np.array(
                 #                    [[-0.5675, 0.7192, 0.4009],
                 #                     [-0.5808, -0.0045, -0.8140],
                 #                     [-0.5836, -0.6948, 0.4203]],
                 #                    dtype='float32')[::-1, ::-1]
                 #                )
                ]
            ),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.CenterCrop((224, 224)),
        ]
    return augmentors


def get_imagenet_dataflow(
        datadir, name, batch_size, unlabeled_batch_mult, 
        augmentors_l, augmentors_u1, augmentors_u2, parallel=None, balanced=True
    ):
    """
    See explanations in the tutorial:
    http://tensorpack.readthedocs.io/en/latest/tutorial/efficient-dataflow.html
    """
    assert name in ['train', 'val', 'test']
    assert datadir is not None
    assert isinstance(augmentors_l, list)
    assert isinstance(augmentors_u1, list)
    assert isinstance(augmentors_u2, list)
    isTrain = name == 'train'
    if parallel is None:
        parallel = min(40, 8)  # assuming hyperthreading
    if isTrain:
        ds1 = ilsvrcsemi.ILSVRC12(datadir, name, shuffle=True, labeled=True, balanced=balanced)
        ds1 = AugmentImageComponent(ds1, augmentors_l, copy=False)
        ds2 = ilsvrcsemi.ILSVRC12(datadir, name, shuffle=True, labeled=False, balanced=balanced)
        ds2 = AugmentImageComponentTwice(ds2, 
            [GoogleNetResize()], augmentors_u1, augmentors_u2, copy=False
        )
        
        ds1 = BatchData(ds1, batch_size, remainder=False)
        ds2 = BatchData(ds2, batch_size * unlabeled_batch_mult, remainder=False)
        ds = JoinData([ds1, ds2])

        if parallel < 16:
            logger.warn("DataFlow may become the bottleneck when too few processes are used.")
        ds = PrefetchDataZMQ(ds, parallel)
    else:
        ds = dataset.ILSVRC12Files(datadir, name, shuffle=False)
        aug = imgaug.AugmentorList(augmentors_l)

        def mapf(dp):
            fname, cls = dp
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            im = aug.augment(im)
            return im, cls, im, cls, im
        
        def mapf1(dp):
            fname, cls = dp
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            im = aug.augment(im)
            return im, cls

        def mapf2(dp):
            fname, cls = dp
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            im = aug.augment(im)
            return im, cls, im
        
        ds = MultiThreadMapData(ds, parallel, mapf, buffer_size=5000, strict=True)
        ds = BatchData(ds, batch_size, remainder=True)

        #ds2 = ilsvrcsemi.ILSVRC12(datadir, 'train', shuffle=False, labeled=False, balanced=balanced)
        #ds2 = AugmentImageComponentTwice(ds2,
        #    [GoogleNetResize()], augmentors_u1, augmentors_u1, copy=False
        #)
        #ds2 = BatchData(ds2, batch_size, remainder=False)
        #ds = JoinData([ds, ds2])
        ds = PrefetchDataZMQ(ds, 1)
    return ds


def eval_on_ILSVRC12(model, sessinit, dataflow):
    pred_config = PredictConfig(
        model=model,
        session_init=sessinit,
        input_names=['input', 'label', 'input2', 'label2', 'image3'],
        #input_names = ['input', 'label'],
        output_names=['wrong-top1', 'wrong-top5']
    )
    acc1, acc5 = RatioCounter(), RatioCounter()

    # This does not have a visible improvement over naive predictor,
    # but will have an improvement if image_dtype is set to float32.
    pred = FeedfreePredictor(pred_config, StagingInput(QueueInput(dataflow), device='/gpu:0'))
    for _ in tqdm.trange(dataflow.size()):
        top1, top5 = pred()
        batch_size = top1.shape[0]
        acc1.feed(top1.sum(), batch_size)
        acc5.feed(top5.sum(), batch_size)

    print("Top1 Error: {}".format(acc1.ratio))
    print("Top5 Error: {}".format(acc5.ratio))


class ImageNetModel(ModelDesc):
    image_shape = 224

    """
    uint8 instead of float32 is used as input type to reduce copy overhead.
    It might hurt the performance a liiiitle bit.
    The pretrained models were trained with float32.
    """
    image_dtype = tf.uint8

    """
    Either 'NCHW' or 'NHWC'
    """
    data_format = 'NCHW'

    """
    Whether the image is BGR or RGB. If using DataFlow, then it should be BGR.
    """
    image_bgr = True

    """
    To apply on normalization parameters, use '.*/W|.*/gamma|.*/beta'
    """
    weight_decay_pattern = '.*/W|.*/beta|.*/gamma'
    #weight_decay_pattern = '.*/W'
    weight_decay = 2e-4

    """
    Scale the loss, for whatever reasons (e.g., gradient averaging, fp16 training, etc)
    """
    loss_scale = 1.

    """
    Label smoothing (See tf.losses.softmax_cross_entropy)
    """
    label_smoothing = 0.1
    """
    UDA configuration
    """
    uda_temp = 0.4
    uda_threshold = 0.5
    warmup_epochs = 1
    """
    Other training configuration
    """
    epochs = 18
    train_size = 1280000
    initial_lr = 0.3
    bn_momentum = 0.99


    def inputs(self):
        return [tf.placeholder(self.image_dtype, (None, self.image_shape, self.image_shape, 3), 'input'),
                tf.placeholder(tf.int32, (None,), 'label'),
                tf.placeholder(self.image_dtype, (None, self.image_shape, self.image_shape, 3), 'input2'),
                tf.placeholder(tf.int32, (None,), 'label2'),
                tf.placeholder(self.image_dtype, (None, self.image_shape, self.image_shape, 3), 'input3')
               ]

    def build_graph(self, image1, label1, image2, _, image3):
        global_step = tf.cast(get_global_step_var(), tf.float32)
        is_training = get_current_tower_context().is_training
        image1, image2, image3 = \
            [self.image_preprocess(x) for x in (image1, image2, image3)]

        assert self.data_format in ('NCHW', 'NHWC')
        if self.data_format == 'NCHW':
            image1, image2, image3 = \
                [tf.transpose(x, (0, 3, 1, 2)) for x in (image1, image2, image3)]

        get_logits_fn = functools.partial(
            self.get_logits,
            bn_fn=functools.partial(BatchNorm, momentum=self.bn_momentum),
            bnrelu_fn=functools.partial(BNReLU, momentum=self.bn_momentum)
        )

        # Training graph
        if is_training:
            image2_ori, image2_aug = image2, image3
            image_diff = tf.reduce_mean(tf.abs(image2_ori - image2_aug), name='image-diff')
            add_moving_summary(image_diff)

            image_all = tf.concat([image1, image2_ori, image2_aug], axis=0)
            logits_all, _ = get_logits_fn(image_all)

            batch_size = tf.shape(image1)[0]
            logits_sup, logits_unsup_ori, logits_unsup_aug = tf.split(
                logits_all, 
                [batch_size, batch_size*self.unl_mult, batch_size*self.unl_mult],
                axis=0
            )
            # Supervised loss
            label_sup_ori = label1
            label_sup = tf.to_float(tf.one_hot(label1, 1000))
            loss_sup = ImageNetModel.compute_loss_and_error(
                logits_sup, label_sup, label_smoothing=self.label_smoothing,
                label_ori=label_sup_ori
            )
            # Instance-wise unsupervised loss
            label_unsup_ori = tf.nn.softmax(logits_unsup_ori / self.uda_temp, axis=-1)
            label_unsup_ori = tf.stop_gradient(label_unsup_ori)
            loss_unsup = - tf.reduce_sum(
                label_unsup_ori * tf.nn.log_softmax(logits_unsup_aug, axis=-1),
                axis=-1
            )
            # Mask for instance-wise unsupervised losses
            largest_probs = tf.reduce_max(label_unsup_ori, axis=-1)
            mask_unsup = tf.cast(tf.greater_equal(largest_probs, self.uda_threshold), tf.float32)
            mask_ratio = tf.reduce_mean(mask_unsup, name='mask-ratio')
            add_moving_summary(mask_ratio)
            # Masked unsupervised loss
            loss_unsup = tf.reduce_mean(loss_unsup * mask_unsup, name='loss-unsupervised')
            add_moving_summary(loss_unsup)
            # UDA loss weight scheduling
            uda_weight = self.uda_weight * tf.minimum(
                (global_step + 1) / (self.warmup_epochs * self.train_size / self.batch_size),
                1.
            )
            tf.summary.scalar('loss-weight-uda', uda_weight)
            # Total loss
            loss = loss_sup + loss_unsup * uda_weight

            # Supervised plus entropy for debugging only
            #loss_ent = - tf.reduce_sum(
            #    label_unsup_ori * tf.nn.log_softmax(logits_unsup_aug, axis=-1), 
            #    axis=-1
            #)
            #loss_ent = tf.reduce_mean(loss_ent, name='loss-entmin')
            #add_moving_summary(loss_ent)
            #loss = loss_sup + loss_ent * uda_weight
    
        # Test graph
        else:
            logits, _ = get_logits_fn(image1)
            label_ori =label1
            label = tf.to_float(tf.one_hot(label1, 1000))
            loss_sup = ImageNetModel.compute_loss_and_error(
                logits, label, label_smoothing=self.label_smoothing, 
                label_ori=label_ori
            )
            loss = loss_sup
        add_moving_summary(loss_sup)

        if self.weight_decay > 0:
            wd_loss = regularize_cost(self.weight_decay_pattern,
                                      tf.contrib.layers.l2_regularizer(self.weight_decay),
                                      name='l2_regularize_loss')
            add_moving_summary(wd_loss)
            total_cost = tf.add_n([loss, wd_loss], name='cost')
        else:
            total_cost = tf.identity(loss, name='cost')

        if self.loss_scale != 1.:
            logger.info("Scaling the total loss by {} ...".format(self.loss_scale))
            return total_cost * self.loss_scale
        else:
            return total_cost

    @abstractmethod
    def get_logits(self, image):
        """
        Args:
            image: 4D tensor of ``self.input_shape`` in ``self.data_format``

        Returns:
            Nx#class logits
        """

    def optimizer(self):
        lr = tf.get_variable('step_learning_rate', initializer=self.initial_lr, trainable=False)
        #lr = tf.train.cosine_decay(
        #    learning_rate=self.initial_lr,
        #    global_step=get_global_step_var(),
        #    decay_steps=tf.cast(self.epochs * self.train_size / self.batch_size, tf.int32),
        #    alpha=0.,
        #    name='cosine-learning-rate'
        #)
        tf.summary.scalar('step-learning-rate-summary', lr)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

    def image_preprocess(self, image):
        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)
            mean = [0.485, 0.456, 0.406]    # rgb
            std = [0.229, 0.224, 0.225]
            if self.image_bgr:
                mean = mean[::-1]
                std = std[::-1]
            image_mean = tf.constant(mean, dtype=tf.float32) * 255.
            image_std = tf.constant(std, dtype=tf.float32) * 255.
            image = (image - image_mean) / image_std
            return image

    @staticmethod
    def compute_loss_and_error(logits, label, label_smoothing=0., label_ori=-1):
        loss = tf.losses.softmax_cross_entropy(
                label, logits, label_smoothing=label_smoothing)
        loss = tf.reduce_mean(loss, name='loss-supervised')

        def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
            with tf.name_scope('prediction_incorrect'):
                x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
            return tf.cast(x, tf.float32, name=name)

        wrong = prediction_incorrect(logits, label_ori, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

        wrong = prediction_incorrect(logits, label_ori, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))
        return loss


if __name__ == '__main__':
    import argparse
    from tensorpack.dataflow import TestDataSpeed
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--aug', choices=['train', 'val'], default='val')
    args = parser.parse_args()

    if args.aug == 'val':
        augs = fbresnet_augmentor(False)
    elif args.aug == 'train':
        augs = fbresnet_augmentor(True)
    df = get_imagenet_dataflow(
        args.data, 'train', args.batch, augs)
    # For val augmentor, Should get >100 it/s (i.e. 3k im/s) here on a decent E5 server.
    TestDataSpeed(df).start()
