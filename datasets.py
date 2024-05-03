# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os

from torchvision import datasets, transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


def build_dataset(is_train, args):

    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, "train" if is_train else "val")
    if args.is_test:
        transform = build_transform(False, args)
        root = os.path.join(args.data_path, "test")
    dataset = datasets.ImageFolder(root, transform=transform)
    nb_classes = args.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    # resize_im = args.input_size > 32
    # if is_train and args.augment_train_data:
    #     # this should always dispatch to transforms_imagenet_train
    #     transform = create_transform(
    #         input_size=args.input_size,
    #         is_training=True,
    #         color_jitter=args.color_jitter,
    #         auto_augment=args.aa,
    #         interpolation=args.train_interpolation,
    #         re_prob=args.reprob,
    #         re_mode=args.remode,
    #         re_count=args.recount,
    #     )
    #     if not resize_im:
    #         # replace RandomResizedCropAndInterpolation with
    #         # RandomCrop
    #         transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
    #     return transform

    t = []
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

    # if resize_im and args.augment_train_data:
    #     size = int(args.input_size / args.eval_crop_ratio)
    #     t.append(
    #         transforms.Resize(
    #             size, interpolation=3
    #         ),  # to maintain same ratio w.r.t. 224 images
    #     )
    #     t.append(transforms.CenterCrop(args.input_size))
    #     return transforms.Compose(t)

    t.append(transforms.Resize(tuple(args.input_size)))
    return transforms.Compose(t)
