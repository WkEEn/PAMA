import argparse
import builtins
import os
import sys
import time
import math
import random
import warnings
from pathlib import Path
import csv
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
import math

from PIL import Image
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils_patch as utils

import vision_transformer as vits
from vision_transformer import DINOHead
from loader_patch import SlideLocalTileDataset

torch.autograd.set_detect_anomaly(True)

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # dataset set
    parser.add_argument('root', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--data_path', default='../data/test.csv', type=str, metavar='PATH',
                        help='path to train data_root (default: none)')
    parser.add_argument('--test', default='../data/EGFR_slideId_alias_label_test_clean.csv', type=str, metavar='PATH',
                        help='path to test data_root (default: none)')
    parser.add_argument('--fold', default=0, type=int, help='fold for val')

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'],
                        help="""Name of architecture to train. For quick experiments with ViTs,
            we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
                        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd', 'lars'],
                        help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--n_last_blocks', default=1, type=int, help="""Concatenate [CLS] tokens
            for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
                        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
            We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    # parser.add_argument('--output-dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--save-dir', default=".", type=str, help='Path to save feature pth.')
    parser.add_argument('--error-txt', default=".", type=str, help='txt to save failed pth.')
    parser.add_argument('--saveckp_freq', default=2, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=None, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist-url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    # WSI set
    parser.add_argument('--mask-level', default=3, type=int,
                        help='')
    parser.add_argument('--image-level', default=1, type=int,
                        help='')

    return parser


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # slurmd settings
    args.rank = int(os.environ["SLURM_PROCID"])
    args.world_size = int(os.environ["SLURM_NPROCS"])

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # suppress printing if not master
    if args.multiprocessing_distributed and args.rank != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    utils.fix_random_seeds()
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    # if the network is a XCiT
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
        embed_dim = model.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
        model.fc = nn.Identity()
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)
    torch.cuda.set_device(args.gpu)
    model.cuda()
    model.eval()
    # load weights to evaluate
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Model {args.arch} built.")
    # multi-crop wrapper handles forward with inputs of different resolutions

    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    print("Starting DINO extract !")
    print(model)

    root = args.root
    obj = "Medium"

    wsi_list = []

    f = csv.reader(open(args.data_path))
    for i, row in enumerate(f):
        slide_name = str(row[0])
        wsi_list.append(slide_name)
    print("csv loaded!")

    data_path_new = args.save_dir
    os.makedirs(data_path_new, exist_ok=True)

    extract_feature(wsi_list, data_path_new, root, args, model, obj, transform, args.n_last_blocks, args.avgpool_patchtokens)




def add_orientation(polar_pos, bins):
    unit = np.pi / bins
    new_polar_pos = {}
    for kernal_x in polar_pos:
        k_x = polar_pos[kernal_x]
        temp_new_polar_pos = []
        for k in k_x:
            new_kernal_x = []
            for polar_degree in k:
                orientation = polar_degree // unit
                new_kernal_x.append([int(orientation) + bins, polar_degree])
            temp_new_polar_pos.append(new_kernal_x)
        new_polar_pos[kernal_x] = np.asarray(temp_new_polar_pos)
    return new_polar_pos



def extract_feature(slide_list, data_path_new, root, args, model, obj, transform, n, avgpool):
    for slide_name in slide_list:
        print("extracting {}".format(slide_name))
        slide_path_new = os.path.join(data_path_new, slide_name)
        if os.path.exists(os.path.join('{}.pth'.format(slide_path_new))):
            print(f'{slide_path_new} exists, skip!')
            continue
        slide_folder = os.path.join(root, slide_name, obj)
        if not os.path.exists(slide_folder):
            print(f'{slide_folder} not exists')
            continue
        img_size = 256
        # embedding info
        rl = args.mask_level - args.image_level
        filter_size = img_size >> rl
        max_nodes = 4096
        step = 256
        frstep = step >> rl
        intensity_thred = 25
        try:
            tissue_mask = utils.get_tissue_mask(cv2.imread(
                os.path.join(root, slide_name, 'Overview.jpg')))
            content_mat = cv2.blur(
                tissue_mask, ksize=(filter_size, filter_size), anchor=(0, 0))
            content_mat = content_mat[::frstep, ::frstep] > intensity_thred  
            patches_in_graph = np.sum(content_mat)

            sampling_mat = np.copy(content_mat)
            down_factor = 1

            if patches_in_graph > max_nodes:
                down_factor = int(np.sqrt(patches_in_graph / max_nodes)) + 1
                tmp = np.zeros(sampling_mat.shape, np.uint8) > 0
                tmp[::down_factor, ::down_factor] = sampling_mat[::down_factor, ::down_factor]
                sampling_mat = tmp
                patches_in_graph = np.sum(sampling_mat)

            patch_pos = np.transpose(np.asarray(np.where(sampling_mat)))
            # ajdacency_mat
            adj, re_dist = utils.connectivity_and_dist(patch_pos, down_factor)

            # bin re_dist
            bin_count = 64
            bin_dis = (np.max(re_dist) - np.min(re_dist)) / bin_count
            new_re_dist = re_dist // bin_dis

            PATCH_NUMBER_PER_ANCHOR = [9, 18, 36, 64, 100, 144, 256, 400]
            npks = PATCH_NUMBER_PER_ANCHOR
            k_indexes, kns = [], []
            for npk in npks:
                kn = int(patches_in_graph / npk) + 1 if int(patches_in_graph / npk) < np.sum(sampling_mat) else np.sum(
                    sampling_mat)
                kns.append(kn)
                kmeans_worker = KMeans(n_clusters=kn, random_state=9, max_iter=10)
                kmeans_worker.fit(patch_pos).transform(patch_pos)
                k_indexes.append(np.argmin(kmeans_worker.fit(patch_pos).transform(patch_pos), axis=0))

            polar_pos = {}
            for index, kn in zip(k_indexes, kns):
                ker_pos = patch_pos[index]
                ker_polar = np.zeros((kn, patches_in_graph))
                for i, k in enumerate(ker_pos):
                    for j, p in enumerate(patch_pos):
                        recor = p - k
                        ker_polar[i][j] = math.atan2(recor[1], recor[0])
                polar_pos[kn] = ker_polar
            
            polar_list_dict ={}
            for bins in [2, 4, 8, 16]:
                new_polar_pos = add_orientation(polar_pos, int(bins / 2))
                polar_list_dict[bins]=new_polar_pos


            # embedding end

            slide_dataset = SlideLocalTileDataset(slide_folder, patch_pos * step, transform,
                                                    256, 256)
            slide_loader = torch.utils.data.DataLoader(
                slide_dataset, batch_size=args.batch_size_per_gpu, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, drop_last=False)
            patch_num = slide_dataset.__len__()
            print("slide {}: {} patches".format(slide_name, patch_num))
            patch_feature = []
            # switch to evaluate mode
            model.eval()
            print("slide {} inf model \n".format(slide_name))
            with torch.no_grad():
                end = time.time()
                for i, images in enumerate(slide_loader):
                    images = images.cuda(args.gpu, non_blocking=True)
                    intermediate_output = model.module.get_intermediate_layers(images, 1)
                    p1 = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                    if avgpool:
                        output = torch.cat(
                            (output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                        p1 = output.reshape(output.shape[0], -1)
                    dim = p1.shape[-1]
                    batch_feature = p1.cuda().data.cpu().numpy()
                    for z_ in batch_feature:
                        patch_feature.append(z_)

                feature_array = np.asarray(patch_feature).reshape([patch_num, dim])
                WSI_feature = {
                    'feature_array': feature_array,
                    'adj': adj,
                    're_dist': new_re_dist,
                    'kns': kns,
                    'npks': npks,
                    'k_index': k_indexes,
                    'cm': content_mat,
                    'down_factor': down_factor,
                    'patch_pos': patch_pos,
                    'polar_pos': polar_list_dict
                }
                torch.save(
                    WSI_feature,
                    os.path.join('{}.pth'.format(slide_path_new)))
            print("slide {} Done \n".format(slide_name))
        except Exception as e:
            print(f'{slide_name}: {e}')
            with open(args.error_txt, 'a', newline='') as f:
                f.write(f'{slide_name}\n')
            continue



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    main()
