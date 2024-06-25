
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import sys
import os
from module.weight_init import trunc_normal_



torch.autograd.set_detect_anomaly(True)



import models_posemb


from loader import *
from utils import *


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='MAE pre-training')
parser.add_argument('root', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--train', default='../data/train.csv', type=str, metavar='PATH',
                    help='path to train data_root (default: none)')
parser.add_argument('--test', default='../data/test.csv', type=str, metavar='PATH',
                    help='path to test data_root (default: none)')
parser.add_argument('--fold', default=4, type=int, help='fold for val')
parser.add_argument('--test-only', action='store_true',
                    help='')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--weighted-sample', action='store_true',
                    help='')

# additional configs:
parser.add_argument('--finetune', default='', type=str,
                    help='finetune from checkpoint')
parser.add_argument('--global_pool', action='store_true')
parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                    help='Use class token instead of global pool for classification')
parser.add_argument('--lars', action='store_true',
                    help='Use LARS')

# pama specific configs:
parser.add_argument('--max-size', default=8, type=int,
                    help='number of classes (default: 5)')
parser.add_argument('--max-kernel-num', default=128, type=int,
                    help='images input size')
parser.add_argument('--patch-per-kernel', default=18, type=int,
                    help='images input size')
parser.add_argument('--num-classes', default=3, type=int,
                    help='number of classes (default: 3)')
parser.add_argument('--list-classes', default=['Normal', 'LUAD', 'LUAC'], type=list,
                    help='name list of classes')
parser.add_argument('--model', default='pama_vit_base', type=str, metavar='MODEL',
                    help='Name of model to train')
parser.add_argument('--polar-bins', default=8, type=int,
                    help='in_chans')
parser.add_argument('--kernel-drop', default=0.2, type=float,
                    help='kernel dropout rate.')
parser.add_argument('--input_size', default=2048, type=int,
                    help='images input size')

parser.add_argument('--in-chans', default=256, type=int,
                    help='in_chans')

parser.add_argument('--mask_ratio', default=0.75, type=float,
                    help='Masking ratio (percentage of removed patches).')

parser.add_argument('--norm_pix_loss', action='store_true',
                    help='Use (per-patch) normalized pixels as targets for computing loss')
parser.set_defaults(norm_pix_loss=False)

parser.add_argument('--save-path', default='../exp_results/bs1_075_vit_L_p16/',
                    help='Path where save the model checkpoint')


def main():
    args = parser.parse_args()
    args.checkpoint = os.path.join(args.save_path, "checkpoints")
    args.checkpoint_matrix = os.path.join(args.save_path, "checkpoint-matrix")
    args.checkpoint_roc = os.path.join(args.save_path, "checkpoint_roc")
    args.checkpoint_csv = args.save_path

    if args.checkpoint is not None:
        os.makedirs(args.checkpoint, exist_ok=True)
    if args.checkpoint_matrix:
        os.makedirs(args.checkpoint_matrix, exist_ok=True)
    if args.checkpoint_roc:
        os.makedirs(args.checkpoint_roc, exist_ok=True)

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

    # create model
    # print("=> creating model '{}'".format(args.arch))
    model = models_posemb.__dict__[args.model](num_kernel=args.max_kernel_num,
                                               in_chans=args.in_chans,
                                               kernel_drop=args.kernel_drop,
                                               embed_dim=1024, depth=4, num_heads=8,
                                               mlp_ratio=4., num_classes=args.num_classes,
                                               norm_pix_loss=args.norm_pix_loss,
                                               polar_bins=args.polar_bins)


    # load from pre-trained, before DistributedDataParallel constructor
    if args.finetune and not args.test_only:
        if os.path.isfile(args.finetune):
            print("=> loading checkpoint '{}'".format(args.finetune))
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.finetune, map_location='cpu')
            state_dict = checkpoint['state_dict']

            # rename byol pre-trained keys
            for k in list(state_dict.keys()):
                if k.startswith('module.'):
                    # remove prefix
                    state_dict[k[len("module."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            new_state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in state_dict and state_dict[k].shape != new_state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del state_dict[k]


            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            if args.global_pool:
                assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            else:
                assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
            print("=> missing_keys\n", msg.missing_keys)

            print("=> loaded pre-trained model '{}'".format(args.finetune))
            # manually initialize fc layer: following MoCo v3
            trunc_normal_(model.head.weight, std=0.01)
        else:
            print("=> no checkpoint found at '{}'".format(args.finetune))


    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    print(model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of requires_grad params (M): %.2f' % (n_parameters / 1.e6))
    n_parameters_full = sum(p.numel() for p in model.parameters())
    print('number of the whole params (M): %.2f' % (n_parameters_full / 1.e6))


    optimizer = torch.optim.Adam(model.parameters(), init_lr)
    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)

    if args.lars:
        print("=> use LARS optimizer.")
        from apex.parallel.LARC import LARC
        optimizer = LARC(optimizer=optimizer, trust_coefficient=.001, clip=False)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    # TCGALungKDataset
    if not args.test_only:
        train_dataset = TCGALungKDataset(
            args.root,
            args.train,
            set='finetune',
            max_size=args.max_size,
            fold=args.fold,
            max_kernel_num=args.max_kernel_num,
            patch_per_kernel=args.patch_per_kernel,
            polar_bins=args.polar_bins,
            args=args)
        valid_dataset = TCGALungKDataset(
            args.root,
            args.test,
            set="test",
            max_size=args.max_size,
            fold=args.fold,
            max_kernel_num=args.max_kernel_num,
            patch_per_kernel=args.patch_per_kernel,
            polar_bins=args.polar_bins,
            args=args)
    else:
        valid_dataset = TCGALungKDataset(
            args.root,
            args.test,
            set="test",
            max_size=args.max_size,
            fold=args.fold,
            max_kernel_num=args.max_kernel_num,
            patch_per_kernel=args.patch_per_kernel,
            polar_bins=args.polar_bins,
            args=args)

    print("train:", len(train_dataset))
    print("val:", len(valid_dataset))
    if args.weighted_sample:
        print('activate weighted sampling')
        if args.distributed:
            train_sampler = DistributedWeightedSampler(
                train_dataset, train_dataset.get_weights(), args.world_size, args.rank)
        else:
            train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
                train_dataset.get_weights(), len(train_dataset), replacement=True
            )
    else:
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    recorder = Record(args.checkpoint_csv + 'record.csv')

    if args.evaluate:
        validate(val_loader, model, 'test', args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train_record = train(train_loader, model, criterion, optimizer, epoch, args)
        # queue.print_mix_level()

        # evaluate on validation set
        val_record = validate(val_loader, model, criterion, epoch, args)
        recorder.update([str(epoch)] + list(train_record) + list(val_record))

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, False, filename='{}/checkpoint_{:04d}.pth.tar'.format(args.checkpoint, epoch))



def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1, top2],
        prefix='Train: ')

    cm = ConfusionMatrix(args.list_classes)
    auc_metric = AUCMetric(args.list_classes)

    model.train()

    end = time.time()
    for i, (wsidata, labels, slide_ids) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        device = args.gpu
        if args.gpu is not None:
            wsi_feat = wsidata[0].float().cuda(args.gpu, non_blocking=True)
            wsi_rd = wsidata[1].int().cuda(args.gpu, non_blocking=True)
            wsi_polar = wsidata[2].int().cuda(args.gpu, non_blocking=True)
            token_mask = wsidata[3].int().cuda(args.gpu, non_blocking=True)
            kernel_mask = wsidata[4].int().cuda(args.gpu, non_blocking=True)
        labels = labels.cuda(args.gpu, non_blocking=True)
        # compute output
        logits, kernel_tokens = model(wsi_feat, wsi_rd, wsi_polar, token_mask, kernel_mask, device,mask_ratio=args.mask_ratio)


        loss = criterion(logits, labels)
        # measure accuracy and record loss
        losses.update(loss.item(), wsi_feat.size(0))

        acc = accuracy(logits, labels, topk=(1, 2))
        acc1, acc2 = acc[0], acc[1]

        top1.update(acc1[0], wsi_feat.size(0))
        top2.update(acc2[0], wsi_feat.size(0))
        Y_prob = F.softmax(logits, dim=-1)
        cm.update_matrix(Y_prob, labels)
        auc_metric.update(logits, labels)

        optimizer.zero_grad()

        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is None:
                print(name)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    try:
        cm.plot_confusion_matrix(
            normalize=True, save_path='{}/[Train][{}] Confusion Matrix.jpg'.format(args.checkpoint_matrix, epoch))

        if args.num_classes == 2:
            # binary class
            micro_auc, macro_auc, weighted_auc = auc_metric.calc_binary_auc_score(), 0.0, 0.0
            auc_metric.plot_binary_roc_curve(
                os.path.join(args.checkpoint_roc, '[Train][{}]_every_class_roc.png'.format(epoch)))
        else:
            micro_auc, macro_auc, weighted_auc = auc_metric.calc_auc_score()
            auc_metric.plot_every_class_roc_curve(
                os.path.join(args.checkpoint_roc, '[Train][{}]_every_class_roc.png'.format(epoch)))
    except Exception as e:
        micro_auc, macro_auc, weighted_auc = 0, 0, 0

    print('[Train] train-loss={:.3f}\t loss={:.3f}\t  acc1={:.3f}\t micro_auc={:.3f}\n'.format(losses.avg, losses.avg, top1.avg, micro_auc))


    return '{:.3f}'.format(losses.avg), '{:.3f}'.format(losses.avg), \
           '{:.3f}'.format(top1.avg), '{:.3f}'.format(top2.avg), \
           '{:.3f}'.format(micro_auc), '{:.3f}'.format(macro_auc), '{:.3f}'.format(weighted_auc)



def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    # cls_losses = AverageMeter('Cls_loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top2],
        prefix='Test: ')

    cm = ConfusionMatrix(args.list_classes)
    auc_metric = AUCMetric(args.list_classes)
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        # try:
        for i, (wsidata, labels, slide_ids) in enumerate(val_loader):
            device = args.gpu
            if args.gpu is not None:
                wsi_feat = wsidata[0].float().cuda(args.gpu, non_blocking=True)
                wsi_rd = wsidata[1].int().cuda(args.gpu, non_blocking=True)
                wsi_polar = wsidata[2].int().cuda(args.gpu, non_blocking=True)
                token_mask = wsidata[3].int().cuda(args.gpu, non_blocking=True)
                kernel_mask = wsidata[4].int().cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)

            # compute output
            logits, kernel_tokens = model(wsi_feat, wsi_rd, wsi_polar, token_mask, kernel_mask, device,mask_ratio=args.mask_ratio)

            loss = criterion(logits, labels)

            acc = accuracy(logits, labels, topk=(1, 2))
            acc1, acc2 = acc[0], acc[1]

            top1.update(acc1[0], wsi_feat.size(0))
            top2.update(acc2[0], wsi_feat.size(0))

            # measure accuracy and record loss
            losses.update(loss.item(), wsi_feat.size(0))

            Y_prob = F.softmax(logits, dim=1)
            cm.update_matrix(Y_prob, labels)
            auc_metric.update(logits, labels)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        cm.plot_confusion_matrix(
            normalize=True, save_path='{}/[Eval][{}] Confusion Matrix.jpg'.format(args.checkpoint_matrix, epoch))
        if args.num_classes == 2:
            # binary class
            micro_auc, macro_auc, weighted_auc = auc_metric.calc_binary_auc_score(), 0.0, 0.0
            f1_micro, f1_macro = auc_metric.calc_f1_score()
            auc_metric.plot_binary_roc_curve(
                os.path.join(args.checkpoint_roc, '[Eval][{}]_every_class_roc.png'.format(epoch)))
        else:
            micro_auc, macro_auc, weighted_auc = auc_metric.calc_auc_score()
            f1_micro, f1_macro = auc_metric.calc_f1_score()
            auc_metric.plot_every_class_roc_curve(
                os.path.join(args.checkpoint_roc, '[Eval][{}]_every_class_roc.png'.format(epoch)))
        print('[Eval] eval-loss={:.3f}\t loss={:.3f}\t  acc1={:.3f}\t micro_auc={:.3f}\n'.format(losses.avg, losses.avg, top1.avg, micro_auc))

        return '{:.3f}'.format(losses.avg), '{:.3f}'.format(losses.avg), \
               '{:.3f}'.format(top1.avg), '{:.3f}'.format(top2.avg), \
               '{:.3f}'.format(micro_auc), '{:.3f}'.format(macro_auc), '{:.3f}'.format(weighted_auc), \
               '{:.3f}'.format(f1_micro), '{:.3f}'.format(f1_macro)



if __name__ == '__main__':
    main()
