# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
from typing import Tuple
from altair import sample
import numpy as np
import os
import time
from pathlib import Path

import sys
sys.path.append('/home/dingzhengyao/Work/PPG-ECG/Project/P2E_v2/Signal_SSL')
import torch
from torch.utils.data import Subset, ConcatDataset,RandomSampler
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
# sys.path.append("..")
import timm

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.callbacks import EarlyStop
import model.models_mae as models_mae
from engine_pretrain import train_one_epoch, evaluate
from data.dataset import Processed_Signal_dataset


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    # Basic parameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--debug', action='store_true')
    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patchX', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_channels', type=int, default=1, metavar='N',
                        help='input channels')
    parser.add_argument('--input_electrodes', type=int, default=1, metavar='N',
                        help='input electrodes')
    parser.add_argument('--time_steps', type=int, default=2560, metavar='N',
                        help='input length')
    parser.add_argument('--input_size', default=(1, 2560), type=Tuple,
                        help='images input size')
                        
    parser.add_argument('--patch_height', type=int, default=1, metavar='N',
                        help='patch height')
    parser.add_argument('--patch_width', type=int, default=128, metavar='N',
                        help='patch width')
    parser.add_argument('--patch_size', default=(1, 128), type=Tuple,
                        help='patch size')

    parser.add_argument('--norm_pix_loss', action='store_true', default=False,
                        help='Use (per-patch) normalized pixels as targets for computing loss')

    parser.add_argument('--ncc_weight', type=float, default=0.1,
                        help='Add normalized cross-correlation (ncc) as additional loss term')
    
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Callback parameters
    parser.add_argument('--patience', default=-1, type=float,
                        help='Early stopping whether val is worse than train for specified nb of epochs (default: -1, i.e. no early stopping)')
    parser.add_argument('--max_delta', default=0, type=float,
                        help='Early stopping threshold (val has to be worse than (train+delta)) (default: 0)')

    # Dataset parameters
    parser.add_argument('--data_path', default="/mnt/data2/PPG/dataset_all/processed_data/ppg_data_train.npz", type=str,
                        help='dataset path')
    parser.add_argument('--val_data_path', default="/mnt/data2/PPG/dataset_all/processed_data/ppg_data_test.npz", type=str,
                        help='validation dataset path')
    parser.add_argument('--miniset', default='false', type=str2bool)
    parser.add_argument('--signal', default='ppg', type=str)
    parser.add_argument('--output_dir', default=None, help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None, help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda:3',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def main(args):
    args.input_size = (args.input_electrodes, args.time_steps)
    args.patch_size = (args.patch_height, args.patch_width)
    args.distributed = False

    if args.signal == 'ppg':
        # args.data_path = '/mnt/data2/PPG/dataset_all/processed_data/ppg_data_train.npz'
        args.data_path = '/mnt/sda1/dingzhengyao/Work/PPG_ECG/processed_data/ppg_data_train.npz'
        # args.val_data_path = '/mnt/data2/PPG/dataset_all/processed_data/ppg_data_test.npz'
        args.val_data_path = '/mnt/sda1/dingzhengyao/Work/PPG_ECG/processed_data/ppg_data_test.npz'

    elif args.signal == 'ecg':
        # args.data_path = '/mnt/data2/PPG/dataset_all/processed_data/ecg_data_train.npz'
        args.data_path = '/mnt/sda1/dingzhengyao/Work/PPG_ECG/processed_data/ecg_data_train.npz'
        # args.val_data_path = '/mnt/data2/PPG/dataset_all/processed_data/ecg_data_test.npz'
        args.val_data_path = '/mnt/sda1/dingzhengyao/Work/PPG_ECG/processed_data/ecg_data_test.npz'

    device = torch.device(args.device)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # define the model
    model = models_mae.__dict__[args.model](
        img_size=args.input_size,
        patch_size=args.patch_size,
        norm_pix_loss=args.norm_pix_loss,
        ncc_weight=args.ncc_weight,
        in_chans=args.input_channels,
    )
    # load data
    print(f'Loading data')
    dataset_train = Processed_Signal_dataset(data_index=args.data_path, args=args)
    dataset_val = Processed_Signal_dataset(data_index=args.val_data_path, args=args)

    print("Training set size: ", len(dataset_train))
    print("Validation set size: ", len(dataset_val))



    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        sampler=RandomSampler(dataset_train),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        sampler=RandomSampler(dataset_val),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    
    

    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))
    print('Number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 4

    print("base lr: %.2e" % (args.lr * 4 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Define callbacks
    early_stop = EarlyStop(patience=args.patience, max_delta=args.max_delta)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    eval_criterion = "ncc"
    best_stats = {'loss':np.inf, 'ncc':0.0}
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        # if args.output_dir and (epoch % 100 == 0 or epoch + 1 == args.epochs):
        #     misc.save_model(
        #         args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #         loss_scaler=loss_scaler, epoch=epoch)

        val_stats = evaluate(data_loader_val, model, device, epoch, log_writer=log_writer, args=args)
        print(f"Loss / Normalized CC of the network on the {len(dataset_val)} val images: {val_stats['loss']:.4f} / {val_stats['ncc']:.2f}")

        if eval_criterion == "loss":
            if early_stop.evaluate_decreasing_metric(val_metric=val_stats[eval_criterion]):
                break
            if args.output_dir and val_stats[eval_criterion] <= best_stats[eval_criterion]:
                misc.save_best_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, test_stats=val_stats, evaluation_criterion=eval_criterion)
        else:
            if early_stop.evaluate_increasing_metric(val_metric=val_stats[eval_criterion]):
                break
            if args.output_dir and val_stats[eval_criterion] >= best_stats[eval_criterion]:
                misc.save_best_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, test_stats=val_stats, evaluation_criterion=eval_criterion)

        best_stats['loss'] = min(best_stats['loss'], val_stats['loss'])
        best_stats['ncc'] = max(best_stats['ncc'], val_stats['ncc'])
            
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
