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
from random import shuffle
from typing import Tuple
import numpy as np
import os
import time
from pathlib import Path
# from pyinstrument import Profiler
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
import model.SingnalEncoder as SingnalEncoder
from engine_pretrain import train_one_epoch, evaluate
from data.dataset import get_dataset
from model.BothEncoder import BothEncoder
from model.EnhancedEncoder import EnhancedEncoder
import random
def set_random_seed(seed: int):
    """
    固定所有随机数种子，以确保实验结果可重复。
    
    参数:
    seed (int): 随机种子值
    """
    
    # 1. 固定 Python 随机数生成器的种子
    random.seed(seed)
    
    # 2. 固定 NumPy 随机数生成器的种子
    np.random.seed(seed)
    
    # 3. 固定 PyTorch 随机数生成器的种子
    torch.manual_seed(seed)
    
    # 4. 如果有 CUDA 支持，固定 GPU 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个GPU的话
        
        # 设置CUDA的初始化方式，使得结果可重复
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # 为了可复现，不启用优化

    print(f"随机种子 {seed} 已经设置完成，确保实验可复现。")

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
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory '
                             'constraints)')
    # Model parameters
    parser.add_argument('--startfrom_pretrained_model',default=None, type=str,help='path of pretrained model')
    parser.add_argument('--latent_dim', default=1, type=int, metavar='N',help='latent_dim')
    parser.add_argument('--model', default=None, type=str)
    parser.add_argument('--ecg_generator', default='/mnt/data2/PPG/Work/P2E_v2/Signal_generate/P2E-WGAN-ecg-ppg-reconstruction/172_0.3/saved_models/p2e_wgan/multi_models_10.pth', type=str, metavar='MODEL')
    # ECG Model parameters
    parser.add_argument('--ecg_pretrained', default=None, type=str,help='SSL or Align or None')
    parser.add_argument('--ecg_model', default='vit_base_patchX', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--ecg_pretrained_model',default="/mnt/data2/PPG/Work/P2E_v2/Signal_SSL/mae_vit_base_patchX/160_2.0_usecleandata/ep400_WARM_EP40_lr5e-7_bs3072_wd1e-4_sgecg/checkpoint-112-ncc-0.81.pth",
                        type=str, metavar='MODEL', help='path of pretaained model')
    parser.add_argument('--ecg_alignment_model', default='/mnt/data2/PPG/Work/P2E_v2/Signal_alignment/vit_base_patchX/159_0.1/ep200_WARM_EP20_lr5e-8_bs4096_wd0.05_freezeECGFalse/checkpoint-57-loss-2.13.pth', type=str, metavar='MODEL')
    parser.add_argument('--ecg_input_channels', type=int, default=1, metavar='N',
                        help='ecginput_channels')
    parser.add_argument('--ecg_input_electrodes', type=int, default=1, metavar='N',
                        help='ecg input electrodes')
    parser.add_argument('--ecg_time_steps', type=int, default=2560, metavar='N',
                        help='ecg input length')
    parser.add_argument('--ecg_input_size', default=(1, 2560), type=Tuple,
                        help='ecg input size')
    parser.add_argument('--ecg_patch_height', type=int, default=1, metavar='N',
                        help='ecg patch height')
    parser.add_argument('--ecg_patch_width', type=int, default=128, metavar='N',
                        help='ecg patch width')
    parser.add_argument('--ecg_patch_size', default=(1, 128), type=Tuple,
                        help='ecg patch size')
    parser.add_argument('--ecg_globle_pool', default=False,type=str2bool, help='ecg_globle_pool')
    parser.add_argument('--ecg_drop_out', default=0.1, type=float)
    
    # PPG Model parameters
    parser.add_argument('--ppg_pretrained', default=None, type=str,help='SSL or Align or None')
    parser.add_argument('--ppg_model', default='vit_base_patchX', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--ppg_pretrained_model',
                        default="/mnt/data2/PPG/Work/P2E_v2/Signal_SSL/mae_vit_base_patchX/160_2.0_usecleandata/ep400_WARM_EP40_lr5e-7_bs3072_wd1e-4_sgppg/checkpoint-394-ncc-0.98.pth",
                        type=str, metavar='MODEL', help='path of pretaained model')
    parser.add_argument('--ppg_alignment_model', default='/mnt/data2/PPG/Work/P2E_v2/Signal_alignment/vit_base_patchX/159_0.1/ep200_WARM_EP20_lr5e-8_bs4096_wd0.05_freezeECGFalse/checkpoint-57-loss-2.13.pth', type=str, metavar='MODEL')
    parser.add_argument('--ppg_input_channels', type=int, default=1, metavar='N',
                        help='ppginput_channels')
    parser.add_argument('--ppg_input_electrodes', type=int, default=1, metavar='N',
                        help='ppg input electrodes')
    parser.add_argument('--ppg_time_steps', type=int, default=2560, metavar='N',
                        help='ppg input length')
    parser.add_argument('--ppg_input_size', default=(1, 2560), type=Tuple,
                        help='ppg input size')
    parser.add_argument('--ppg_patch_height', type=int, default=1, metavar='N',
                        help='ppg patch height')
    parser.add_argument('--ppg_patch_width', type=int, default=128, metavar='N',
                        help='ppg patch width')
    parser.add_argument('--ppg_patch_size', default=(1, 128), type=Tuple,
                        help='ppg patch size')
    parser.add_argument('--ppg_globle_pool', default=False,type=str2bool, help='ppg_globle_pool')
    parser.add_argument('--ppg_drop_out', default=0.1, type=float)

    
    
    

    # LOSS parameters
    
    parser.add_argument('--temperature', default=0.1, type=float, metavar='TEMPERATURE',
                        help='temperature for nt_xent loss')
    parser.add_argument('--alpha_weight', default=0.5, type=float, metavar='ALPHA_WEIGHT',
                        help='alpha_weight for nt_xent loss')
    
    # Augmentation parameters
    parser.add_argument('--input_size', type=tuple, default=(1, 2560))

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Callback parameters
    parser.add_argument('--patience', default=10, type=float,
                        help='Early stopping whether val is worse than train for specified nb of epochs (default: -1, i.e. no early stopping)')
    parser.add_argument('--max_delta', default=0.005, type=float,
                        help='Early stopping threshold (val has to be worse than (train+delta)) (default: 0)')
    parser.add_argument('--metric_save_path', default=None, type=str, help='metric save path')
    # Dataset parameters
    parser.add_argument('--signal', default='ppg', type=str, help='signal type')
    parser.add_argument('--downtask_type', default='BCE', type=str, help='downtask type')
    parser.add_argument('--label', default='Hypertension', type=str, help='label type')
    parser.add_argument('--dataset', default='MIMIC', type=str, help='dataset type')
    

    parser.add_argument('--output_dir', default=None,
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None, help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda:3',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True, 
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    return parser

def main(args):
    args.input_size = (args.ecg_input_electrodes, args.ecg_time_steps)
    args.patch_size = (args.ecg_patch_height, args.ecg_patch_width)
    args.distributed = False

    args.metric_save_path = os.path.join(args.output_dir, 'test')
    # 如果args.metric_save_path文件夹不存在，则创造文件夹
    if not os.path.exists(args.metric_save_path):
        os.makedirs(args.metric_save_path)
    if args.downtask_type == 'BCE':
        assert args.latent_dim == 1
        args.criterion = 'AUC'
    elif args.downtask_type == 'CE':
        assert args.latent_dim > 1
        args.criterion = 'AUC'
    elif args.downtask_type == 'Regression':
        assert args.latent_dim == 1
        args.criterion = 'Pearsonr'

    device = torch.device(args.device)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    set_random_seed(seed)
    cudnn.benchmark = True
    assert args.ecg_model == args.ppg_model
    if args.ecg_model == 'vit_base_patchX':
        args.embed_dim = 768
    elif args.ecg_model == 'vit_large_patchX':
        args.embed_dim = 1024
    elif args.ecg_model == 'vit_huge_patchX':
        args.embed_dim = 1280

    print(f'args.signal: {args.signal}')
    # load data
    print(f'Loading data')
    dataset_train = get_dataset(args, mode='train')
    dataset_val = get_dataset(args, mode='test')

    print("Training set size: ", len(dataset_train))
    print("Validation set size: ", len(dataset_val))
    

    print('load model')
    if args.signal == 'ecg':
        model = SingnalEncoder.__dict__[args.ecg_model](
                img_size=args.ecg_input_size,
                patch_size=args.ecg_patch_size,
                in_chans=args.ecg_input_channels,
                num_classes=args.latent_dim,
                drop_rate=args.ecg_drop_out,
                args=args,)
        if args.ecg_pretrained:
            if args.ecg_pretrained == 'SSL':
                print("load pretrained ecg_model")
                ecg_checkpoint = torch.load(args.ecg_pretrained_model, map_location='cpu')
                ecg_checkpoint_model = ecg_checkpoint['model']
                msg = model.load_state_dict(ecg_checkpoint_model, strict=False)
                print('load ECG SSL model')
                print(msg)
            elif args.ecg_pretrained == 'Align':
                print("load Align ecg_model")
                ecg_checkpoint = torch.load(args.ecg_alignment_model, map_location='cpu')
                ecg_checkpoint_model = ecg_checkpoint['model']
                ecg_encoder_keys = {k: v for k, v in ecg_checkpoint_model.items() if
                                    k.startswith('ECG_encoder')}
                # remove ECG_encoder.head
                ecg_encoder_keys = {k: v for k, v in ecg_encoder_keys.items() if not k.startswith('ECG_encoder.head')}
                ecg_checkpoint_model = {k.replace('ECG_encoder.', ''): v for k, v in ecg_encoder_keys.items()}
                msg = model.load_state_dict(ecg_checkpoint_model, strict=False)
                print('load aligned ecg model')
                print(msg)
            else:
                print(f'ecg_pretrained error')
                exit()
        else:
            print(f'no pretrained model or alignment model')
    elif args.signal == 'ppg':
        model = SingnalEncoder.__dict__[args.ppg_model](
                img_size=args.ppg_input_size,
                patch_size=args.ppg_patch_size,
                in_chans=args.ppg_input_channels,
                num_classes=args.latent_dim,
                drop_rate=args.ppg_drop_out,
                args=args)
        if args.ppg_pretrained:
            if args.ppg_pretrained == 'SSL':
                print("load pretrained ppg_model")
                ppg_checkpoint = torch.load(args.ppg_pretrained_model, map_location='cpu')
                ppg_checkpoint_model = ppg_checkpoint['model']
                msg = model.load_state_dict(ppg_checkpoint_model, strict=False)
                print('load PPG SSL model')
                print(msg)
            elif args.ppg_pretrained == 'Align':
                print("load Align ppg_model")
                ppg_checkpoint = torch.load(args.ppg_alignment_model, map_location='cpu')
                ppg_checkpoint_model = ppg_checkpoint['model']
                ppg_encoder_keys = {k: v for k, v in ppg_checkpoint_model.items() if
                                    k.startswith('PPG_encoder')}
                # remove PPG_encoder.head
                ppg_encoder_keys = {k: v for k, v in ppg_encoder_keys.items() if not k.startswith('PPG_encoder.head')}
                ppg_checkpoint_model = {k.replace('PPG_encoder.', ''): v for k, v in ppg_encoder_keys.items()}
                msg = model.load_state_dict(ppg_checkpoint_model, strict=False)
                print('load aligned ppg model')
                print(msg)
            else:
                print(f'ppg_pretrained error')
                exit()
        else:
            print(f'no pretrained model or alignment model')
    elif args.signal == 'both':
        model = BothEncoder(args=args)
    elif args.signal == 'enhancedppg':
        model = EnhancedEncoder(args=args)
    else:
        print(f'signal error')
        exit()
    



    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        shuffle=False,
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
    eval_criterion = args.criterion
    best_stats = {'loss':np.inf, eval_criterion: -np.inf}
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

        val_stats,val_result = evaluate(data_loader_val, model, device, epoch, log_writer=log_writer, args=args)
        

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
                # profiler=Profiler()
                # profiler.start()
                
                misc.save_best_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, test_stats=val_stats, evaluation_criterion=eval_criterion)
                

        best_stats['loss'] = min(best_stats['loss'], val_stats['loss'])
        best_stats[eval_criterion] = max(best_stats[eval_criterion], val_stats[eval_criterion])
            
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        # profiler.stop()
        # profiler.print()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
