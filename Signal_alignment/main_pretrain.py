import argparse
import datetime
import json
from typing import Tuple
import numpy as np
import os
import time
from pathlib import Path

import sys
import torch
from torch.utils.data import Subset, ConcatDataset
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model.CEPP import CEPP
# sys.path.append("..")
from data.dataset import Processed_Signal_dataset
import timm.optim.optim_factory as optim_factory
from torch.utils.tensorboard import SummaryWriter
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.callbacks import EarlyStop
from engine_pretrain import train_one_epoch, evaluate


# from engine_pretrain import train_one_epoch, evaluate

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
    parser.add_argument('--latent_dim', default=128, type=int, metavar='N',help='latent_dim')
    parser.add_argument('--model', default=None, type=str)
    parser.add_argument('--freezeECG', default=False, type=str2bool, help='freezeECG')
    # ECG Model parameters
    parser.add_argument('--ecg_pretrained', default=True, type=str2bool,help='ecg_pretrained or not')
    parser.add_argument('--ecg_model', default='vit_base_patchX', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--ecg_pretrained_model',default="/mnt/data2/PPG/Work/P2E_v2/Signal_SSL/mae_vit_base_patchX/1.1/ep100_WARM_EP20_lr1e-6_bs4096_wd1e-4_sgecg/checkpoint-55-ncc-0.81.pth",
                        type=str, metavar='MODEL', help='path of pretaained model')
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
    parser.add_argument('--ppg_pretrained', default=True, type=str2bool,help='ppg_pretrained or not')
    parser.add_argument('--ppg_model', default='vit_base_patchX', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--ppg_pretrained_model',
                        default="/mnt/data2/PPG/Work/P2E_v2/Signal_SSL/mae_vit_base_patchX/1.1/ep100_WARM_EP20_lr1e-6_bs4096_wd1e-4_sgppg/checkpoint-99-ncc-0.94.pth",
                        type=str, metavar='MODEL', help='path of pretaained model')
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
    parser.add_argument('--max_delta', default=0.015, type=float,
                        help='Early stopping threshold (val has to be worse than (train+delta)) (default: 0)')

    # Dataset parameters

    parser.add_argument('--data_path',
                        default="/mnt/sda1/dingzhengyao/Work/PPG_ECG/processed_data/ppg_data_train.npz",
                        type=str,
                        help='dataset path')
    parser.add_argument('--val_data_path',
                        default="/mnt/sda1/dingzhengyao/Work/PPG_ECG/processed_data/ppg_data_test.npz",
                        type=str,
                        help='validation dataset path')
    

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
    device = torch.device(args.device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.ecg_model = args.model
    args.ppg_model = args.model

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # load data
    dataset_train = Processed_Signal_dataset(ppg_path=args.data_path, ecg_path=args.data_path.replace('ppg', 'ecg'),args=args)
    dataset_val = Processed_Signal_dataset(ppg_path=args.val_data_path, ecg_path=args.val_data_path.replace('ppg', 'ecg'),args=args)
    
    print("Training set size: ", len(dataset_train))
    print("Validation set size: ", len(dataset_val))
    
    model = CEPP(global_pool=args.ecg_globle_pool, device=device, args=args)
    model.to(device)
    # print(model)
    
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

    print(f'model device:{next(model.parameters()).device}')
    # state_dict = model.state_dict()
    # for name, param in state_dict.items():
    #     print(f'Parameter name: {name}')
    
    if args.freezeECG:
        for name, param in model.ECG_encoder.named_parameters():
            if not name.startswith('head'):
                param.requires_grad = False
        print(f'freeze ECG encoder')

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of params (M): %.2f' % (n_parameters / 1.e6))
    eff_batch_size = args.batch_size * args.accum_iter

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 4

    print("base lr: %.2e" % (args.lr * 4 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)

    # Define callbacks
    early_stop = EarlyStop(patience=args.patience, max_delta=args.max_delta)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    eval_criterion = "loss"
    best_stats = {'loss': np.inf}
   
    for epoch in range(args.start_epoch, args.epochs):
        
        train_stats, train_history = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        val_stats, test_history = evaluate(data_loader_val, model, device, epoch,log_writer=log_writer, args=args)
        print(f"Loss of the network on the {len(dataset_val)} val dataset: {val_stats['loss']:.4f}")

        if eval_criterion == "loss":
            if early_stop.evaluate_decreasing_metric(val_metric=val_stats[eval_criterion]):
                break
            if args.output_dir and val_stats[eval_criterion] <= best_stats[eval_criterion]:
                misc.save_best_model(
                    args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, test_stats=val_stats, evaluation_criterion=eval_criterion)
        else:
            if early_stop.evaluate_increasing_metric(val_metric=val_stats[eval_criterion]):
                break
            if args.output_dir and val_stats[eval_criterion] >= best_stats[eval_criterion]:
                misc.save_best_model(
                    args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, test_stats=val_stats, evaluation_criterion=eval_criterion)

        best_stats['loss'] = min(best_stats['loss'], val_stats['loss'])

        

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    return 0


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    args.ecg_input_size = (args.ecg_input_electrodes, args.ecg_time_steps)
    args.ppg_input_size = (args.ppg_input_electrodes, args.ppg_time_steps)

    args.ecg_patch_size = (args.ecg_patch_height, args.ecg_patch_width)
    args.ecg_patch_num = (args.ecg_time_steps // args.ecg_patch_width) * (
                args.ecg_input_electrodes // args.ecg_patch_height) + 1
    
    args.ppg_patch_size = (args.ppg_patch_height, args.ppg_patch_width)
    args.ppg_patch_num = (args.ppg_time_steps // args.ppg_patch_width) * (
                args.ppg_input_channels // args.ppg_patch_height) + 1
    

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
