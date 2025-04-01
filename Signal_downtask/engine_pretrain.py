# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable
import test
import torch
import util.misc as misc
import util.lr_sched as lr_sched
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_auc_score, confusion_matrix
from scipy.stats import pearsonr

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = args.accum_iter
    optimizer.zero_grad()
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    if args.downtask_type == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.downtask_type == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    elif args.downtask_type == 'Regression':
        loss_fn = nn.SmoothL1Loss()
    
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        if args.signal == 'ppg' or args.signal == 'ecg' or args.signal == 'enhancedppg':
            if args.signal == 'enhancedppg':
                signal = batch['ppg'].to(device, dtype=torch.float32)
                signal = signal.unsqueeze(1).unsqueeze(1)
            else:
                signal = batch[args.signal].to(device, dtype=torch.float32)
                signal = signal.unsqueeze(1).unsqueeze(1)
        elif args.signal == 'both':
            ppg = batch['ppg'].to(device, dtype=torch.float32)
            ecg = batch['ecg'].to(device, dtype=torch.float32)
            ppg = ppg.unsqueeze(1).unsqueeze(1)
            ecg = ecg.unsqueeze(1).unsqueeze(1)
        else:
            raise ValueError('signal type error')
        
        if args.downtask_type == 'BCE' or args.downtask_type == 'Regression':
            target = batch['label'].to(device, dtype=torch.float32).unsqueeze(1)
        elif args.downtask_type == 'CE':
            target = batch['label'].to(device, dtype=torch.long)
        else:
            raise ValueError('downtask type error')
        
        if args.signal == 'ppg' or args.signal == 'ecg' or args.signal == 'enhancedppg':
            with torch.cuda.amp.autocast():
                if args.signal == 'enhancedppg':
                    out = model(signal)
                else:
                    _, out = model(signal)
                loss = loss_fn(out, target)
        elif args.signal == 'both':
            with torch.cuda.amp.autocast():
                out = model(ppg, ecg)
                loss = loss_fn(out, target)
        
        
        loss_value = loss.item()
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

    

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if log_writer is not None:
        log_writer.add_scalar('train/train_loss', train_stats["loss"], epoch)
        log_writer.add_scalar('lr', train_stats["lr"], epoch)

    return train_stats


@torch.no_grad()
def evaluate(data_loader, model, device, epoch, log_writer=None, args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    if args.downtask_type == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.downtask_type == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    elif args.downtask_type == 'Regression':
        loss_fn = nn.SmoothL1Loss()
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # switch to evaluation mode
    model.eval()
    pred = []
    real = []
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if args.signal == 'ppg' or args.signal == 'ecg' or args.signal == 'enhancedppg':
            if args.signal == 'enhancedppg':
                signal = batch['ppg'].to(device, dtype=torch.float32)
                signal = signal.unsqueeze(1).unsqueeze(1)
            else:
                signal = batch[args.signal].to(device, dtype=torch.float32)
                signal = signal.unsqueeze(1).unsqueeze(1)
        elif args.signal == 'both':
            ppg = batch['ppg'].to(device, dtype=torch.float32)
            ecg = batch['ecg'].to(device, dtype=torch.float32)
            ppg = ppg.unsqueeze(1).unsqueeze(1)
            ecg = ecg.unsqueeze(1).unsqueeze(1)
        else:
            raise ValueError('signal type error')
        
        if args.downtask_type == 'BCE' or args.downtask_type == 'Regression':
            target = batch['label'].to(device, dtype=torch.float32).unsqueeze(1)
        elif args.downtask_type == 'CE':
            target = batch['label'].to(device, dtype=torch.long)
        else:
            raise ValueError('downtask type error')
        
        if args.signal == 'ppg' or args.signal == 'ecg' or args.signal == 'enhancedppg':
            with torch.cuda.amp.autocast():
                if args.signal == 'enhancedppg':
                    out = model(signal)
                else:
                    _, out = model(signal)
                
                loss = loss_fn(out, target)
                pred.append(out)
                real.append(target)
        elif args.signal == 'both':
            with torch.cuda.amp.autocast():
                out = model(ppg, ecg)
                loss = loss_fn(out, target)
                pred.append(out)
                real.append(target)
        loss_value = loss.item()
        # batch_size = samples.shape[0]
        metric_logger.update(loss=loss_value)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged validation stats:", metric_logger)
    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}


    pred = torch.cat(pred, dim=0)
    real = torch.cat(real, dim=0)
    if args.downtask_type == 'BCE':
        pred = torch.sigmoid(pred)
        pred = pred.cpu().numpy()
        real = real.cpu().numpy()
        # calculate auc
        AUC = roc_auc_score(real, pred)
        test_stats['AUC'] = AUC
    elif args.downtask_type == 'CE':
        pred = torch.softmax(pred, dim=1)
        pred = pred.cpu().numpy()
        num_classes = pred.shape[1] 
        real_one_hot = torch.nn.functional.one_hot(real, num_classes=num_classes)
        real = real_one_hot.cpu().numpy()
        # calculate auc
        AUC = roc_auc_score(real, pred, multi_class='ovr')
        test_stats['AUC'] = AUC
    elif args.downtask_type == 'Regression':
        pred = pred.cpu().numpy().reshape(-1)
        real = real.cpu().numpy().reshape(-1)
        
        # calculate Pearsonr
        corr, p = pearsonr(pred, real)
        test_stats['Pearsonr'] = corr
        test_stats['Pearsonr_p'] = p
    else:
        raise ValueError('downtask type error')

    # log evaluation results
    if log_writer is not None:
        log_writer.add_scalar('val/val_loss', test_stats["loss"], epoch)
        if args.downtask_type == 'BCE' or args.downtask_type == 'CE':
            log_writer.add_scalar('val/AUC', test_stats["AUC"], epoch)
            print(f'epoch: {epoch}, AUC: {test_stats["AUC"]}')
        elif args.downtask_type == 'Regression':
            log_writer.add_scalar('val/Pearsonr', test_stats["Pearsonr"], epoch)
            print(f'epoch: {epoch}, Pearsonr: {test_stats["Pearsonr"]}')

    return test_stats,{
        'y_pred': pred,
        'y_true': real
    }