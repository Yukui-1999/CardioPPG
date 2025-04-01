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
import torch
import util.misc as misc
import util.lr_sched as lr_sched
import matplotlib.pyplot as plt


def norm(data:torch.Tensor()) -> torch.Tensor():
    """
    Zero-Normalize data to have mean=0 and standard_deviation=1

    Parameters
    ----------
    data:  tensor
    """
    mean = torch.mean(data, dim=-1, keepdim=True)
    var = torch.var(data, dim=-1, keepdim=True)

    return (data - mean) / (var + 1e-12)**0.5

def ncc(data_0:torch.Tensor(), data_1:torch.Tensor()) -> torch.Tensor():
    """
    Zero-Normalized cross-correlation coefficient between two data sets

    Zero-Normalized cross-correlation equals the cosine of the angle between the unit vectors F and T, 
    being thus 1 if and only if F equals T multiplied by a positive scalar. 

    Parameters
    ----------
    data_0, data_1 :  tensors of same size
    """

    nb_of_signals = 1
    for dim in range(data_0.dim()-1): # all but the last dimension (which is the actual signal)
        nb_of_signals = nb_of_signals * data_0.shape[dim]

    cross_corrs = (1.0 / (data_0.shape[-1]-1)) * torch.sum(norm(data=data_0) * norm(data=data_1), dim=-1)

    return (cross_corrs.sum() / nb_of_signals)

def plot_comparison(real_signal, rec_signal, mask_signal):
    plt.figure(figsize=(12, 6))

    # 绘制真实信号
    plt.subplot(3, 1, 1)  # Adjusted to 3x1 grid
    plt.plot(real_signal.cpu().numpy(), label="Real Signal", color='blue')
    plt.title('Real Signal')
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # 绘制生成信号
    plt.subplot(3, 1, 2)  # Adjusted to 3x1 grid
    plt.plot(rec_signal.cpu().numpy(), label="Rec Signal", color='orange')
    plt.title('Rec Signal')
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # 绘制mask信号
    plt.subplot(3, 1, 3)  # Adjusted to 3x1 grid
    plt.plot(mask_signal.cpu().numpy(), label="Mask Signal", color='red')
    plt.title('Mask Signal')  # Corrected title
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)

    fig = plt.gcf()
    return fig

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
        training_history = {}
    # print(next(model.parameters()).dtype)
    for data_iter_step, signal in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        # signal = batch_data[args.signal]
        signal = signal.to(device, dtype=torch.float32)
        signal = signal.unsqueeze(1).unsqueeze(1)
        # print(f'signal shape: {signal.shape}')
        with torch.cuda.amp.autocast():
            loss, samples_hat, samples_hat_masked = model(signal, mask_ratio=args.mask_ratio)
        
        loss_value = loss.item()
        # print(loss_value)
        
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

        batch_size = signal.shape[0]
        normalized_corr = ncc(signal, samples_hat).item()
        metric_logger.meters['ncc'].update(normalized_corr, n=batch_size)

        if args.debug and data_iter_step > 5:
            break
        #loss_value_reduce = misc.all_reduce_mean(loss_value)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if log_writer is not None:
        log_writer.add_scalar('train/train_loss', train_stats["loss"], epoch)
        log_writer.add_scalar('lr', train_stats["lr"], epoch)
        log_writer.add_scalar('train/normalized_corr_coef', train_stats["ncc"], epoch)

    return train_stats


@torch.no_grad()
def evaluate(data_loader, model, device, epoch, log_writer=None, args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # switch to evaluation mode
    model.eval()

    for data_iter_step, signal in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # signal = batch_data[args.signal]
        signal = signal.to(device, dtype=torch.float32)
        signal = signal.unsqueeze(1).unsqueeze(1)

        # compute output
        with torch.cuda.amp.autocast():
            loss, samples_hat, samples_hat_masked = model(signal, mask_ratio=args.mask_ratio)

        loss_value = loss.item()
        # batch_size = samples.shape[0]
        metric_logger.update(loss=loss_value)

        batch_size = signal.shape[0]
        normalized_corr = ncc(signal, samples_hat).item()
        metric_logger.meters['ncc'].update(normalized_corr, n=batch_size)
        if args.debug and data_iter_step > 5:
            break
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged validation stats:", metric_logger)

    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # log evaluation results
    if log_writer is not None:
        log_writer.add_scalar('val/val_loss', test_stats["loss"], epoch)
        log_writer.add_scalar('val/val_normalized_corr_coef', test_stats["ncc"], epoch)

        num_samples = min(20, signal.shape[0])
        for idx in range(num_samples):
            fig = plot_comparison(signal[idx, 0, 0, :], samples_hat[idx, 0, 0, :], samples_hat_masked[idx, 0, 0, :])
            log_writer.add_figure(f'val/val_samples_{idx}', fig, epoch)

    return test_stats