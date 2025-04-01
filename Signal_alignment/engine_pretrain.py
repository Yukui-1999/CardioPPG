import math
import sys
from typing import Iterable
import torch
import utils.misc as misc
import utils.lr_sched as lr_sched


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
    training_history = {}
    optimizer.zero_grad()
    

    for data_iter_step, (ppg,ecg) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        ecg = ecg.float().unsqueeze(1).unsqueeze(1).to(device)
        ppg = ppg.float().unsqueeze(1).unsqueeze(1).to(device)
        # print(f'ecg shape:{ecg.shape},ppg shape:{ppg.shape}')
        with torch.cuda.amp.autocast():
            loss,_,_ = model(ecg,ppg)
            loss_value = loss.item()
            total_loss = loss

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        total_loss /= accum_iter
        loss_scaler(total_loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

    print("stats:", metric_logger)
    print(f'current device : {torch.cuda.current_device()}')

    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if log_writer is not None:
        log_writer.add_scalar('train/train_loss', train_stats["loss"], epoch)
        log_writer.add_scalar('lr', train_stats["lr"], epoch)
    
    return train_stats, training_history

@torch.no_grad()
def evaluate(data_loader, model, device, epoch,log_writer=None, args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    test_history = {}  
    model.eval()
    for batch in metric_logger.log_every(data_loader, 20, header):
        
        ppg = batch[0].float().unsqueeze(1).unsqueeze(1).to(device)
        ecg = batch[1].float().unsqueeze(1).unsqueeze(1).to(device)
        
        
        with torch.cuda.amp.autocast():
            loss,_,_ = model(ecg,ppg)
            loss_value = loss.item()

        metric_logger.update(loss=loss_value)

    print("validation stats:", metric_logger)
    print(f'current device : {torch.cuda.current_device()}')
    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if log_writer is not None:
        log_writer.add_scalar('val/val_loss', test_stats["loss"], epoch)
    return test_stats, test_history