"""
Stage2: prior learning

run `python stage2.py`
"""
import argparse
from argparse import ArgumentParser
from typing import Union
import random

import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from preprocessing.data_pipeline import build_data_pipeline, build_custom_data_pipeline
from preprocessing.preprocess_ucr import DatasetImporterUCR, DatasetImporterCustom
import pandas as pd
import pytorch_lightning as pl
from evaluation.generator import generator
from utils import get_root_dir, load_yaml_param_settings, str2bool


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    parser.add_argument('--dataset_names', nargs='+', help="e.g., Adiac Wafer Crop`.", default='')
    parser.add_argument('--gpu_device_idx', default=0, type=int)
    parser.add_argument('--use_neural_mapper', type=str2bool, default=False, help='Use the neural mapper')
    parser.add_argument('--feature_extractor_type', type=str, default='rocket', help='supervised_fcn | rocket for evaluation.')
    parser.add_argument('--use_custom_dataset', type=str2bool, default=False, help='Using a custom dataset, then set it to True.')
    parser.add_argument('--sampling_batch_size', type=int, default=None, help='batch size when sampling.')
    return parser.parse_args()

def normalize_per_sample(data):
    """
    对每个样本单独进行 [0, 1] 归一化
    :param data: 形状为 (b, length) 的 NumPy 数组
    :return: 归一化后的数据，形状为 (b, length)
    """
    # 计算每个样本的最小值和最大值
    min_vals = np.min(data, axis=1, keepdims=True)
    max_vals = np.max(data, axis=1, keepdims=True)
    
    # 归一化
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    
    return normalized_data

def evaluate(config: dict,
             dataset_name: str,
             gpu_device_idx,
             use_neural_mapper:bool,
             feature_extractor_type:str,
             use_custom_dataset:bool=False,
             sampling_batch_size=None,
             rand_seed:Union[int,None]=None,
             ppg:np.ndarray=None
             ):
    """
    :param do_validate: if True, validation is conducted during training with a test dataset.
    """
    print(f'rand_seed:{rand_seed}')
    if not isinstance(rand_seed, type(None)):
        np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        random.seed(rand_seed)
    ppg = np.expand_dims(ppg, axis=1)

    n_classes = 1 # 无用数值
    samples, in_channels, input_length = ppg.shape
    

    # conditional sampling
    print('generate...')
    evaluation = generator(dataset_name, in_channels, input_length, n_classes, gpu_device_idx, config, 
                            use_neural_mapper=use_neural_mapper,
                            feature_extractor_type=feature_extractor_type,
                            ppg=ppg,
                            use_custom_dataset=use_custom_dataset,
                            dataset_importer=None).to(gpu_device_idx)

    xhat = evaluation.sample(samples, 'conditional', batch_size=sampling_batch_size)
    
    
    
    x_gen = 2 * normalize_per_sample(np.array(xhat).squeeze()) - 1
    return x_gen


if __name__ == '__main__':
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)
    pl.seed_everything(config['seed'])
    
    dataset_names = args.dataset_names
    print('dataset_names:', dataset_names)

    for dataset_name in dataset_names:
        print('dataset_name:', dataset_name)
        ppg = np.load("data/example_ppg.npy")
        print(f'ppg.shape:{ppg.shape}')
        ecg_gen = evaluate(config, dataset_name, args.gpu_device_idx, args.use_neural_mapper, args.feature_extractor_type, args.use_custom_dataset, args.sampling_batch_size,ppg=ppg)
        print(f'ecg_gen.shape:{ecg_gen.shape}')  
        np.save('data/gen_ecg.npy', ecg_gen)     
        # clean memory
        torch.cuda.empty_cache()

