import torch.utils.data as data
import os
import torch
import numpy as np
from typing import Any, Tuple
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import pickle
import neurokit2 as nk
import pandas as pd
import pickle
from torch.utils.data import DataLoader, Dataset
import random
from pyinstrument import Profiler

def process_ppg(ppg):
    ppg = nk.ppg_clean(ppg, sampling_rate=256)
    # normalize
    ppg = 2 * (ppg - np.min(ppg)) / (np.max(ppg) - np.min(ppg)) -1
    return ppg


def process_ecg(ecg):
    ecg, is_inverted = nk.ecg_invert(ecg, sampling_rate=256)
    ecg = nk.ecg_clean(ecg, sampling_rate=256, method="pantompkins1985")
    # normalize
    ecg = 2 * (ecg - np.min(ecg)) / (np.max(ecg) - np.min(ecg)) -1
    return ecg

class Processed_Signal_dataset(data.Dataset):
    def __init__(self, data_index, args=None):
        self.data = np.load(data_index)['signal_data']
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        signal = self.data[index]
        return torch.tensor(signal, dtype=torch.float32)
    


class Single_Signal_dataset(data.Dataset):
    def __init__(self, data_index,args=None):
        self.data_index = pd.read_csv(data_index)
        self.args = args
        if self.args.miniset:
            self.data_index = self.data_index.iloc[:len(self.data_index)//100]
        print(f'length of dataset: {len(self.data_index)}')
    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, index):
        sample = self.data_index.iloc[index]
        file_path = sample['file_path']
        subject = sample['subject']
        segment = sample['segment']
        index_in_dataset = sample['index']
        batch_info = {'subject': subject, 'segment': segment, 'index': index_in_dataset}
        with open(file_path, 'rb') as f:
            file = pickle.load(f)
        ecg = file['ecg']
        ppg = file['ppg']
        ecg = process_ecg(ecg)
        ppg = process_ppg(ppg)
        return {'ecg': torch.tensor(ecg, dtype=torch.float32), 'ppg': torch.tensor(ppg, dtype=torch.float32)}, batch_info

class Signal_dataset(data.Dataset):
    def __init__(self, data_index: list, time_window: int):
        self.data_index = data_index
        self.time_window = time_window

    def __len__(self) -> int:
        return len(self.data_index)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # self.Profiler.start()
        sample = self.data_index[index]
        subject = sample['subject']
        segment = sample['segment']
        files_path = sample['file_path']
        index_in_dataset = sample['index']
        batch_info = {'subject': subject, 'segment': segment, 'index': index_in_dataset}
        file_list = []
        for file_path in files_path:
            # replace the path in sda1
            file_path = file_path.replace('/mnt/data2/PPG/dataset_all', '/mnt/sda1/dingzhengyao/Work/PPG_ECG')
            with open(file_path, 'rb') as f:
                file_list.append(pickle.load(f))
        ecg_list = []
        ppg_list = []
        for file in file_list:
            ecg = file['ecg']
            ppg = file['ppg']
            ecg_list.append(ecg)
            ppg_list.append(ppg)
        ecg_data = np.hstack(ecg_list)
        ppg_data = np.hstack(ppg_list)
        ecg_data = process_ecg(ecg_data)
        ppg_data = process_ppg(ppg_data)
        batch_data = {
            'ecg': torch.tensor(ecg_data, dtype=torch.float32),
            'ppg': torch.tensor(ppg_data, dtype=torch.float32)
        }
        # self.Profiler.stop()
        # self.Profiler.print()
        return batch_data, batch_info

class Pretraining_Dataset_All:
    def __init__(self, data_index_path: str,miniset: bool):
        self.data_index_path = data_index_path
        data_index_all = {
            "10s":[],
            "20s":[],
            "30s":[],
            "40s":[],
            "50s":[],
            "60s":[],
            # ...
        }
        with open(data_index_path, 'rb') as f:
            data_index = pickle.load(f)
        for obj in data_index:
            time_window = str(len(obj['segment'])) + '0s'
            data_index_all[time_window].append(obj)
        if miniset:
            for key in data_index_all.keys():
                # set lenth is 1/100 of the original dataset
                data_index_all[key] = data_index_all[key][:len(data_index_all[key])//100]

        for key in data_index_all.keys():
            print(f"{key}: {len(data_index_all[key])}")
        self.datasets = []
        for key in data_index_all.keys():
            self.datasets.append(Signal_dataset(data_index=data_index_all[key], time_window=int(key[:-1])))

    def get_datasets(self):
        return self.datasets

class MultiDatasetLoader:
    def __init__(self, datasets, batch_size, shuffle=True, num_workers=0):
        self.dataloaders = [
            DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
            for dataset in datasets
        ]
        self.dataset_lengths = [len(dataset) for dataset in datasets]  # 每个数据集的长度
        self.dataset_iterators = [iter(dl) for dl in self.dataloaders]  # 每个 DataLoader 的迭代器
        self.samples_seen = [0] * len(datasets)  # 每个数据集已遍历的样本数
        self.total_samples = sum(self.dataset_lengths)  # 总样本数
        self.samples_processed = 0  # 已处理的总样本数

    def __iter__(self):
        return self

    def __len__(self):
        return sum(len(dl) for dl in self.dataloaders)

    def __next__(self):

        # 如果所有数据集都已遍历完成，停止迭代
        if self.samples_processed >= self.total_samples:

            raise StopIteration

        # 随机选择一个未遍历完的数据集
        available_indices = [i for i, seen in enumerate(self.samples_seen) if seen < self.dataset_lengths[i]]
        if not available_indices:

            raise StopIteration  # 再次检查，避免意外情况

        dataset_idx = random.choice(available_indices)  # 随机选择一个未完成的数据集
        try:
            # 获取下一个 batch
            batch = next(self.dataset_iterators[dataset_idx])
        except StopIteration:
            # 如果该 DataLoader 数据集遍历完，则跳过
            raise StopIteration

        # 更新计数器
        self.samples_seen[dataset_idx] += len(batch)
        self.samples_processed += len(batch)

        return batch


    def reset(self):
        """
        重置计数器和迭代器，用于新 epoch 的开始
        """
        self.dataset_iterators = [iter(dl) for dl in self.dataloaders]  # 重置每个 DataLoader 的迭代器
        self.samples_seen = [0] * len(self.dataloaders)  # 清零每个数据集的样本计数
        self.samples_processed = 0  # 清零总样本计数

if '__main__' == __name__:
    dataset = Pretraining_Dataset_All(data_index_path='/mnt/data2/PPG/dataset_all/processed_data/pretrained_train.pkl')
    mutiloader = MultiDatasetLoader(datasets=dataset.datasets, batch_size=32, shuffle=True, num_workers=8)
    for i, (batch_data, batch_info) in enumerate(mutiloader):
        print(f'batch_data: {batch_data["ppg"].shape}, batch_info: {batch_info}')
        if i == 10:
            break



