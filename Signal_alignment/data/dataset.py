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

import pickle


class Processed_Signal_dataset(data.Dataset):
    def __init__(self, ppg_path, ecg_path, args=None):
        self.ppg = np.load(ppg_path)['signal_data']
        self.ecg = np.load(ecg_path)['signal_data']
        
        if 'train' in ppg_path:
            print('training set')
            ppg_quality = pickle.load(open("/mnt/data2/PPG/dataset_all/processed_data/ppg_data_train_quality_indices.pkl", "rb"))
            ecg_quality = pickle.load(open("/mnt/data2/PPG/dataset_all/processed_data/ecg_data_train_quality_indices.pkl", "rb"))

            ppg_acc = ppg_quality['Acc']
            ecg_acc = ecg_quality['Excellent_index']
            ecg_acc.extend(ecg_quality['Barely_index'])

            acc_index = list(set(ppg_acc).intersection(set(ecg_acc)))
            
            self.ppg = self.ppg[acc_index]
            self.ecg = self.ecg[acc_index]

        elif 'test' in ppg_path:
            print('testing set')
            ppg_quality = pickle.load(open("/mnt/data2/PPG/dataset_all/processed_data/ppg_data_test_quality_indices.pkl", "rb"))
            ecg_quality = pickle.load(open("/mnt/data2/PPG/dataset_all/processed_data/ecg_data_test_quality_indices.pkl", "rb"))

            ppg_acc = ppg_quality['Acc']
            ecg_acc = ecg_quality['Excellent_index']
            ecg_acc.extend(ecg_quality['Barely_index'])
            acc_index = list(set(ppg_acc).intersection(set(ecg_acc)))
            
            self.ppg = self.ppg[acc_index]
            self.ecg = self.ecg[acc_index]

    def __len__(self):
        return self.ppg.shape[0]

    def __getitem__(self, index):
        ppg = self.ppg[index]
        ecg = self.ecg[index]
        return torch.tensor(ppg, dtype=torch.float32), torch.tensor(ecg, dtype=torch.float32)