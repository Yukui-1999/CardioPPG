import os
import sys
from typing import Any, Tuple

import numpy as np
from bs4 import BeautifulSoup
from statsmodels.tsa.seasonal import seasonal_decompose
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

import util.transformations as transformations
import util.augmentations as augmentations
class ECGDataset(Dataset):
    """Fast ECGDataset (fetching prepared data and labels from files)"""
    def __init__(self, data_path, labels_path=None, labels_mask_path=None, downstream_task=None, 
                 transform=False, augment=False, args=None) -> None:
        """load data and labels from files"""
        self.downstream_task = downstream_task
        self.transform = transform
        self.augment = augment
        self.args = args
        self.data = torch.load(data_path, map_location=torch.device('cpu')).float() # load to ram
        print(self.data.shape)

        # valid_indices = []
        # for i in range(self.data.size(0)):
        #     sub_tensor = self.data[i]  # 获取第 i 个子张量
        #     flag = False
        #     for channel in range(sub_tensor.shape[1]):
        #         unique_values = torch.unique(sub_tensor[0, channel, :])
        #         if len(unique_values) == 1:
        #             print(f"Channel {channel} of sub_tensor {i} has all values the same.")
        #             flag = True
        #     if flag:  # 如果子张量中有任何 NaN 值
        #         continue  # 跳过当前的迭代
        #     valid_indices.append(i)

        # # 使用有效的索引来更新原张量
        # self.data = self.data[valid_indices]
        # torch.save(self.data, "ecg_data_array_train_v2.pt")
        # print(self.data.shape)
        if labels_path:
            self.labels = torch.load(labels_path, map_location=torch.device('cpu'))#[..., None] # load to ram
        else:
            self.labels = torch.zeros(size=(len(self.data), ))

        if labels_mask_path:
            self.labels_mask = torch.load(labels_mask_path, map_location=torch.device('cpu')) # load to ram
        else:
            self.labels_mask = torch.ones_like(self.labels)

    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """return a sample from the dataset at index idx"""

        if self.downstream_task == 'regression':
            data, label, label_mask = self.data[idx], self.labels[idx][..., self.args.lower_bnd:self.args.upper_bnd], self.labels_mask[idx][..., self.args.lower_bnd:self.args.upper_bnd]
        else:
            data, label, label_mask = self.data[idx], self.labels[idx], self.labels_mask[idx]
        
        # if self.args.input_size[0] == 1:
        #     data = data.unsqueeze(dim=0)
        
        if self.transform == True:
            transform = transforms.Compose([
                augmentations.CropResizing(lower_bnd=1,upper_bnd=1,fixed_crop_len=None, start_idx=0, resize=True),
                transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise")
            ])
            data = transform(data)

        if self.augment == True:
            augment = transforms.Compose([
                augmentations.CropResizing(fixed_crop_len=self.args.input_size[-1], resize=False),
                augmentations.TimeFlip(prob=0.33),
                augmentations.SignFlip(prob=0.33),
                transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
            ])
            data = augment(data)
            
        if self.downstream_task == 'classification':
            label = label.type(torch.LongTensor).argmax(dim=-1)
            label_mask = torch.ones_like(label)
        
        # has_nan = torch.isnan(data).any()
        # print('Does data have any NaN values?', has_nan)
        
        return data, label, label_mask
# dataset = ECGDataset(data_path='/home/dingzhengyao/Work/ECG_CMR/mae/mae-ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/ecg_data_array_train.pt')

class processECGDATA(Dataset):
    """Fast ECGDataset (fetching prepared data and labels from files)"""
    def __init__(self, data_path, labels_path=None, labels_mask_path=None, downstream_task=None, 
                 transform=False, augment=False, args=None) -> None:
        """load data and labels from files"""
        self.downstream_task = downstream_task
        
        self.transform = transform
        self.augment = augment
        
        self.args = args

        csv_file = pd.read_csv(data_path)
        _1 = csv_file['20205_2_0'].values
        _2 = csv_file['20205_3_0'].dropna().values
        
        self.data = np.concatenate((_1, _2), axis=0)
        self.ecg_data = []
        self.ecg_data_save = {}
        for i in range(len(self.data)):
            ecg_path = os.path.join("/mnt/data/ukb_heartmri/ukb_20205", self.data[i])
            print(f'ecg_path: {ecg_path}')
            ecgdata = self.get_ecg(ecg_path)
            if ecgdata is not None:
                self.ecg_data.append(ecgdata)
                self.ecg_data_save[self.data[i].replace(".xml","")] = ecgdata
            else:
                self.data[i] = None
        torch.save(self.ecg_data_save, "ecg_data_test.pt")
        self.ecg_data = np.array(self.ecg_data)
        print(self.ecg_data.shape)
        

        if labels_path:
            self.labels = torch.load(labels_path, map_location=torch.device('cpu'))#[..., None] # load to ram
        else:
            self.labels = torch.zeros(size=(len(self.data), ))

        if labels_mask_path:
            self.labels_mask = torch.load(labels_mask_path, map_location=torch.device('cpu')) # load to ram
        else:
            self.labels_mask = torch.ones_like(self.labels)

    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        return len(self.data)
    def get_ecg(self, ecg_path):
        ecg_file = open(ecg_path).read()
        bs = None
        try:
            bs = BeautifulSoup(ecg_file, features="lxml")
        except:
            pass
        ecg_waveform_length = 5000
        if ecg_waveform_length == 600:
            waveform = bs.body.cardiologyxml.mediansamples
        else:
            waveform = bs.body.cardiologyxml.stripdata
        # print(waveform)
        # print(type(waveform))
        data_numpy = None
        bs_measurement = bs.body.cardiologyxml.restingecgmeasurements
        heartbeat = int(bs_measurement.find_all("VentricularRate".lower())[0].string)

        for each_wave in waveform.find_all("waveformdata"):
            each_data = each_wave.string.strip().split(",")
            each_data = [s.replace('\n\t\t', '') for s in each_data]
            each_data = np.array(each_data, dtype=np.float32)
            # plt.plot(each_data)
            try:
                seasonal_decompose_result = seasonal_decompose(each_data, model="additive",
                                                            period=int(ecg_waveform_length*6/heartbeat))
            except:
                return None
            trend = seasonal_decompose_result.trend
            start, end = 0, ecg_waveform_length - 1
            sflag, eflag = False, False
            for i in range(ecg_waveform_length):
                if np.isnan(trend[i]):
                    start += 1
                else:
                    sflag = True
                if np.isnan(trend[ecg_waveform_length-1-i]):
                    end -= 1
                else:
                    eflag = True
                if sflag and eflag:
                    break
            trend[:start] = trend[start]
            trend[end:] = trend[end]
            # trend[np.isnan(trend)] = 0.0
            result = np.array(seasonal_decompose_result.observed - trend)
            # plt.plot(result)
            # plt.show()
            # exit()
            if data_numpy is None:
                data_numpy = result
            else:
                data_numpy = np.vstack((data_numpy, result))

        return data_numpy

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """return a sample from the dataset at index idx"""

        if self.downstream_task == 'regression':
            data, label, label_mask = self.data[idx], self.labels[idx][..., self.args.lower_bnd:self.args.upper_bnd], self.labels_mask[idx][..., self.args.lower_bnd:self.args.upper_bnd]
        else:
            data, label, label_mask = self.data[idx], self.labels[idx], self.labels_mask[idx]
        
        if self.args.input_size[0] == 1:
            data = data.unsqueeze(dim=0)
        
        if self.transform == True:
            transform = transforms.Compose([
                augmentations.CropResizing(lower_bnd=1,upper_bnd=1,fixed_crop_len=None, start_idx=0, resize=True),
                # transformations.PowerSpectralDensity(fs=1000, nperseg=1000, return_onesided=False),
                # transformations.Normalization(mode="group_wise",groups=[[0, 1, 2], [3, 4, 5], [6, 7, 8, 9, 10, 11]]),
                transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise")
            ])
            data = transform(data)

        if self.augment == True:
            augment = transforms.Compose([
                augmentations.CropResizing(fixed_crop_len=self.args.input_size[-1], resize=False),
                # transformations.PowerSpectralDensity(fs=100, nperseg=1000, return_onesided=False),
                
                # augmentations.FTSurrogate(phase_noise_magnitude=ft_surr_phase_noise, prob=0.5),
                # augmentations.Jitter(sigma=jitter_sigma),
                # augmentations.Rescaling(sigma=rescaling_sigma),
                augmentations.TimeFlip(prob=0.33),
                augmentations.SignFlip(prob=0.33),
                transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
            ])



# dataset = ECGDataset(data_path='/mnt/data/ukb_collation/ukb_ecg_cmr/data/test.csv')
class SignalDataset(Dataset):
    """Fast EEGDataset (fetching prepared data and labels from files)"""
    def __init__(self, data_path, labels_path=None, labels_mask_path=None, downstream_task=None, 
                 transform=False, augment=False, args=None) -> None:
        """load data and labels from files"""
        self.downstream_task = downstream_task
        
        self.transform = transform
        self.augment = augment
        
        self.args = args

        self.data = torch.load(data_path, map_location=torch.device('cpu')) # load to ram

        if labels_path:
            self.labels = torch.load(labels_path, map_location=torch.device('cpu'))#[..., None] # load to ram
        else:
            self.labels = torch.zeros(size=(len(self.data), ))

        if labels_mask_path:
            self.labels_mask = torch.load(labels_mask_path, map_location=torch.device('cpu')) # load to ram
        else:
            self.labels_mask = torch.ones_like(self.labels)

    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """return a sample from the dataset at index idx"""

        if self.downstream_task == 'regression':
            data, label, label_mask = self.data[idx], self.labels[idx][..., self.args.lower_bnd:self.args.upper_bnd], self.labels_mask[idx][..., self.args.lower_bnd:self.args.upper_bnd]
        else:
            data, label, label_mask = self.data[idx], self.labels[idx], self.labels_mask[idx]
        
        if self.args.input_size[0] == 1:
            data = data.unsqueeze(dim=0)

        data = data[:, :self.args.input_electrodes, :]
        
        if self.transform == True:
            transform = transforms.Compose([
                augmentations.CropResizing(fixed_crop_len=self.args.input_size[-1], start_idx=0, resize=False),
                # transformations.PowerSpectralDensity(fs=100, nperseg=1000, return_onesided=False),
                # transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise")
            ])
            data = transform(data)

        if self.augment == True:
            augment = transforms.Compose([
                augmentations.CropResizing(fixed_crop_len=self.args.input_size[-1], resize=False),
                # transformations.PowerSpectralDensity(fs=100, nperseg=1000, return_onesided=False),
                # transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
                augmentations.FTSurrogate(phase_noise_magnitude=self.args.ft_surr_phase_noise, prob=0.5),
                augmentations.Jitter(sigma=self.args.jitter_sigma),
                augmentations.Rescaling(sigma=self.args.rescaling_sigma),
                # augmentations.TimeFlip(prob=0.33),
                # augmentations.SignFlip(prob=0.33)
            ])
            data = augment(data)
        
        if self.downstream_task == 'classification':
            label = label.type(torch.LongTensor).argmax(dim=-1)
            label_mask = torch.ones_like(label)

        return data, label, label_mask