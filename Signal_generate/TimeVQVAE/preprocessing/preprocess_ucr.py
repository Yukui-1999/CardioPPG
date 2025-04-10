"""
`Dataset` (pytorch) class is defined.
"""
from typing import Union
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

from utils import get_root_dir, download_ucr_datasets


class DatasetImporterUCR(object):
    """
    This uses train and test sets as given.
    To compare with the results from ["Unsupervised scalable representation learning for multivariate time series"]
    """
    def __init__(self, dataset_name: str, data_scaling: bool, **kwargs):
        """
        :param dataset_name: e.g., "ElectricDevices"
        :param data_scaling
        """
        download_ucr_datasets()
        # self.data_root = get_root_dir().joinpath("datasets", "UCRArchive_2018", dataset_name)
        self.data_root = get_root_dir().joinpath("datasets", "UCRArchive_2018_resplit", dataset_name)

        # fetch an entire dataset
        df_train = pd.read_csv(self.data_root.joinpath(f"{dataset_name}_TRAIN.tsv"), sep='\t', header=None)
        df_test = pd.read_csv(self.data_root.joinpath(f"{dataset_name}_TEST.tsv"), sep='\t', header=None)

        self.X_train, self.X_test = df_train.iloc[:, 1:].values[:, np.newaxis, :], df_test.iloc[:, 1:].values[:, np.newaxis, :]  # (b 1 l)
        self.Y_train, self.Y_test = df_train.iloc[:, [0]].values[:, np.newaxis, :], df_test.iloc[:, [0]].values[:, np.newaxis, :]  # (b 1 l)

        le = LabelEncoder()
        self.Y_train = le.fit_transform(self.Y_train.ravel())[:, None]
        self.Y_test = le.transform(self.Y_test.ravel())[:, None]

        # if data_scaling:
        #     # following [https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/blob/dcc674541a94ca8a54fbb5503bb75a297a5231cb/ucr.py#L30]
        #     mean = np.nanmean(self.X_train)
        #     var = np.nanvar(self.X_train)
        #     self.X_train = (self.X_train - mean) / math.sqrt(var)
        #     self.X_test = (self.X_test - mean) / math.sqrt(var)
        self.mean, self.std = 1., 1.
        if data_scaling:
            self.mean = np.nanmean(self.X_train, axis=(0, 2))[None,:,None]  # (1 c 1)
            self.std = np.nanstd(self.X_train, axis=(0, 2))[None,:,None]  # (1 c 1)
            self.X_train = (self.X_train - self.mean) / self.std  # (b c l)
            self.X_test = (self.X_test - self.mean) / self.std  # (b c l)

        np.nan_to_num(self.X_train, copy=False)
        np.nan_to_num(self.X_test, copy=False)

        print('self.X_train.shape:', self.X_train.shape)
        print('self.X_test.shape:', self.X_test.shape)

        print("# unique labels (train):", np.unique(self.Y_train.reshape(-1)))
        print("# unique labels (test):", np.unique(self.Y_test.reshape(-1)))


class UCRDataset(Dataset):
    def __init__(self,
                 kind: str,
                 dataset_importer: DatasetImporterUCR,
                 **kwargs):
        """
        :param kind: "train" / "test"
        :param dataset_importer: instance of the `DatasetImporter` class.
        """
        super().__init__()
        self.kind = kind

        if kind == "train":
            self.X, self.Y = dataset_importer.X_train, dataset_importer.Y_train
        elif kind == "test":
            self.X, self.Y = dataset_importer.X_test, dataset_importer.Y_test
        else:
            raise ValueError

        self._len = self.X.shape[0]
    
    def __getitem__(self, idx):
        x, y = self.X[idx, :], self.Y[idx, :]
        return x, y

    def __len__(self):
        return self._len



# class DatasetImporterCustom(object):
#     def __init__(self, data_scaling:bool=True, **kwargs):
#         # training and test datasets
#         # typically, you'd load the data, for example, using pandas
#         self.X_train, self.Y_train = np.random.rand(800, 1, 100), np.random.randint(0, 2, size=(800,1))  # X:(n_training_samples, 1, ts_length); 1 denotes a univariate time series; 2 classes in this example
#         self.X_test, self.Y_test = np.random.rand(200, 1, 100), np.random.randint(0, 2, size=(200,1))  # (n_test_samples, 1, ts_length)

#         if data_scaling:
#             # following [https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/blob/dcc674541a94ca8a54fbb5503bb75a297a5231cb/ucr.py#L30]
#             mean = np.nanmean(self.X_train)
#             var = np.nanvar(self.X_train)
#             self.X_train = (self.X_train - mean) / math.sqrt(var)
#             self.X_test = (self.X_test - mean) / math.sqrt(var)

#         np.nan_to_num(self.X_train, copy=False)
#         np.nan_to_num(self.X_test, copy=False)

import pickle
class DatasetImporterCustom(object):
    def __init__(self, **kwargs):
        print(f'loading custom dataset...')
        # all data v2
        # self.X_train = np.load("/mnt/data2/PPG/dataset_all/processed_data/ecg_data_train_v2.npz")['signal_data']
        # self.X_train = np.expand_dims(self.X_train, axis=1)
        # self.Y_train = np.load("/mnt/data2/PPG/dataset_all/processed_data/ppg_data_train_cond.npy")
        
        # self.X_test = np.load("/mnt/data2/PPG/dataset_all/processed_data/ecg_data_test_v2.npz")['signal_data']
        # self.X_test = np.expand_dims(self.X_test, axis=1)
        # self.Y_test = np.load("/mnt/data2/PPG/dataset_all/processed_data/ppg_data_test_cond.npy")
        
        # all data
        # self.X_train = np.load("/mnt/data2/PPG/dataset_all/processed_data/ecg_data_train.npz")['signal_data']
        # ecg_quality = pickle.load(open("/mnt/data2/PPG/dataset_all/processed_data/ecg_data_train_quality_indices.pkl", "rb"))
        # ecg_acc = ecg_quality['Excellent_index']
        # ecg_acc.extend(ecg_quality['Barely_index'])
        # self.X_train = self.X_train[ecg_acc]
        # self.X_train = np.expand_dims(self.X_train, axis=1)

        # self.X_test = np.load("/mnt/data2/PPG/dataset_all/processed_data/ecg_data_test.npz")['signal_data']
        # ecg_quality = pickle.load(open("/mnt/data2/PPG/dataset_all/processed_data/ecg_data_test_quality_indices.pkl", "rb"))
        # ecg_acc = ecg_quality['Excellent_index']
        # ecg_acc.extend(ecg_quality['Barely_index'])
        # self.X_test = self.X_test[ecg_acc]
        # self.X_test = np.expand_dims(self.X_test, axis=1)
        
        
        # 1/10 data
        self.X_train = np.load("/mnt/data2/PPG/dataset_all/processed_data/mini_data/mini_ecg_train.npy")
        self.X_train = np.expand_dims(self.X_train, axis=1)
        self.X_test = np.load("/mnt/data2/PPG/dataset_all/processed_data/mini_data/mini_ecg_test.npy")
        self.X_test = np.expand_dims(self.X_test, axis=1)
        self.Y_train = np.load("/mnt/data2/PPG/dataset_all/processed_data/mini_data/mini_ppg_train_cond.npy")
        self.Y_test = np.load("/mnt/data2/PPG/dataset_all/processed_data/mini_data/mini_ppg_test_cond.npy")
       

        # small small data
        # self.X_train = np.load("/mnt/data2/PPG/dataset_all/processed_data/ecg_data_test_mini.npz")['signal_data']
        # self.X_train = np.expand_dims(self.X_train, axis=1)
        # self.X_test = np.load("/mnt/data2/PPG/dataset_all/processed_data/ecg_data_test_mini.npz")['signal_data']
        # self.X_test = np.expand_dims(self.X_test, axis=1)
        # self.Y_train = np.load("/mnt/data2/PPG/dataset_all/processed_data/ppg_data_test_mini_cond.npy")
        # self.Y_test = np.load("/mnt/data2/PPG/dataset_all/processed_data/ppg_data_test_mini_cond.npy")
        

        
        # stage1 test y_data
        # self.Y_train = np.random.randint(0, 2, size=(self.X_train.shape[0],1))
        # self.Y_test = np.random.randint(0, 2, size=(self.X_test.shape[0],1))
        
        self.mean = np.nanmean(self.X_train, axis=(0, 2))[None,:,None]  # (1 c 1)
        self.std = np.nanstd(self.X_train, axis=(0, 2))[None,:,None]  # (1 c 1)
        print(f'self.mean.shape:', self.mean.shape, 'self.std.shape:', self.std.shape)
        print(f'self.X_train.shape:', self.X_train.shape, 'self.X_test.shape:', self.X_test.shape, 'self.Y_train.shape:', self.Y_train.shape, 'self.Y_test.shape:', self.Y_test.shape)

        

class CustomDataset(Dataset):
    def __init__(self, kind: str, dataset_importer:DatasetImporterCustom, **kwargs):
        """
        :param kind: "train" | "test"
        :param dataset_importer: instance of the `DatasetImporter` class.
        """
        super().__init__()
        kind = kind.lower()
        assert kind in ['train', 'test']
        self.kind = kind
        
        if kind == "train":
            self.X, self.Y = dataset_importer.X_train, dataset_importer.Y_train
        elif kind == "test":
            self.X, self.Y = dataset_importer.X_test, dataset_importer.Y_test
        else:
            raise ValueError

        self._len = self.X.shape[0]
    
    def __getitem__(self, idx):
        x, y = self.X[idx, :], self.Y[idx, :]
        return x, y

    def __len__(self):
        return self._len


if __name__ == "__main__":
    dataset = DatasetImporterCustom()
    # import os
    # import matplotlib.pyplot as plt
    # import torch
    # from torch.utils.data import DataLoader
    # os.chdir("../")

    # # data pipeline
    # dataset_importer = DatasetImporterUCR("ScreenType", data_scaling=True)
    # dataset = UCRDataset("train", dataset_importer)
    # data_loader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=True)

    # # get a mini-batch of samples
    # for batch in data_loader:
    #     x, y = batch
    #     break
    # print('x.shape:', x.shape)
    # print('y.shape:', y.shape)
    # print(y.flatten())

    # # plot
    # n_samples = 10
    # c = 0
    # fig, axes = plt.subplots(n_samples, 2, figsize=(3.5*2, 1.7*n_samples))
    # for i in range(n_samples):
    #     axes[i,0].plot(x[i, c])
    #     axes[i,0].set_title(f'class: {y[i,0]}')
    #     xf = torch.stft(x[[i], c], n_fft=4, hop_length=1, normalized=False)
    #     print('xf.shape:', xf.shape)
    #     xf = np.sqrt(xf[0,:,:,0]**2 + xf[0,:,:,1]**2)
    #     axes[i,1].imshow(xf, aspect='auto')
    #     axes[i, 1].invert_yaxis()
    # plt.tight_layout()
    # plt.show()
