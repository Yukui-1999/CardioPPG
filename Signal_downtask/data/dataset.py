from ast import arg
import pickle
from re import S
import torch.utils.data as data
import torch
import numpy as np
import neurokit2 as nk
from scipy.signal import resample
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import os 


def balance_data_all(ecg_data, ppg_data, labels):
    # 获取0和1的标签数量
    print(f'labels:{labels}')
    labels = np.array(labels)
    label_0_count = np.sum(labels == 0)
    label_1_count = np.sum(labels == 1)
    print(f'label_0_count:{label_0_count}')
    print(f'label_1_count:{label_1_count}')
    print(f'ecg_data shape: {ecg_data.shape}')
    print(f'ppg_data shape: {ppg_data.shape}')
    # 找到较少的标签数量
    min_count = min(label_0_count, label_1_count)
    
    # 挑选少标签数量对应的数据
    ecg_data_0 = ecg_data[labels == 0]
    ppg_data_0 = ppg_data[labels == 0]
    print(f'ecg_data_0 shape: {ecg_data_0.shape}')
    print(f'ppg_data_0 shape: {ppg_data_0.shape}')
    ecg_data_1 = ecg_data[labels == 1]
    ppg_data_1 = ppg_data[labels == 1]
    
    # 对多标签进行随机抽样，数量为少标签数量
    if label_0_count > label_1_count:
        # 同时对ecg和ppg进行随机抽样，保证顺序一致
        chosen_indices_0 = np.random.choice(label_0_count, min_count, replace=False)
        print(f'chosen_indices_0 shape: {chosen_indices_0.shape}')
        chosen_ecg_data_0 = ecg_data_0[chosen_indices_0]
        chosen_ppg_data_0 = ppg_data_0[chosen_indices_0]
        chosen_ecg_data_1 = ecg_data_1
        chosen_ppg_data_1 = ppg_data_1
    else:
        # 同时对ecg和ppg进行随机抽样，保证顺序一致
        chosen_indices_1 = np.random.choice(label_1_count, min_count, replace=False)
        print(f'chosen_indice_1 shape: {chosen_indices_1.shape}')
        chosen_ecg_data_0 = ecg_data_0
        chosen_ppg_data_0 = ppg_data_0
        chosen_ecg_data_1 = ecg_data_1[chosen_indices_1]
        chosen_ppg_data_1 = ppg_data_1[chosen_indices_1]
    
    print(f'Chosen ECG Data (Label 0) shape: {chosen_ecg_data_0.shape}')
    print(f'Chosen PPG Data (Label 0) shape: {chosen_ppg_data_0.shape}')
    print(f'Chosen ECG Data (Label 1) shape: {chosen_ecg_data_1.shape}')
    print(f'Chosen PPG Data (Label 1) shape: {chosen_ppg_data_1.shape}')
    print(f'min_count: {min_count}')
    
    # 合并ECG、PPG数据和标签
    balanced_ecg_data = np.concatenate([chosen_ecg_data_0, chosen_ecg_data_1], axis=0)
    balanced_ppg_data = np.concatenate([chosen_ppg_data_0, chosen_ppg_data_1], axis=0)
    balanced_labels = np.concatenate([np.zeros(min_count), np.ones(min_count)], axis=0)
    
    # 打乱数据和标签
    shuffle_indices = np.random.permutation(len(balanced_labels))
    balanced_ecg_data = balanced_ecg_data[shuffle_indices]
    balanced_ppg_data = balanced_ppg_data[shuffle_indices]
    balanced_labels = balanced_labels[shuffle_indices]
    
    return balanced_ecg_data, balanced_ppg_data, balanced_labels

def balance_data(data, labels):
    # 获取0和1的标签数量
    label_0_count = np.sum(labels == 0)
    label_1_count = np.sum(labels == 1)
    
    # 找到较少的标签数量
    min_count = min(label_0_count, label_1_count)
    
    # 挑选少标签数量对应的数据
    data_0 = data[labels == 0]
    data_1 = data[labels == 1]
    
    # 对多标签进行随机抽样，数量为少标签数量
    if label_0_count > label_1_count:
        chosen_data_0 = data_0[np.random.choice(label_0_count, min_count, replace=False)]
        chosen_data_1 = data_1
    else:
        chosen_data_0 = data_0
        chosen_data_1 = data_1[np.random.choice(label_1_count, min_count, replace=False)]
    print(f'chosen_data_0 shape: {chosen_data_0.shape}')
    print(f'chosen_data_1 shape: {chosen_data_1.shape}')
    print(f'min_count: {min_count}')
    # 合并数据和标签
    balanced_data = np.concatenate([chosen_data_0, chosen_data_1], axis=0)
    balanced_labels = np.concatenate([np.zeros(min_count), np.ones(min_count)], axis=0)
    
    # 打乱数据和标签
    shuffle_indices = np.random.permutation(len(balanced_labels))
    balanced_data = balanced_data[shuffle_indices]
    balanced_labels = balanced_labels[shuffle_indices]
    
    return balanced_data, balanced_labels

def process(data):
    if type(data) == np.ndarray:
        return float(data[0][0])
    else:
        return data
def processMIMICBP(data):
    data_ = [process(i) for i in data]
    return np.array(data_)
def process_ppg(signal,original_sampling_rate=None):
    # assert 信号没有nan值并且信号的值不处处相同
    try:
        assert not np.isnan(signal).any()
        assert not np.all(signal == signal[0])
        # resample ppg to 256Hz
        if original_sampling_rate == 125 or 1000:
            signal = resample(signal, 2560)
        assert len(signal) == 2560, f'len(signal): {len(signal)}'
        # no need for else because the original sampling rate is 256Hz (preprocessed)
        ppg = nk.ppg_clean(signal, sampling_rate=256)
        ppg = 2 * (ppg - np.min(ppg)) / (np.max(ppg) - np.min(ppg)) - 1
        return ppg
    except Exception as e:
        print(f'error: {e}')
        return None
    
    

def process_ecg(ecg,original_sampling_rate=None):
    # assert 信号没有nan值并且信号的值不处处相同
    try:
        assert not np.isnan(ecg).any()
        assert not np.all(ecg == ecg[0])
        if original_sampling_rate == 125 or 1000:
            ecg = resample(ecg, 2560)
        assert len(ecg) == 2560
        # no need for else because the original sampling rate is 256Hz (preprocessed)
        ecg, is_inverted = nk.ecg_invert(ecg, sampling_rate=256)
        ecg = nk.ecg_clean(ecg, sampling_rate=256, method="pantompkins1985")
        # normalize
        ecg = 2 * (ecg - np.min(ecg)) / (np.max(ecg) - np.min(ecg)) - 1
        return ecg
    except Exception as e:
        print(f'error: {e}')
        return None

class PPGBP_Signal_dataset(data.Dataset):
    def __init__(self, data_index, args=None):
        self.data = pickle.load(open(data_index, 'rb'))
        print(f'ppg length {len(self.data["ppg"][0])}')
        
        self.ppg = [process_ppg(signal,1000) for signal in self.data['ppg']]
        valid_indices = [index for index, ppg_signal in enumerate(self.ppg) if ppg_signal is not None]
        
        
        self.ppg = [self.ppg[index] for index in valid_indices]
        self.Hypertension = [self.data['label_b_Hypertension'][index] for index in valid_indices]
        self.Diabetes = [self.data['label_b_Diabetes'][index] for index in valid_indices]
        self.CerebralInfarction = [self.data['label_b_CerebralInfarction'][index] for index in valid_indices]
        self.CerebrovascularDisease = [self.data['label_b_CerebrovascularDisease'][index] for index in valid_indices]
        assert len(valid_indices) == len(self.Hypertension) == len(self.Diabetes) == len(self.CerebralInfarction) == len(self.CerebrovascularDisease), 'valid_indices error'
        
        if args.label == 'Hypertension':
            self.target = self.Hypertension
        elif args.label == 'Diabetes':
            self.target = self.Diabetes
            # print(self.target)
            
        elif args.label == 'CerebralInfarction':
            self.target = self.CerebralInfarction
        elif args.label == 'CerebrovascularDisease':
            self.target = self.CerebrovascularDisease
        
        # print label 1 num and label 0 num
        print(f'{args.label} label 1 num: {np.sum(self.target)}')
        print(f'{args.label} label 0 num: {len(self.target) - np.sum(self.target)}')
    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, index):
        ppg = self.ppg[index]
        label = self.target[index]
        batch = {'ppg': ppg, 'label': label}
        return batch


class MIMIC_Signal_dataset(data.Dataset):
    def __init__(self, data_index, args=None):

        self.args = args
        if 'train' in data_index:
            self.train = True
        else:
            self.train = False
        
        if args.label == '410':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label410_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label410_test.npy")
        elif args.label == '412':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label412_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label412_test.npy")
        elif args.label == '414':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label414_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label414_test.npy")
        elif args.label == '416':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label416_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label416_test.npy")
        elif args.label == '424':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label424_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label424_test.npy")
        elif args.label == '425':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label425_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label425_test.npy")
        elif args.label == '427':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label427_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label427_test.npy")
        elif args.label == '428':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label428_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label428_test.npy")
        elif args.label == '396':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label396_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label396_test.npy")
        elif args.label == '413':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label413_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label413_test.npy")
        elif args.label == '440':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label440_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label440_test.npy")
        elif args.label == '426':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label426_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label426_test.npy")
        elif args.label == '415':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label415_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label415_test.npy")
        elif args.label == '459':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label459_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label459_test.npy")
        elif args.label == '411':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label411_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label411_test.npy")
        elif args.label == '423':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label423_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label423_test.npy")
        elif args.label == '397':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label397_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label397_test.npy")
        elif args.label == '429':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label429_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label429_test.npy")
        elif args.label == '444':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label444_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label444_test.npy")
        elif args.label == '420':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label420_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label420_test.npy")
        elif args.label == '421':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label421_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label421_test.npy")
        elif args.label == '427_41':
            if self.train:
                self.target = np.load('/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label_427_41_train.npy')
            else:
                self.target = np.load('/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label_427_41_test.npy')
        elif args.label == '427_0':
            if self.train:
                self.target = np.load('/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label_427_0_train.npy')
            else:
                self.target = np.load('/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label_427_0_test.npy')
        elif args.label == '427_1':
            if self.train:
                self.target = np.load('/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label_427_1_train.npy')
            else:
                self.target = np.load('/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label_427_1_test.npy')
        elif args.label == '426_0_1':
            if self.train:
                self.target = np.load('/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label_426_0_1_train.npy')
            else:
                self.target = np.load('/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label_426_0_1_test.npy')
        elif args.label == '401_405':
            if self.train:
                self.target = np.load('/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label_401_405_train.npy')
            else:
                self.target = np.load('/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label_401_405_test.npy')
        elif args.label == '414_0':
            if self.train:
                self.target = np.load('/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label_414_0_train.npy')
            else:
                self.target = np.load('/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label_414_0_test.npy')
        elif args.label == '410&411&413&414':
            print('merge 410, 411, 413, 414')
            if self.train:
                target_410 = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label410_train.npy")
                target_411 = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label411_train.npy")
                target_413 = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label413_train.npy")
                target_414 = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label414_train.npy")
                assert len(target_410) == len(target_411) == len(target_413) == len(target_414), 'len error'
                self.target = np.zeros(len(target_410))
                for i in range(len(target_410)):
                    if target_410[i] == 1 or target_411[i] == 1 or target_413[i] == 1 or target_414[i] == 1:
                        self.target[i] = 1
            else:
                target_410 = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label410_test.npy")
                target_411 = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label411_test.npy")
                target_413 = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label413_test.npy")
                target_414 = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_label414_test.npy")
                assert len(target_410) == len(target_411) == len(target_413) == len(target_414), 'len error'
                self.target = np.zeros(len(target_410))
                for i in range(len(target_410)):
                    if target_410[i] == 1 or target_411[i] == 1 or target_413[i] == 1 or target_414[i] == 1:
                        self.target[i] = 1
        elif args.label == 'segDBP':
            if self.train:
                self.target = processMIMICBP(pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_labelDBP_train.pkl", 'rb')))
                segDBP_scaler_savepath = "/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/segDBP_scaler.pkl"
                if not os.path.exists(segDBP_scaler_savepath):
                    segDBP_scaler = StandardScaler()
                    self.target = segDBP_scaler.fit_transform(np.array(self.target).reshape(-1, 1)).reshape(-1)
                    pickle.dump(segDBP_scaler, open(segDBP_scaler_savepath, 'wb'))
                else:
                    segDBP_scaler = pickle.load(open(segDBP_scaler_savepath, 'rb'))
                    self.target = segDBP_scaler.transform(np.array(self.target).reshape(-1, 1)).reshape(-1)
            else:
                self.target = processMIMICBP(pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_labelDBP_test.pkl", 'rb')))
                segDBP_scaler = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/segDBP_scaler.pkl", 'rb'))
                self.target = segDBP_scaler.transform(np.array(self.target).reshape(-1, 1)).reshape(-1)
        elif args.label == 'segSBP':
            if self.train:
                self.target = processMIMICBP(pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_labelSBP_train.pkl", 'rb')))
                segSBP_scaler_savepath = "/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/segSBP_scaler.pkl"
                if not os.path.exists(segSBP_scaler_savepath):
                    segSBP_scaler = StandardScaler()
                    self.target = segSBP_scaler.fit_transform(np.array(self.target).reshape(-1, 1)).reshape(-1)
                    pickle.dump(segSBP_scaler, open(segSBP_scaler_savepath, 'wb'))
                else:
                    segSBP_scaler = pickle.load(open(segSBP_scaler_savepath, 'rb'))
                    self.target = segSBP_scaler.transform(np.array(self.target).reshape(-1, 1)).reshape(-1)
            else:
                self.target = processMIMICBP(pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/MIMIC_labelSBP_test.pkl", 'rb')))
                segSBP_scaler = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/MIMIC/segSBP_scaler.pkl", 'rb'))
                self.target = segSBP_scaler.transform(np.array(self.target).reshape(-1, 1)).reshape(-1)

        
        if args.signal == 'ecg':
            self.signal = np.load(data_index.replace('ppg', 'ecg'))['signal_data']
            if self.train:
                index_list = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/ecg_test_index_train_MIMIC.pkl", 'rb'))
            else:
                index_list = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/ecg_test_index_test_MIMIC.pkl", 'rb'))
            sorted_order = np.argsort(index_list)
            self.signal = self.signal[sorted_order]
        elif args.signal == 'ppg' or args.signal == 'enhancedppg':
            self.signal = np.load(data_index)['signal_data']
            if self.train:
                index_list = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/ppg_test_index_train_MIMIC.pkl", 'rb'))
            else:
                index_list = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/ppg_test_index_test_MIMIC.pkl", 'rb'))
            sorted_order = np.argsort(index_list)
            self.signal = self.signal[sorted_order]
        elif args.signal == 'both':
            self.ecg = np.load(data_index.replace('ppg', 'ecg'))['signal_data']
            self.signal = np.load(data_index)['signal_data']
            if self.train:
                index_list_ppg = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/ppg_test_index_train_MIMIC.pkl", 'rb'))
                index_list_ecg = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/ecg_test_index_train_MIMIC.pkl", 'rb'))
            else:
                index_list_ppg = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/ppg_test_index_test_MIMIC.pkl", 'rb'))
                index_list_ecg = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/ecg_test_index_test_MIMIC.pkl", 'rb'))
            sorted_order_ppg = np.argsort(index_list_ppg)
            sorted_order_ecg = np.argsort(index_list_ecg)
            self.ecg = self.ecg[sorted_order_ecg]
            self.signal = self.signal[sorted_order_ppg]
            assert len(self.ecg) == len(self.signal), 'len error'
        else:
            print('signal error')
            exit()

        if args.signal != 'both' and args.downtask_type == 'BCE' and args.balance:
            print(f'label:{args.label}')
            self.signal, self.target = balance_data(self.signal, self.target)
            if self.train:
                if args.training_percentage < 1 and args.training_percentage > 0:
                    print(f'args.training_percentage:{args.training_percentage}')
                    num_samples = int(self.signal.shape[0] * args.training_percentage)
                    indices = torch.randperm(self.signal.shape[0])[:num_samples]
                    self.signal = self.signal[indices]
                    self.target = self.target[indices]
                    print(f'after training_percentage:{self.signal.shape}')
                elif args.training_percentage == 1:
                    print(f'args.training_percentage:{args.training_percentage}')
                else:
                    print('training_percentage error')
                    exit()
                
                
         
    def __len__(self):
        return len(self.signal)

    def __getitem__(self, index):
        signal = self.signal[index]
        target = self.target[index]
        if self.args.signal == 'both':
            ecg = self.ecg[index]
            return {'ppg': signal, 'ecg': ecg, 'label': target}
        elif self.args.signal == 'ppg' or self.args.signal == 'enhancedppg':
            return {'ppg': signal, 'label': target}
        elif self.args.signal == 'ecg':
            return {'ecg': signal, 'label': target}



class VITAL_Signal_dataset(data.Dataset):
    def __init__(self, data_index, args=None):
        self.args = args
        if 'train' in data_index:
            self.train = True
        else:
            self.train = False
        if args.label == 'icu':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Vital/Vital_labelicu_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Vital/Vital_labelicu_test.npy")
        elif args.label == 'optype':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Vital/Vital_labeloptype_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Vital/Vital_labeloptype_test.npy")
            
            # 'Biliary/Pancreas', 'Breast', 'Colorectal', 'Hepatic', 'Major resection', 'Minor resection', 'Others', 'Stomach','Thyroid', 'Transplantation', 'Vascular'
            # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
            unique_labels = np.unique(self.target)
            label_map = {name: idx for idx, name in enumerate(unique_labels)}
            self.target = np.array([label_map[name] for name in self.target])
            print(np.unique(self.target, return_counts=True))

        elif args.label == 'htn':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Vital/Vital_labelhtn_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Vital/Vital_labelhtn_test.npy")
        elif args.label == 'dm':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Vital/Vital_labeldm_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Vital/Vital_labeldm_test.npy")
        elif args.label == 'pft':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Vital/Vital_labelpft_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Vital/Vital_labelpft_test.npy")
        elif args.label == 'segDBP':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Vital/Vital_labelDBP_train.npy")
                segDBP_scaler_savepath = "/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Vital/segDBP_scaler.pkl"
                if not os.path.exists(segDBP_scaler_savepath):
                    segDBP_scaler = StandardScaler()
                    self.target = segDBP_scaler.fit_transform(np.array(self.target).reshape(-1, 1)).reshape(-1)
                    pickle.dump(segDBP_scaler, open(segDBP_scaler_savepath, 'wb'))
                else:
                    segDBP_scaler = pickle.load(open(segDBP_scaler_savepath, 'rb'))
                    self.target = segDBP_scaler.transform(np.array(self.target).reshape(-1, 1)).reshape(-1)
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Vital/Vital_labelDBP_test.npy")
                segDBP_scaler = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Vital/segDBP_scaler.pkl", 'rb'))
                self.target = segDBP_scaler.transform(np.array(self.target).reshape(-1, 1)).reshape(-1)
        elif args.label == 'segSBP':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Vital/Vital_labelSBP_train.npy")
                segSBP_scaler_savepath = "/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Vital/segSBP_scaler.pkl"
                if not os.path.exists(segSBP_scaler_savepath):
                    segSBP_scaler = StandardScaler()
                    self.target = segSBP_scaler.fit_transform(np.array(self.target).reshape(-1, 1)).reshape(-1)
                    pickle.dump(segSBP_scaler, open(segSBP_scaler_savepath, 'wb'))
                else:
                    segSBP_scaler = pickle.load(open(segSBP_scaler_savepath, 'rb'))
                    self.target = segSBP_scaler.transform(np.array(self.target).reshape(-1, 1)).reshape(-1)
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Vital/Vital_labelSBP_test.npy")
                segSBP_scaler = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Vital/segSBP_scaler.pkl", 'rb'))
                self.target = segSBP_scaler.transform(np.array(self.target).reshape(-1, 1)).reshape(-1)


        if args.signal == 'ecg':
            self.signal = np.load(data_index.replace('ppg', 'ecg'))['signal_data']
            if self.train:
                index_list = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/ecg_test_index_train_Vital.pkl", 'rb'))
            else:
                index_list = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/ecg_test_index_test_Vital.pkl", 'rb'))
            sorted_order = np.argsort(index_list)
            self.signal = self.signal[sorted_order]
        elif args.signal == 'ppg' or args.signal == 'enhancedppg':
            self.signal = np.load(data_index)['signal_data']
            if self.train:
                index_list = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/ppg_test_index_train_Vital.pkl", 'rb'))
            else:
                index_list = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/ppg_test_index_test_Vital.pkl", 'rb'))
            sorted_order = np.argsort(index_list)
            self.signal = self.signal[sorted_order]
        elif args.signal == 'both':
            self.ecg = np.load(data_index.replace('ppg', 'ecg'))['signal_data']
            self.signal = np.load(data_index)['signal_data']
            if self.train:
                index_list_ppg = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/ppg_test_index_train_Vital.pkl", 'rb'))
                index_list_ecg = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/ecg_test_index_train_Vital.pkl", 'rb'))
            else:
                index_list_ppg = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/ppg_test_index_test_Vital.pkl", 'rb'))
                index_list_ecg = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/ecg_test_index_test_Vital.pkl", 'rb'))
            sorted_order_ppg = np.argsort(index_list_ppg)
            sorted_order_ecg = np.argsort(index_list_ecg)
            self.ecg = self.ecg[sorted_order_ecg]
            self.signal = self.signal[sorted_order_ppg]
            assert len(self.ecg) == len(self.signal), 'len error'
        else:
            print('signal error')
            exit()
        assert len(self.signal) == len(self.target), 'len error'

    def __len__(self):
        return len(self.signal)

    def __getitem__(self, index):
        signal = self.signal[index]
        target = self.target[index]
        if self.args.signal == 'both':
            ecg = self.ecg[index]
            return {'ppg': signal, 'ecg': ecg, 'label': target}
        elif self.args.signal == 'ppg' or self.args.signal == 'enhancedppg':
            return {'ppg': signal, 'label': target}
        elif self.args.signal == 'ecg':
            return {'ecg': signal, 'label': target}


class MESA_Signal_dataset(data.Dataset):
    def __init__(self, data_index, args=None):
        self.args = args
        if 'train' in data_index:
            self.train = True
        else:
            self.train = False
        if args.label == 'ahi3':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Mesa/Mesa_labelahi3_train.npy")
                ahi3_scaler_savepath = "/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Mesa/ahi3_scaler.pkl"
                if not os.path.exists(ahi3_scaler_savepath):
                    ahi3_scaler = StandardScaler()
                    self.target = ahi3_scaler.fit_transform(np.array(self.target).reshape(-1, 1)).reshape(-1)
                    pickle.dump(ahi3_scaler, open(ahi3_scaler_savepath, 'wb'))
                else:
                    ahi3_scaler = pickle.load(open(ahi3_scaler_savepath, 'rb'))
                    self.target = ahi3_scaler.transform(np.array(self.target).reshape(-1, 1)).reshape(-1)
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Mesa/Mesa_labelahi3_test.npy")
                ahi3_scaler = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Mesa/ahi3_scaler.pkl", 'rb'))
                self.target = ahi3_scaler.transform(np.array(self.target).reshape(-1, 1)).reshape(-1)

        elif args.label == 'ahi4':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Mesa/Mesa_labelahi4_train.npy")
                ahi4_scaler_savepath = "/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Mesa/ahi4_scaler.pkl"
                if not os.path.exists(ahi4_scaler_savepath):
                    ahi4_scaler = StandardScaler()
                    self.target = ahi4_scaler.fit_transform(np.array(self.target).reshape(-1, 1)).reshape(-1)
                    pickle.dump(ahi4_scaler, open(ahi4_scaler_savepath, 'wb'))
                else:
                    ahi4_scaler = pickle.load(open(ahi4_scaler_savepath, 'rb'))
                    self.target = ahi4_scaler.transform(np.array(self.target).reshape(-1, 1)).reshape(-1)
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Mesa/Mesa_labelahi4_test.npy")
                ahi4_scaler = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Mesa/ahi4_scaler.pkl", 'rb'))
                self.target = ahi4_scaler.transform(np.array(self.target).reshape(-1, 1)).reshape(-1)

        elif args.label == 'Apnea':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Mesa/Mesa_lable_b_Apnea_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Mesa/Mesa_lable_b_Apnea_test.npy")
        elif args.label == 'Arousal':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Mesa/Mesa_lable_b_Arousal_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Mesa/Mesa_lable_b_Arousal_test.npy")
        elif args.label == 'Hypopnea':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Mesa/Mesa_lable_b_Hypopnea_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Mesa/Mesa_lable_b_Hypopnea_test.npy")
        elif args.label == 'Limb_movement':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Mesa/Mesa_lable_b_Limb_movement_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Mesa/Mesa_lable_b_Limb_movement_test.npy")
        elif args.label == 'PLM':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Mesa/Mesa_lable_b_PLM_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Mesa/Mesa_lable_b_PLM_test.npy")
        elif args.label == 'SpO2_desaturation':
            if self.train:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Mesa/Mesa_lable_b_SpO2_desaturation_train.npy")
            else:
                self.target = np.load("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/Mesa/Mesa_lable_b_SpO2_desaturation_test.npy")


        
        if args.signal == 'ecg':
            self.signal = np.load(data_index.replace('ppg', 'ecg'))['signal_data']
            if self.train:
                index_list = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/ecg_test_index_train_Mesa.pkl", 'rb'))
            else:
                index_list = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/ecg_test_index_test_Mesa.pkl", 'rb'))
            sorted_order = np.argsort(index_list)
            self.signal = self.signal[sorted_order]
        elif args.signal == 'ppg' or args.signal == 'enhancedppg':
            self.signal = np.load(data_index)['signal_data']
            if self.train:
                index_list = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/ppg_test_index_train_Mesa.pkl", 'rb'))
            else:
                index_list = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/ppg_test_index_test_Mesa.pkl", 'rb'))
            sorted_order = np.argsort(index_list)
            self.signal = self.signal[sorted_order]
        elif args.signal == 'both':
            self.ecg = np.load(data_index.replace('ppg', 'ecg'))['signal_data']
            self.signal = np.load(data_index)['signal_data']
            if self.train:
                index_list_ppg = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/ppg_test_index_train_Mesa.pkl", 'rb'))
                index_list_ecg = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/ecg_test_index_train_Mesa.pkl", 'rb'))
            else:
                index_list_ppg = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/ppg_test_index_test_Mesa.pkl", 'rb'))
                index_list_ecg = pickle.load(open("/mnt/data2/ppg/dataset_all/processed_data/downtask_data/ecg_test_index_test_Mesa.pkl", 'rb'))
            sorted_order_ppg = np.argsort(index_list_ppg)
            sorted_order_ecg = np.argsort(index_list_ecg)
            self.ecg = self.ecg[sorted_order_ecg]
            self.signal = self.signal[sorted_order_ppg]
            assert len(self.ecg) == len(self.signal), 'len error'
        else:
            print('signal error')
            exit()
        assert len(self.signal) == len(self.target), 'len error'

    def __len__(self):
        return len(self.signal)

    def __getitem__(self, index):
        signal = self.signal[index]
        target = self.target[index]
        if self.args.signal == 'both':
            ecg = self.ecg[index]
            return {'ppg': signal, 'ecg': ecg, 'label': target}
        elif self.args.signal == 'ppg' or self.args.signal == 'enhancedppg':
            return {'ppg': signal, 'label': target}
        elif self.args.signal == 'ecg':
            return {'ecg': signal, 'label': target}


class MIMICAF_Signal_dataset(data.Dataset):
    def __init__(self, data_index, args=None):
        if 'train' in data_index:
            self.train = True
        else:
            self.train = False
        self.data = pickle.load(open(data_index, 'rb'))

        self.ppg = [process_ppg(signal,125) for signal in self.data['ppg']]
        valid_ppg_indices = [index for index, ppg_signal in enumerate(self.ppg) if ppg_signal is not None]
        self.ecg = [process_ecg(signal,125) for signal in self.data['ecg']]
        valid_ecg_indices = [index for index, ecg_signal in enumerate(self.ecg) if ecg_signal is not None]
        valid_indices = [index for index in valid_ppg_indices if index in valid_ecg_indices]

        self.af = [self.data['label'][index] for index in valid_indices]
        self.ppg = [self.ppg[index] for index in valid_indices]
        self.ecg = [self.ecg[index] for index in valid_indices]
        if args.label == 'AF':
            self.target = self.af
        else:
            print('label error')
            exit()
        # Ensure self.target contains only 0 or 1
        assert set(self.target).issubset({0, 1}), 'self.target contains values other than 0 and 1'
        
        # Print the number of 0s and 1s in self.target
        num_ones = np.sum(self.target)
        num_zeros = len(self.target) - num_ones
        print(f'AF label 1 num: {num_ones}')
        print(f'AF label 0 num: {num_zeros}')
        self.ecg = np.array(self.ecg)
        self.ppg = np.array(self.ppg)

        self.ecg, self.ppg, self.target = balance_data_all(self.ecg, self.ppg, self.target)

        if self.train:
            if args.training_percentage < 1 and args.training_percentage > 0:
                print(f'args.training_percentage:{args.training_percentage}')
                num_samples = int(self.ecg.shape[0] * args.training_percentage)
                indices = torch.randperm(self.ecg.shape[0])[:num_samples]
                self.ecg = self.ecg[indices]
                self.ppg = self.ppg[indices]
                self.target = self.target[indices]
            elif args.training_percentage == 1:
                print(f'args.training_percentage:{args.training_percentage}')
            else:
                print('training_percentage error')
                exit()

    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, index):
        ppg = self.ppg[index]
        ecg = self.ecg[index]
        label = self.target[index]
        batch = {'ppg': ppg, 'ecg': ecg, 'label': label}
        return batch

class BIDMC_Signal_dataset(data.Dataset):
    def __init__(self, data_index, args=None):
        self.data = np.load(data_index)['signal_data']
    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, index):
        signal = self.data[index]
        return torch.tensor(signal, dtype=torch.float32)
    

class WESAD_Signal_dataset(data.Dataset):
    def __init__(self, data_index, args=None):
        # 5分类 [0,1,2,3,4]
        self.data = pickle.load(open(data_index, 'rb'))

        self.ppg = [process_ppg(signal) for signal in self.data['PPG']]
        valid_ppg_indices = [index for index, ppg_signal in enumerate(self.ppg) if ppg_signal is not None]
        self.ecg = [process_ecg(signal) for signal in self.data['ECG']]
        valid_ecg_indices = [index for index, ecg_signal in enumerate(self.ecg) if ecg_signal is not None]
        valid_indices = [index for index in valid_ppg_indices if index in valid_ecg_indices]

        self.label = [self.data['label'][index] for index in valid_indices]
        self.ppg = [self.ppg[index] for index in valid_indices]
        self.ecg = [self.ecg[index] for index in valid_indices]

        # 去除label中的5 6 7标签，因为数据集说明里写这些标签无用
        label_valid_indices = [index for index, label in enumerate(self.label) if label in [0, 1, 2, 3, 4]]
        self.label = [self.label[index] for index in label_valid_indices]
        self.ppg = [self.ppg[index] for index in label_valid_indices]
        self.ecg = [self.ecg[index] for index in label_valid_indices]

        if args.label == 'Emotion':
            self.target = self.label
            print(np.unique(self.target, return_counts=True))
        else:
            print('label error')
            exit()
    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, index):
        ppg = self.ppg[index]
        ecg = self.ecg[index]
        label = self.target[index]
        batch = {'ppg': ppg, 'ecg': ecg, 'label': label}
        return batch
    

class Capon_Signal_dataset(data.Dataset):
    def __init__(self, data_index, args=None):
        self.data = np.load(data_index)['signal_data']
    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, index):
        signal = self.data[index]
        return torch.tensor(signal, dtype=torch.float32)
    

class BCG_Signal_dataset(data.Dataset):
    def __init__(self, data_index, args=None):
        self.data = np.load(data_index)['signal_data']
    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, index):
        signal = self.data[index]
        return torch.tensor(signal, dtype=torch.float32)
    

class PPGDaila_Signal_dataset(data.Dataset):
    def __init__(self, data_index, args=None):
        # 9分类 [0,1,2,3,4,5,6,7,8]
        self.data = pickle.load(open(data_index, 'rb'))

        self.ppg = [process_ppg(signal) for signal in self.data['PPG']]
        valid_ppg_indices = [index for index, ppg_signal in enumerate(self.ppg) if ppg_signal is not None]
        self.ecg = [process_ecg(signal) for signal in self.data['ECG']]
        valid_ecg_indices = [index for index, ecg_signal in enumerate(self.ecg) if ecg_signal is not None]
        valid_indices = [index for index in valid_ppg_indices if index in valid_ecg_indices]

        self.label = [self.data['label'][index] for index in valid_indices]
        self.ppg = [self.ppg[index] for index in valid_indices]
        self.ecg = [self.ecg[index] for index in valid_indices]

        
        label_valid_indices = [index for index, label in enumerate(self.label) if label in [0, 1, 2, 3, 4, 5, 6, 7, 8]]
        self.label = [self.label[index] for index in label_valid_indices]
        self.ppg = [self.ppg[index] for index in label_valid_indices]
        self.ecg = [self.ecg[index] for index in label_valid_indices]

        if args.label == 'Action':
            self.target = self.label
            print(np.unique(self.target, return_counts=True))
        else:
            print('label error')
            exit()
    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, index):
        ppg = self.ppg[index]
        ecg = self.ecg[index]
        label = self.target[index]
        batch = {'ppg': ppg, 'ecg': ecg, 'label': label}
        return batch
    

class Uqvital_Signal_dataset(data.Dataset):
    def __init__(self, data_index, args=None):
        self.data = pickle.load(open(data_index, 'rb'))
        self.train = True if 'train' in data_index else False
        self.ppg = [process_ppg(signal) for signal in tqdm(self.data['ppg'])]
        valid_ppg_indices = [index for index, ppg_signal in enumerate(self.ppg) if ppg_signal is not None]
        self.ecg = [process_ecg(signal) for signal in tqdm(self.data['ecg'])]
        valid_ecg_indices = [index for index, ecg_signal in enumerate(self.ecg) if ecg_signal is not None]
        valid_indices = [index for index in valid_ppg_indices if index in valid_ecg_indices]

        self.ppg = [self.ppg[index] for index in valid_indices]
        self.ecg = [self.ecg[index] for index in valid_indices]
        self.pulse = [self.data['pulse'][index] for index in valid_indices]
        self.spo2 = [self.data['spo2'][index] for index in valid_indices]
        self.perf = [self.data['perf'][index] for index in valid_indices]
        self.awp = [self.data['awp'][index] for index in valid_indices]
        self.awf = [self.data['awf'][index] for index in valid_indices]
        self.awv = [self.data['awv'][index] for index in valid_indices]
        if self.train:
            
            pulse_scaler_savepath = os.path.join(os.path.dirname(data_index), 'pulse_scaler.pkl')
            if not os.path.exists(pulse_scaler_savepath):
                pulse_scaler = StandardScaler()
                self.pulse = pulse_scaler.fit_transform(np.array(self.pulse).reshape(-1, 1)).reshape(-1)
                pickle.dump(pulse_scaler, open(pulse_scaler_savepath, 'wb'))
            else:
                pulse_scaler = pickle.load(open(pulse_scaler_savepath, 'rb'))
                self.pulse = pulse_scaler.transform(np.array(self.pulse).reshape(-1, 1)).reshape(-1)
            
            spo2_scaler_savepath = os.path.join(os.path.dirname(data_index), 'spo2_scaler.pkl')
            if not os.path.exists(spo2_scaler_savepath):
                spo2_scaler = StandardScaler()
                self.spo2 = spo2_scaler.fit_transform(np.array(self.spo2).reshape(-1, 1)).reshape(-1)
                pickle.dump(spo2_scaler, open(spo2_scaler_savepath, 'wb'))
            else:
                spo2_scaler = pickle.load(open(spo2_scaler_savepath, 'rb'))
                self.spo2 = spo2_scaler.transform(np.array(self.spo2).reshape(-1, 1)).reshape(-1)

            perf_scaler_savepath = os.path.join(os.path.dirname(data_index), 'perf_scaler.pkl')
            if not os.path.exists(perf_scaler_savepath):
                perf_scaler = StandardScaler()
                self.perf = perf_scaler.fit_transform(np.array(self.perf).reshape(-1, 1)).reshape(-1)
                pickle.dump(perf_scaler, open(perf_scaler_savepath, 'wb'))
            else:
                perf_scaler = pickle.load(open(perf_scaler_savepath, 'rb'))
                self.perf = perf_scaler.transform(np.array(self.perf).reshape(-1, 1)).reshape(-1)

            awp_scaler_savepath = os.path.join(os.path.dirname(data_index), 'awp_scaler.pkl')
            if not os.path.exists(awp_scaler_savepath):
                awp_scaler = StandardScaler()
                self.awp = awp_scaler.fit_transform(np.array(self.awp).reshape(-1, 1)).reshape(-1)
                pickle.dump(awp_scaler, open(awp_scaler_savepath, 'wb'))
            else:
                awp_scaler = pickle.load(open(awp_scaler_savepath, 'rb'))
                self.awp = awp_scaler.transform(np.array(self.awp).reshape(-1, 1)).reshape(-1)

            awf_scaler_savepath = os.path.join(os.path.dirname(data_index), 'awf_scaler.pkl')
            if not os.path.exists(awf_scaler_savepath):
                awf_scaler = StandardScaler()
                self.awf = awf_scaler.fit_transform(np.array(self.awf).reshape(-1, 1)).reshape(-1)
                pickle.dump(awf_scaler, open(awf_scaler_savepath, 'wb'))
            else:
                awf_scaler = pickle.load(open(awf_scaler_savepath, 'rb'))
                self.awf = awf_scaler.transform(np.array(self.awf).reshape(-1, 1)).reshape(-1)

            
            awv_scaler_savepath = os.path.join(os.path.dirname(data_index), 'awv_scaler.pkl')
            if not os.path.exists(awv_scaler_savepath):
                awv_scaler = StandardScaler()
                self.awv = awv_scaler.fit_transform(np.array(self.awv).reshape(-1, 1)).reshape(-1)
                pickle.dump(awv_scaler, open(awv_scaler_savepath, 'wb'))
            else:
                awv_scaler = pickle.load(open(awv_scaler_savepath, 'rb'))
                self.awv = awv_scaler.transform(np.array(self.awv).reshape(-1, 1)).reshape(-1)

        else:
            pulse_scaler = pickle.load(open(os.path.join(os.path.dirname(data_index), 'pulse_scaler.pkl'), 'rb'))
            self.pulse = pulse_scaler.transform(np.array(self.pulse).reshape(-1, 1)).reshape(-1)

            spo2_scaler = pickle.load(open(os.path.join(os.path.dirname(data_index), 'spo2_scaler.pkl'), 'rb'))
            self.spo2 = spo2_scaler.transform(np.array(self.spo2).reshape(-1, 1)).reshape(-1)

            perf_scaler = pickle.load(open(os.path.join(os.path.dirname(data_index), 'perf_scaler.pkl'), 'rb'))
            self.perf = perf_scaler.transform(np.array(self.perf).reshape(-1, 1)).reshape(-1)

            awp_scaler = pickle.load(open(os.path.join(os.path.dirname(data_index), 'awp_scaler.pkl'), 'rb'))
            self.awp = awp_scaler.transform(np.array(self.awp).reshape(-1, 1)).reshape(-1)

            awf_scaler = pickle.load(open(os.path.join(os.path.dirname(data_index), 'awf_scaler.pkl'), 'rb'))
            self.awf = awf_scaler.transform(np.array(self.awf).reshape(-1, 1)).reshape(-1)

            awv_scaler = pickle.load(open(os.path.join(os.path.dirname(data_index), 'awv_scaler.pkl'), 'rb'))
            self.awv = awv_scaler.transform(np.array(self.awv).reshape(-1, 1)).reshape(-1)

        if args.label == 'Pulse':
            self.target = self.pulse
        elif args.label == 'Spo2':
            self.target = self.spo2
        elif args.label == 'Perf':
            self.target = self.perf
        elif args.label == 'Awp':
            self.target = self.awp
        elif args.label == 'Awf':
            self.target = self.awf
        elif args.label == 'Awv':
            self.target = self.awv
        else:
            print('label error')
            exit()
            


    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, index):
        ppg = self.ppg[index]
        ecg = self.ecg[index]
        label = self.target[index]
        batch = {'ppg': ppg, 'ecg': ecg, 'label': label}
        return batch


class CLBPE_Signal_dataset(data.Dataset):
    def __init__(self, data_index, args=None):
        self.data = np.load(data_index)['signal_data']
    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, index):
        signal = self.data[index]
        return torch.tensor(signal, dtype=torch.float32)
    

class Sleepdisorder_Signal_dataset(data.Dataset):
    def __init__(self, data_index, args=None):
        self.data = pickle.load(open(data_index, 'rb'))
        print(len(self.data['ppg'][0]))
        self.ppg = [process_ppg(signal) for signal in tqdm(self.data['ppg'], desc='Processing PPG signals')]
        valid_indices = [index for index, ppg_signal in enumerate(self.ppg) if ppg_signal is not None]
        
        self.ahi = [self.data['ahi'][index] for index in valid_indices]
        self.ppg = [self.ppg[index] for index in valid_indices]
        
        if args.label == 'Ahi':
            self.target = self.ahi
        else:
            print('label error')
            exit()
        # Ensure self.target contains only 0 or 1
        assert set(self.target).issubset({0, 1}), 'self.target contains values other than 0 and 1'
        
        # Print the number of 0s and 1s in self.target
        num_ones = np.sum(self.target)
        num_zeros = len(self.target) - num_ones
        print(f'Ahi label 1 num: {num_ones}')
        print(f'Ahi label 0 num: {num_zeros}')

    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, index):
        ppg = self.ppg[index]
        label = self.target[index]
        batch = {'ppg': ppg, 'label': label}
        return batch
    

class ECSMP_Signal_dataset(data.Dataset):
    def __init__(self, data_index, args=None):
        self.data = pickle.load(open(data_index, 'rb'))
    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, index):
        signal = self.data[index]
        return torch.tensor(signal, dtype=torch.float32)




def get_dataset(args, mode='train'):
    if mode == 'train':
        if args.dataset == 'CLBPE':
            return CLBPE_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/Cuffless_BP/train.pkl", args)
        elif args.dataset == 'Sleepdisorder':
            return Sleepdisorder_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/sleepdisorder/train.pkl", args)
        elif args.dataset == 'ECSMP':
            return ECSMP_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/ECSMP/train.pkl", args)
        elif args.dataset == 'Uqvital':
            return Uqvital_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/Uqvital/train.pkl", args)
        elif args.dataset == 'PPGDaila':
            return PPGDaila_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/PPG_Daila/train.pkl", args)
        elif args.dataset == 'BCG':
            return BCG_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/BCG/train.pkl", args)
        elif args.dataset == 'Capon':
            return Capon_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/Capon/train.pkl", args)
        elif args.dataset == 'WESAD':
            return WESAD_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/WESAD/train.pkl", args)
        elif args.dataset == 'BIDMC':
            return BIDMC_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/BIDMC/train.pkl", args)
        elif args.dataset == 'MIMIC_AF':
            return MIMICAF_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/MIMIC_AF/train.pkl", args)
        elif args.dataset == 'PPGBP':
            return PPGBP_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/PPG_BP/train.pkl", args)
        elif args.dataset == 'MESA':
            return MESA_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/ppg_data_train_Mesa.npz", args)
        elif args.dataset == 'VITAL':
            return VITAL_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/ppg_data_train_Vital.npz", args)
        elif args.dataset == 'MIMIC':
            return MIMIC_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/ppg_data_train_MIMIC.npz", args)
        else:
            print(f'No such dataset: {args.dataset}')
            exit()
    elif mode == 'test':
        if args.dataset == 'CLBPE':
            return CLBPE_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/Cuffless_BP/test.pkl", args)
        elif args.dataset == 'Sleepdisorder':
            return Sleepdisorder_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/sleepdisorder/test.pkl", args)
        elif args.dataset == 'ECSMP':
            return ECSMP_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/ECSMP/test.pkl", args)
        elif args.dataset == 'Uqvital':
            return Uqvital_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/Uqvital/test.pkl", args)
        elif args.dataset == 'PPGDaila':
            return PPGDaila_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/PPG_Daila/test.pkl", args)
        elif args.dataset == 'BCG':
            return BCG_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/BCG/test.pkl", args)
        elif args.dataset == 'Capon':
            return Capon_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/Capon/test.pkl", args)
        elif args.dataset == 'WESAD':
            return WESAD_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/WESAD/test.pkl", args)
        elif args.dataset == 'BIDMC':
            return BIDMC_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/BIDMC/test.pkl", args)
        elif args.dataset == 'MIMIC_AF':
            return MIMICAF_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/MIMIC_AF/test.pkl", args)
        elif args.dataset == 'PPGBP':
            return PPGBP_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/PPG_BP/test.pkl", args)
        elif args.dataset == 'MESA':
            return MESA_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/ppg_data_test_Mesa.npz", args)
        elif args.dataset == 'VITAL':
            return VITAL_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/ppg_data_test_Vital.npz", args)
        elif args.dataset == 'MIMIC':
            return MIMIC_Signal_dataset("/mnt/data2/PPG/dataset_all/processed_data/downtask_data/ppg_data_test_MIMIC.npz", args)
        else:
            print(f'No such dataset: {args.dataset}')
            exit()