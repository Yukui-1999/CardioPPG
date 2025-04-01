import torch
import torch.nn as nn
from generators.SingnalEncoder import vit_base_patchX
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import pickle

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
ppg_conditional_model = vit_base_patchX(
                img_size=(1, 2560),
                patch_size=(1, 128),
                in_chans=1,
                num_classes=128,
                drop_rate=0.1
        )
ppg_checkpoint_model = torch.load('/home/dingzhengyao/Work/PPG-ECG/Project/P2E_v2/Signal_generate/TimeVQVAE/saved_models/aligned_ppg.pth', map_location='cpu')
msg = ppg_conditional_model.load_state_dict(ppg_checkpoint_model, strict=False)
print(f'load ppg conditional model: {msg}')
ppg_conditional_model.eval()
ppg_conditional_model.to(device)
for param in ppg_conditional_model.parameters():
    param.requires_grad = False
    
class Processed_Signal_dataset(Dataset):
    def __init__(self, ppg_path, ecg_path):
        
        self.ppg = np.load(ppg_path)['signal_data']
        self.ecg = np.load(ecg_path)['signal_data']
       
        
        print('debug is false')
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
        
        print(f'self.ppg.shape:', self.ppg.shape, 'self.ecg.shape:', self.ecg.shape)
        
        
        if 'train' in ppg_path:
            np.savez('/mnt/data2/PPG/dataset_all/processed_data/ppg_data_train_v2.npz', signal_data=self.ppg)
            np.savez('/mnt/data2/PPG/dataset_all/processed_data/ecg_data_train_v2.npz', signal_data=self.ecg)
        elif 'test' in ppg_path:
            np.savez('/mnt/data2/PPG/dataset_all/processed_data/ppg_data_test_v2.npz', signal_data=self.ppg)
            np.savez('/mnt/data2/PPG/dataset_all/processed_data/ecg_data_test_v2.npz', signal_data=self.ecg)
        
        exit()
    def __len__(self):
        return self.ppg.shape[0]

    def __getitem__(self, index):
        ppg = self.ppg[index]
        
        return torch.tensor(ppg, dtype=torch.float32)
        
# class TensorDataset(Dataset):

#     def __init__(self):
#         self.data = np.load("/mnt/data2/PPG/dataset_all/processed_data/mini_data/mini_ppg_test.npy")
#         self.data = np.expand_dims(self.data, axis=1)
#         print(f'self.data.shape:', self.data.shape)

#     def __getitem__(self, index):
#         return self.data[index]

#     def __len__(self):
#         return len(self.data)


dataset = Processed_Signal_dataset(ppg_path='/mnt/data2/PPG/dataset_all/processed_data/ppg_data_test.npz', ecg_path='/mnt/data2/PPG/dataset_all/processed_data/ecg_data_test.npz')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False,drop_last=False)


ppg_condition_result = []
for ppg in tqdm(dataloader):
    ppg = ppg.float().unsqueeze(1).unsqueeze(1).to(device)  # (b 1 1 l)
    with torch.no_grad():
        ppg_condition_result.extend(ppg_conditional_model(ppg)[1])

ppg_condition_result = torch.stack(ppg_condition_result, dim=0)
ppg_condition_result = ppg_condition_result.cpu().numpy()
print(f'ppg_condition_result: {ppg_condition_result.shape}')
np.save("/mnt/data2/PPG/dataset_all/processed_data/ppg_data_train_cond.npy", ppg_condition_result)