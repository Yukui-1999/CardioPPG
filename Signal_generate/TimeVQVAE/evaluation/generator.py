import os
from typing import List, Union, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from generators.SingnalEncoder import vit_base_patchX
from experiments.exp_stage2 import ExpStage2
from generators.maskgit import MaskGIT
from preprocessing.data_pipeline import build_data_pipeline
from preprocessing.preprocess_ucr import DatasetImporterUCR, DatasetImporterCustom
from generators.sample import unconditional_sample, conditional_sample
from supervised_FCN_2.example_pretrained_model_loading import load_pretrained_FCN
from supervised_FCN_2.example_compute_FID import calculate_fid
from supervised_FCN_2.example_compute_IS import calculate_inception_score
from utils import time_to_timefreq, timefreq_to_time
from generators.neural_mapper import NeuralMapper
from evaluation.rocket_functions import generate_kernels, apply_kernels
from utils import zero_pad_low_freq, zero_pad_high_freq, remove_outliers
from evaluation.stat_metrics import marginal_distribution_difference, auto_correlation_difference, skewness_difference, kurtosis_difference
from tqdm import tqdm
from torch.utils.data import Dataset


class PPG_dataset(Dataset):
    def __init__(self, ppg):
        self.ppg = ppg
    def __len__(self):
        return len(self.ppg)
    def __getitem__(self, idx):
        return self.ppg[idx]

class generator(nn.Module):
    """
    - FID
    - IS
    - visual inspection
    - PCA
    - t-SNE
    """
    def __init__(self, 
                 dataset_name: str, 
                 in_channels:int,
                 input_length:int, 
                 n_classes:int, 
                 device:int, 
                 config:dict, 
                 use_neural_mapper:bool=False,
                 feature_extractor_type:str='rocket',
                 ppg:np.ndarray=None,
                 use_custom_dataset:bool=False,
                 dataset_importer:DatasetImporterCustom=None,
                 ):
        super().__init__()
        self.dataset_name = dataset_name
        self.device = torch.device(device)
        self.config = config
        self.batch_size = self.config['evaluation']['batch_size']
        

        self.Y_test = ppg.astype(np.float64) 
        

        self.ts_len = ppg.shape[-1]  # time series length
        self.n_classes = 1 #无用参数

        # load the stage2 model
        self.stage2 = ExpStage2.load_from_checkpoint(os.path.join('saved_models', f'stage2-{dataset_name}-2025-3-16-23-26-27.ckpt'), 
                                                      dataset_name=dataset_name, 
                                                      in_channels=in_channels,
                                                      input_length=input_length, 
                                                      config=config,
                                                      n_classes=n_classes,
                                                    #   use_neural_mapper=False,
                                                      feature_extractor_type=feature_extractor_type,
                                                      use_custom_dataset=use_custom_dataset,
                                                      dataset_importer=dataset_importer,
                                                      map_location='cpu',
                                                      strict=False)
        self.stage2.eval()
        self.maskgit = self.stage2.maskgit
        self.stage1 = self.stage2.maskgit.stage1
        self.ppg_conditional_model = vit_base_patchX(
                img_size=(1, 2560),
                patch_size=(1, 128),
                in_chans=1,
                num_classes=128,
                drop_rate=0.1
        )
        ppg_checkpoint_model = torch.load('/home/dingzhengyao/Work/PPG-ECG/Project/P2E_v2/Signal_generate/TimeVQVAE/saved_models/aligned_ppg.pth', map_location='cpu')
        msg = self.ppg_conditional_model.load_state_dict(ppg_checkpoint_model, strict=False)
        print(f'load ppg conditional model: {msg}')
        self.ppg_conditional_model.eval()
        self.ppg_conditional_model.to(device)
        for param in self.ppg_conditional_model.parameters():
            param.requires_grad = False
    
        
    @torch.no_grad()
    def sample(self, n_samples: int, kind: str, class_index:Union[int,None]=None, unscale:bool=False, batch_size=None):
        """
        
        unscale: unscale the generated sample with percomputed mean and std.
        """
        assert kind in ['unconditional', 'conditional']
        class_index = torch.Tensor(self.Y_test).to(self.device) if class_index is None else class_index
        ppgdataset = PPG_dataset(class_index)
        dataloader = torch.utils.data.DataLoader(ppgdataset, batch_size=256, shuffle=False,drop_last=False)
        ppg_condition_result = []
        for ppg in tqdm(dataloader):
            ppg = ppg.float().unsqueeze(1).to(self.device)  # (b 1 1 l)
            with torch.no_grad():
                ppg_condition_result.extend(self.ppg_conditional_model(ppg)[1])

        ppg_condition_result = torch.stack(ppg_condition_result, dim=0)
        print(f'ppg_condition_result.shape: {ppg_condition_result.shape}')
        # sampling
        if kind == 'unconditional':
            x_new_l, x_new_h, x_new = unconditional_sample(self.maskgit, n_samples, self.device, batch_size=batch_size if not isinstance(batch_size, type(None)) else self.batch_size)  # (b c l); b=n_samples, c=1 (univariate)
        elif kind == 'conditional':
            x_new_l, x_new_h, x_new = conditional_sample(self.maskgit, n_samples, self.device, ppg_condition_result, batch_size=batch_size if not isinstance(batch_size, type(None)) else self.batch_size)  # (b c l); b=n_samples, c=1 (univariate)
            # x_new_l, x_new_h, x_new = conditional_sample(self.maskgit, n_samples, self.device, class_index, batch_size=batch_size if not isinstance(batch_size, type(None)) else self.batch_size)  # (b c l); b=n_samples, c=1 (univariate)
        else:
            raise ValueError

      
        
        print(f'x_new_l: {x_new_l.shape}, x_new_h: {x_new_h.shape}, x_new: {x_new.shape}')
   

        return x_new

    