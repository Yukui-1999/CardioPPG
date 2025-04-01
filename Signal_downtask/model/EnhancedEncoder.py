from functools import partial
from turtle import forward

import torch
import torch.nn as nn
import model.SingnalEncoder as SingnalEncoder
from model.ECG_generator import GeneratorUNet
class EnhancedEncoder(nn.Module):
    def __init__(self, args=None):
        super(EnhancedEncoder, self).__init__()
        self.args = args
        self.ecg_generator = GeneratorUNet()
        if args.ecg_generator:
            print("load pretrained ecg_generator")
            ecg_checkpoint = torch.load(args.ecg_generator, map_location='cpu')
            ecg_checkpoint_model = ecg_checkpoint['generator_state_dict']
            msg = self.ecg_generator.load_state_dict(ecg_checkpoint_model, strict=False)
            print('load ecg generator')
            print(msg)

        self.ecg_encoder = SingnalEncoder.__dict__[args.ecg_model](
                img_size=args.ecg_input_size,
                patch_size=args.ecg_patch_size,
                in_chans=args.ecg_input_channels,
                num_classes=args.latent_dim,
                drop_rate=args.ecg_drop_out,
                args=args,)
        if args.ecg_pretrained:
            if args.ecg_pretrained == 'SSL':
                print("load pretrained ecg_model")
                ecg_checkpoint = torch.load(args.ecg_pretrained_model, map_location='cpu')
                ecg_checkpoint_model = ecg_checkpoint['model']
                msg = self.ecg_encoder.load_state_dict(ecg_checkpoint_model, strict=False)
                print('load ecg model')
                print(msg)
            elif args.ecg_pretrained == 'Align':
                print("load pretrained ecg_model")
                ecg_checkpoint = torch.load(args.ecg_alignment_model, map_location='cpu')
                ecg_checkpoint_model = ecg_checkpoint['model']
                ecg_encoder_keys = {k: v for k, v in ecg_checkpoint_model.items() if
                                    k.startswith('ECG_encoder')}
                # remove ECG_encoder.head
                ecg_encoder_keys = {k: v for k, v in ecg_encoder_keys.items() if not k.startswith('ECG_encoder.head')}
                ecg_checkpoint_model = {k.replace('ECG_encoder.', ''): v for k, v in ecg_encoder_keys.items()}
                msg = self.ecg_encoder.load_state_dict(ecg_checkpoint_model, strict=False)
                print('load aligned ecg model')
                print(msg)
            else:
                print(f'ecg_pretrained error')
                exit()
        else:
            print(f'no pretrained model or alignment model')
        self.ppg_encoder = SingnalEncoder.__dict__[args.ppg_model](
                img_size=args.ppg_input_size,
                patch_size=args.ppg_patch_size,
                in_chans=args.ppg_input_channels,
                num_classes=args.latent_dim,
                drop_rate=args.ppg_drop_out,
                args=args)
        if args.ppg_pretrained:
            if args.ppg_pretrained == 'SSL':
                print("load pretrained ppg_model")
                ppg_checkpoint = torch.load(args.ppg_pretrained_model, map_location='cpu')
                ppg_checkpoint_model = ppg_checkpoint['model']
                msg = self.ppg_encoder.load_state_dict(ppg_checkpoint_model, strict=False)
                print('load PPG SSL model')
                print(msg)
            elif args.ppg_pretrained == 'Align':
                print("load pretrained ppg_model")
                ppg_checkpoint = torch.load(args.ppg_alignment_model, map_location='cpu')
                ppg_checkpoint_model = ppg_checkpoint['model']
                ppg_encoder_keys = {k: v for k, v in ppg_checkpoint_model.items() if
                                    k.startswith('PPG_encoder')}
                # remove PPG_encoder.head
                ppg_encoder_keys = {k: v for k, v in ppg_encoder_keys.items() if not k.startswith('PPG_encoder.head')}
                ppg_checkpoint_model = {k.replace('PPG_encoder.', ''): v for k, v in ppg_encoder_keys.items()}
                msg = self.ppg_encoder.load_state_dict(ppg_checkpoint_model, strict=False)
                print('load aligned ppg model')
                print(msg)
            else:
                print(f'ppg_pretrained error')
                exit()
        else:
            print(f'no pretrained model or alignment model')
        
        self.concat_head = nn.Linear(2 * args.embed_dim, args.embed_dim)
        self.head = nn.Linear(args.embed_dim, args.latent_dim)
    

    def forward(self, ppg):
        ppg_latent, _ = self.ppg_encoder(ppg)
        ecg = self.ecg_generator(ppg.squeeze(2))
        # normalize
        ecg = 2 * (ecg - ecg.min(dim=2, keepdim=True)[0]) / (ecg.max(dim=2, keepdim=True)[0] - ecg.min(dim=2, keepdim=True)[0]) - 1
        ecg = ecg.unsqueeze(1)
        ecg_latent, _ = self.ecg_encoder(ecg)
        ppg_latent = self.ppg_encoder.norm(ppg_latent)[:,0]
        ecg_latent = self.ecg_encoder.norm(ecg_latent)[:,0]
        x = torch.cat([ppg_latent, ecg_latent], dim=1)
        x = self.concat_head(x)
        x = self.head(x)
        return x
        