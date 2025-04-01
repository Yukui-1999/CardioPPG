import torch
import torch.nn as nn
import model.SingnalEncoder as SingnalEncoder
from loss.clip_loss import ClipLoss
from itertools import combinations

class CEPP(nn.Module):
    def __init__(self, global_pool=False, device=None, args=None) -> None:
        super().__init__()
        self.device = device
        self.args = args
        
        self.ECG_encoder = SingnalEncoder.__dict__[args.ecg_model](
            img_size=args.ecg_input_size,
            patch_size=args.ecg_patch_size,
            in_chans=args.ecg_input_channels,
            num_classes=args.latent_dim,
            drop_rate=args.ecg_drop_out,
            args=args,
        )
        if args.ecg_pretrained:
            print("load pretrained ecg_model")
            ecg_checkpoint = torch.load(args.ecg_pretrained_model, map_location='cpu')
            ecg_checkpoint_model = ecg_checkpoint['model']
            msg = self.ECG_encoder.load_state_dict(ecg_checkpoint_model, strict=False)
            print('load ecg model')
            print(msg)
        
        
        self.PPG_encoder = SingnalEncoder.__dict__[args.ppg_model](
            img_size=args.ppg_input_size,
            patch_size=args.ppg_patch_size,
            in_chans=args.ppg_input_channels,
            num_classes=args.latent_dim,
            drop_rate=args.ppg_drop_out,
            args=args,
        )
        if args.ppg_pretrained:
            print("load pretrained ppg_model")
            ppg_checkpoint = torch.load(args.ppg_pretrained_model, map_location='cpu')
            ppg_checkpoint_model = ppg_checkpoint['model']
            msg = self.PPG_encoder.load_state_dict(ppg_checkpoint_model, strict=False)
            print('load ppg model')
            print(msg)


        self.loss_fn = ClipLoss(temperature=args.temperature, alpha_weight=args.alpha_weight, args=args)
        if args.startfrom_pretrained_model:
            self.resume_from_pretrained_model(args.startfrom_pretrained_model)

    def resume_from_pretrained_model(self, pretrained_model):
        checkpoint = torch.load(pretrained_model, map_location='cpu')
        self.load_state_dict(checkpoint['model'])
        print(f"load pretrained model from {pretrained_model}")
        
    def forward_loss(self, output_dict):
        all_combinations = combinations(output_dict.keys(), 2)
        loss_dict = {}

        for key_combination in all_combinations:
            # print(key_combination)
            loss_name = f"{key_combination[0][:-8]}_{key_combination[1][:-8]}_loss"
            loss_dict[loss_name] = self.loss_fn(output_dict[key_combination[0]], output_dict[key_combination[1]])
        loss_dict["total_loss"] = sum(loss_dict.values())
        
        return loss_dict["total_loss"]

    def forward(self, ecg, ppg):
        # feature means with head, inter means without head
        ecg_inter,ecg_feature = self.ECG_encoder(ecg)
        # print(f'ecg_inter shape:{ecg_inter.shape},ecg_feature shape:{ecg_feature.shape}')
        ppg_inter,ppg_feature = self.PPG_encoder(ppg)
        # print(f'ppg_inter shape:{ppg_inter.shape},ppg_feature shape:{ppg_feature.shape}')
        loss = self.loss_fn(ecg_feature,ppg_feature)
        
        return loss,ecg_feature,ppg_feature
