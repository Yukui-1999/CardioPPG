import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns

import wandb


def plot_attention(original_signal, attentation_map, idx):
    """
    :input:
    original_signal (B, C, C_sig, T_sig)
    attention_map (B, Heads, C_sig*N_(C_sig), C_sig*N_(C_sig))
    """
    B, C, C_sig, T_sig = original_signal.shape
    B, Heads, N, N = attentation_map.shape

    NpC = int(N / C_sig) # N_(C_sig)

    # only for nice visualization 
    original_signal = (original_signal+0.5*abs(original_signal.min()))

    # (B, Heads, N_(C_sig), N_(C_sig)), attention map of the first signal channel
    channel = 0
    attentation_map = attentation_map[:, :, 1+(channel*NpC):1+((channel+1)*NpC), 1+(channel*NpC):1+((channel+1)*NpC)] # leave the cls token out
    # (B, Heads, N_(C_sig))
    attentation_map = attentation_map.mean(dim=2)
    attentation_map = F.normalize(attentation_map, dim=-1)
    attentation_map = attentation_map.softmax(dim=-1)
    # (B, Heads, T_sig)
    attentation_map = F.interpolate(attentation_map, size=T_sig, mode='linear')

    # (T_sig)
    original_signal = original_signal[idx, 0, 0].cpu()
    # (Heads, T_sig)
    attentation_map = attentation_map[idx].cpu()

    fig, axes = plt.subplots(nrows=Heads, sharex=True)

    for head in range(0, Heads):
        axes[head].plot(range(0, original_signal.shape[-1], 1), original_signal, zorder=2) # (2500)
        sns.heatmap(attentation_map[head, :].unsqueeze(dim=0).repeat(15, 1), linewidth=0.5, # (1, 2500)
                    alpha=0.3,
                    zorder=1,
                    ax=axes[head])
        axes[head].set_ylim(original_signal.min(), original_signal.max())

    # remove y labels of all subplots
    [ax.yaxis.set_visible(False) for ax in axes.ravel()]
    plt.tight_layout()

    wandb.log({"Attention": wandb.Image(fig)})
    plt.close('all')