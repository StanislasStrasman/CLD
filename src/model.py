import torch
import math
from typing import Optional, Union, Any
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear



# This model is a simpler version of the model proposed in tqch/ddpm-torch/blob/master/ddpm_torch/toy/toy_model.py

DEFAULT_DTYPE = torch.float32

def get_timestep_embedding(timesteps: Tensor, embed_dim: int, dtype: torch.dtype = DEFAULT_DTYPE):
    """
    Adapted from fairseq/fairseq/modules/sinusoidal_positional_embedding.py
    The implementation is slightly different from the decription in Section 3.5 of [1]
    [1] Vaswani, Ashish, et al. "Attention is all you need."
     Advances in neural information processing systems 30 (2017).
    """
    half_dim = embed_dim // 2
    embed = math.log(10000) / (half_dim - 1)
    embed = torch.exp(-torch.arange(half_dim, dtype=dtype, device=timesteps.device) * embed)
    embed = torch.outer(timesteps.ravel().to(dtype), embed)
    embed = torch.cat([torch.sin(embed), torch.cos(embed)], dim=1)
    return embed

DEFAULT_NONLINEARITY = nn.ReLU(inplace=True)

class TemporalLayer(nn.Module):
    nonlinearity = DEFAULT_NONLINEARITY

    def __init__(self, in_features: int, out_features: int, temporal_features: int):
        super(TemporalLayer, self).__init__()
        self.fc1 = Linear(in_features, out_features, bias=True)
        self.enc = Linear(temporal_features, out_features, bias=True)

    def forward(self, x: Tensor, t_emb: Tensor):
        out = self.nonlinearity(self.fc1(x) + self.enc(t_emb))
        return x + out # residual network

class Sequential(nn.Sequential):
    def forward(self, input: Tensor, **kwargs: Any):
        for module in self:
            input = module(input, **kwargs)
        return input

class Decoder(nn.Module):
    nonlinearity = DEFAULT_NONLINEARITY

    def __init__(self, in_features: int, mid_features: int, num_temporal_layers: int):
        super(Decoder, self).__init__()

        self.in_fc = Linear(in_features, mid_features, bias=True)
        self.temp_fc = Sequential(*(
            [TemporalLayer(mid_features, mid_features, mid_features)
            for _ in range(num_temporal_layers) ]))
        self.out_fc = Linear(mid_features, in_features//2)
        
        self.t_proj = nn.Sequential(
            Linear(mid_features, mid_features, bias=False),
            self.nonlinearity)
        self.mid_features = mid_features

    def forward(self, x: Tensor, t: Tensor):
        t_emb = get_timestep_embedding(t, self.mid_features)
        t_emb = self.t_proj(t_emb)
        out = self.in_fc(x)
        out = self.temp_fc(out, t_emb=t_emb)
        out = self.out_fc(out)
        return out
    

class DualDecoder(nn.Module):
    def __init__(self, in_features: int, mid_features: int, num_temporal_layers: int, is_cld: bool = True):
        super(DualDecoder, self).__init__()
        self.decoder1 = Decoder(in_features, mid_features, num_temporal_layers)
        self.decoder2 = None
        if not is_cld:
            self.decoder2 = Decoder(in_features, mid_features, num_temporal_layers)

    def forward(self, x: Tensor, t: Tensor):
        out1 = self.decoder1(x, t)  
        if self.decoder2 is not None:
            out2 = self.decoder2(x, t)  
            return torch.cat([out1, out2], dim=1)  
        return out1
