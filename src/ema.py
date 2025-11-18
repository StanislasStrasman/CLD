import torch
from typing import Dict
from torch import nn

class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.model: nn.Module = model
        self.decay: float = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
