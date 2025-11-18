from typing import Any, Callable, Protocol
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

class _SupportsUpdate(Protocol):
    def update(self) -> None: ...

def train_one_epoch(
    loss_fn: Callable[[torch.Tensor, bool], torch.Tensor],
    model: nn.Module,
    dataloader: DataLoader,
    config: Any,
    optimizer: Optimizer,
    ema: _SupportsUpdate,
    is_cld: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x0 in dataloader:
        loss = loss_fn(x0, is_cld)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
        optimizer.step()
        ema.update()

        bs = x0.shape[0]
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / total_samples
