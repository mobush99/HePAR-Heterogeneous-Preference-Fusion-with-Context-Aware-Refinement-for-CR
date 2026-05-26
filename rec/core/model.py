from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class BaseEncoder(ABC, nn.Module):
    @abstractmethod
    def encode_query(self, batch: dict) -> Tensor:
        ...

    @abstractmethod
    def encode_item(self, batch: dict) -> Tensor:
        ...

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: str = "cpu") -> None:
        state = torch.load(path, map_location=device)
        self.load_state_dict(state)
