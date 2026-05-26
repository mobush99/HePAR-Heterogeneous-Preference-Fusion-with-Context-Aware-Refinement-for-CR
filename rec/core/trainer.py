from abc import ABC, abstractmethod
from typing import Dict


class BaseTrainer(ABC):

    @abstractmethod
    def train(self, device: str) -> Dict:
        ...

    @abstractmethod
    def _train_one_epoch(self, *args, **kwargs) -> Dict:
        ...

    def save_checkpoint(self, path: str) -> None:
        self.model.save(path)

    def load_checkpoint(self, path: str, device: str = "cpu") -> None:
        self.model.load(path, device)
