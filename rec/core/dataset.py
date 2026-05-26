from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from torch.utils.data import Dataset


class QueryDataset(Dataset, ABC):
    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ...

    @abstractmethod
    def collate_fn(self, batch: List[Dict]) -> Tuple:
        ...


class ItemDataset(Dataset, ABC):

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ...

    @abstractmethod
    def collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        ...
