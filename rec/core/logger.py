from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class Logger(ABC):

    @abstractmethod
    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        ...

    @abstractmethod
    def finish(self) -> None:
        ...


class ConsoleLogger(Logger):
    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        pass

    def finish(self) -> None:
        pass


class WandbLogger(Logger):
    def __init__(
        self,
        project: str,
        config: dict,
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        group: Optional[str] = None,
    ) -> None:
        import wandb
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            group=group,
            config=config,
        )

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        self._run.log(metrics, step=step)

    def finish(self) -> None:
        self._run.finish()


class CompositeLogger(Logger):
    def __init__(self, loggers: List[Logger]) -> None:
        self.loggers = loggers

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for lg in self.loggers:
            lg.log(metrics, step)

    def finish(self) -> None:
        for lg in self.loggers:
            lg.finish()


def build_logger(config: dict, use_wandb: bool, **wandb_kwargs) -> Logger:
    loggers: List[Logger] = [ConsoleLogger()]
    if use_wandb:
        loggers.append(WandbLogger(config=config, **wandb_kwargs))
    return CompositeLogger(loggers)
