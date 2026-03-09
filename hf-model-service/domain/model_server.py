from abc import ABC, abstractmethod
from pathlib import Path

from domain.model import Model


class ModelServer(ABC):
    """Model server interface."""
    
    @abstractmethod
    def load_model(self, model: Model, model_repo: Path) -> None:
        pass
    
    @abstractmethod
    def invoke(self, prompt: str) -> str:
        pass
