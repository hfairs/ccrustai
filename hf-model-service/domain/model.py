from enum import Enum
from pathlib import Path


class Model(Enum):
    """Available models."""
    FLUX1_DEV = ("black-forest-labs", "FLUX.1-dev")
    FLUX2_KLEIN_4B= ("black-forest-labs", "FLUX.2-klein-4B")
    FLUX2_KLEIN_9B= ("black-forest-labs", "FLUX.2-klein-9B")
    Z_IMAGE_TURBO = ("Tongyi-MAI", "Z-Image-Turbo")
    Z_IMAGE_TURBO_COMFY = ("Comfy-Org", "z_image_turbo")

    def __init__(self, author: str, name: str):
        self.author = author
        self.model_name = name
    
    def get_path(self, model_repo: Path) -> Path:
        return model_repo / self.author / self.model_name
    
    def get_file_path(self, model_repo: Path, file_name: str) -> Path:
        return model_repo / self.author / self.model_name / file_name
