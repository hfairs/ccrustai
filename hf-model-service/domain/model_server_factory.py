from pathlib import Path
from domain.model_server import ModelServer
from domain.model import Model
from util.model_utils import get_filepath_generator


class ModelServerFactory:
    """Factory for creating model servers."""
    
    def __init__(self, model_repo: Path, output_dir: Path):
        self._model_repo = model_repo
        self._filename_generator = get_filepath_generator(output_dir)
    
    def create(self, model: Model) -> ModelServer:
        """Create and return a ModelServer."""
        match model:
            case Model.FLUX1_DEV:
                from domain.model_server_flux import FluxServer
                server = FluxServer(self._filename_generator)
                server.load_model(self._model_repo, model.author, model.model_name)
                return server
            case Model.Z_IMAGE_TURBO:
                from domain.model_server_zimage import ZImageServer
                server = ZImageServer(self._filename_generator)
                server.load_model(self._model_repo, model.author, model.model_name)
                return server
            case _:
                raise ValueError(f"Unsupported model: {model}")
        