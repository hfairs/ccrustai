from pathlib import Path
from domain.model import Model
from domain.model_server import ModelServer
from domain.model_server_flux import FluxServer
from domain.model_server_zimage import ZImageServer
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

                server = FluxServer(self._filename_generator)
                server.load_model(model, self._model_repo)
                return server
            case Model.Z_IMAGE_TURBO:

                server = ZImageServer(self._filename_generator)
                server.load_model(model, self._model_repo)
                return server
            case _:
                raise ValueError(f"Unsupported model: {model}")
