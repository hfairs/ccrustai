import torch

from diffusers import ZImagePipeline
from pathlib import Path

from domain.model import Model
from domain.model_server import ModelServer
from util.model_utils import FileFormat, get_model_path


class ZImageServer(ModelServer):
    """Z-Image model server implementation."""

    def __init__(self, filename_generator=None):
        """Initialize the Z-Image server."""
        self._pipeline = None
        self._filename_generator = filename_generator

    def load_model(self, model: Model, model_repo: Path) -> None:
        model_path = model.get_path(model_repo)
        print(f"Loading Z-Image model from: {model_path}")

        # Create pipeline
        # Use bfloat16 for optimal performance on supported GPUs
        self._pipeline = ZImagePipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        # self._pipeline.to("cuda")

        # [Optional] Attention Backend
        # Diffusers uses SDPA by default. Switch to Flash Attention for better efficiency if supported:
        # pipe.transformer.set_attention_backend("flash")    # Enable Flash-Attention-2
        # pipe.transformer.set_attention_backend("_flash_3") # Enable Flash-Attention-3

        # [Optional] Model Compilation
        # Compiling the DiT model accelerates inference, but the first run will take longer to compile.
        # pipe.transformer.compile()

        # [Optional] CPU Offloading
        # Enable CPU offloading for memory-constrained devices.
        self._pipeline.enable_model_cpu_offload()

    def invoke(self, prompt: str) -> str:
        if self._pipeline is None:
            raise RuntimeError("Model pipeline is not loaded. Call load_model() first.")

        # Generate Image
        image = self._pipeline(
            prompt=prompt,
            height=512,
            width=512,
            num_inference_steps=9,  # This actually results in 8 DiT forwards
            guidance_scale=0.0,  # Guidance should be 0 for the Turbo models
            generator=torch.Generator("cuda").manual_seed(42),
        ).images[0]

        filename = self._filename_generator(FileFormat.PNG)
        image.save(str(filename))
        return str(filename)
