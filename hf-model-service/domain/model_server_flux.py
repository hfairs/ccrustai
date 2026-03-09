import torch

from diffusers import FluxTransformer2DModel, FluxPipeline
from optimum.quanto import freeze, qfloat8, quantize
from pathlib import Path
from transformers import T5EncoderModel

from domain.model import Model
from domain.model_server import ModelServer
from util.model_utils import (
    get_model_path,
    get_model_file_path,
    get_filepath_generator,
    FileFormat,
)


class FluxServer(ModelServer):
    """FLUX model server implementation."""

    def __init__(self, filename_generator):
        """Initialize the FLUX.1 server."""
        self._pipeline = None
        self._dtype = torch.bfloat16
        self._filename_generator = filename_generator

    def load_model(self, model: Model, model_repo: Path) -> FluxPipeline:
        model_path = model.get_path(model_repo)

        # Load quantized transformer
        quantized_model_file_path = model.get_file_path(
            model_repo, f"{model}-FP8", "flux1-dev-fp8.safetensors"
        )
        transformer = FluxTransformer2DModel.from_single_file(
            str(quantized_model_file_path),
            torch_dtype=self._dtype,
            local_files_only=False,
        )
        quantize(transformer, weights=qfloat8)
        freeze(transformer)

        # Load quantized text encoder
        text_encoder_2 = T5EncoderModel.from_pretrained(
            str(model_path),
            subfolder="text_encoder_2",
            torch_dtype=self._dtype,
            local_files_only=True,
        )
        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)

        # Create pipeline
        self._pipeline = FluxPipeline.from_pretrained(
            str(model_path),
            transformer=None,
            text_encoder_2=None,
            torch_dtype=self._dtype,
            local_files_only=True,
        )
        self._pipeline.transformer = transformer
        self._pipeline.text_encoder_2 = text_encoder_2
        self._pipeline.enable_model_cpu_offload()

        return self._pipeline

    def invoke(self, prompt: str) -> str:
        if self._pipeline is None:
            raise RuntimeError("Model pipeline not loaded. Call load_model() first.")

        image = self._pipeline(
            prompt,
            guidance_scale=3.5,
            output_type="pil",
            num_inference_steps=20,
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]

        # Generate unique filename with PNG format and save image
        image_path = self._filename_generator(FileFormat.PNG)
        image.save(str(image_path))

        return image_path
