# HF Model Service

Interactive inference service for HuggingFace models.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install torch torchvision torchaudio --torch-backend=auto
uv pip install -e .
```

## Project Structure

```
├── domain/              # Model servers & factory
│   ├── model.py        # Model enum (FLUX1_DEV, Z_IMAGE_TURBO)
│   ├── model_server.py # Abstract interface
│   └── model_server_*  # Implementations
├── util/               # Utilities
├── inference.py        # Entry point
```

## Usage

```bash
export MODEL_REPO=/path/to/models
export OUTPUT_DIR=/path/to/outputs
python inference.py
```

Supported models:
- `Model.FLUX1_DEV` (black-forest-labs/FLUX.1-dev)
- `Model.Z_IMAGE_TURBO` (Tongyi-MAI/Z-Image-Turbo)
