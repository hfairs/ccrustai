import os
import sys
from domain import Model
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from pathlib import Path

load_dotenv()


def download_model(model: Model, model_repo: Path) -> str:
    model_path = snapshot_download(
        repo_id=f"{model.author}/{model.model_name}",
        local_dir=f"{model_repo}/{model.author}/{model.model_name}",
    )
    return model_path


def main():
    """Download HuggingFace models in batch."""
    model_repo = Path(os.getenv("MODEL_REPO"))

    models_to_download = [
        Model.FLUX2_KLEIN_4B,
        Model.FLUX2_KLEIN_9B,
        Model.Z_IMAGE_TURBO_COMFY,
    ]
    for model in models_to_download:
        try:
            model_path = download_model(model, model_repo)
            print(f"Model downloaded to: {model_path}\n")
        except KeyboardInterrupt:
            print("\nInterrupted.")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
