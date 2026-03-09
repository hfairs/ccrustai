import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from domain import ModelServerFactory, Model

load_dotenv()


def main():
    """Interactive inference."""
    model_repo = Path(os.getenv("MODEL_REPO"))
    output_dir = Path(os.getenv("OUTPUT_DIR"))
    model = Model.Z_IMAGE_TURBO
    
    try:
        factory = ModelServerFactory(model_repo, output_dir)
        server = factory.create(model)
        print("✓ Model loaded!\n")
        
        while True:
            prompt = input("Prompt (quit to exit): ").strip()
            if prompt.lower() == "quit":
                break
            if prompt:
                path = server.invoke(prompt)
                print(f"✓ Saved to: {path}\n")
            else:
                print("Empty prompt.\n")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
