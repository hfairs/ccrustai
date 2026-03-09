from pathlib import Path
from datetime import datetime
from enum import Enum


class FileFormat(Enum):
    """File formats."""

    PNG = ".png"
    JPG = ".jpg"
    JPEG = ".jpeg"
    WEBP = ".webp"


def get_filepath_generator(output_dir: Path = None):
    """Create filepath generator (format: yyyymmdd-ddd)."""
    output_dir = Path(output_dir or Path.cwd())

    def generate(file_format: FileFormat) -> Path:
        date_prefix = datetime.now().strftime("%Y%m%d")
        sequence = 1
        while True:
            filename = f"{date_prefix}-{sequence:03d}"
            if not list(output_dir.glob(f"{filename}.*")):
                return output_dir / f"{filename}{file_format.value}"
            sequence += 1

    return generate
