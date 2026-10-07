"""Run paired-image foreign-object inference without installing the package."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from foreign_object_detection.cli_detect import main

if __name__ == "__main__":
    sys.exit(main())