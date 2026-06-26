from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.ml_pipeline.prepare_selected_event_change_dataset import main


if __name__ == "__main__":
    main()
