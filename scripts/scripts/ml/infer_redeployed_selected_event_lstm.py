from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ml_pipeline.infer_redeployed_selected_event_lstm import main


if __name__ == "__main__":
    main()
