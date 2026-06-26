from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ml_pipeline.generate_final_ml_outputs import main


if __name__ == "__main__":
    main()
