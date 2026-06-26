from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PREPROCESSED_DIR = REPO_ROOT / "datasets" / "preprocessed_selected_events"
OUTPUT_DIR = REPO_ROOT / "machine_learning"
DATASET_DIR = OUTPUT_DIR / "datasets"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"
FIGURE_DIR = OUTPUT_DIR / "figures"

NODE_FILES = {
    "Node1": PREPROCESSED_DIR / "node1_hourly.csv",
    "Node2": PREPROCESSED_DIR / "node2_hourly.csv",
}

FEATURE_COLUMNS = [
    "rainfall",
    "rainrate",
    "waterlevel_clean",
    "waterlevel_change",
    "rainrate_change",
    "rainfall_cum_3h",
    "rainrate_cum_3h",
    "rainfall_cum_6h",
    "rainrate_cum_6h",
    "rainfall_cum_12h",
    "rainrate_cum_12h",
    "rainfall_cum_24h",
    "rainrate_cum_24h",
]

TARGET_COLUMN = "waterlevel_clean"
SEQUENCE_LENGTH_HOURS = 12
FORECAST_HORIZON_HOURS = 3
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
