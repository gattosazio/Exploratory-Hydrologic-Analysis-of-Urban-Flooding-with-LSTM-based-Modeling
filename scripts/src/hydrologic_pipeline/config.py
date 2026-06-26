from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_JSON_PATH = REPO_ROOT / "datasets" / "raw" / "for_hydrologic_analysis.json"
PREPROCESSED_DIR = REPO_ROOT / "datasets" / "preprocessed_selected_events"
OUTPUT_DIR = REPO_ROOT
REPORTS_DIR = OUTPUT_DIR / "reports"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURE_DATA_DIR = OUTPUT_DIR / "figure_data"
DOCS_DIR = REPO_ROOT / "docs"
DATASETS_DIR = REPO_ROOT / "datasets"
LEGACY_PREPROCESSED_DIR = DATASETS_DIR / "preprocessed"
LEGACY_CLEANED_DIR = DATASETS_DIR / "cleaned"
LEGACY_RAW_DIR = DATASETS_DIR / "raw"

HOURLY_FREQUENCY = "1h"
LAG_WINDOW = 100
ROLLING_WINDOWS_HOURS = [3, 6, 12, 24]
PRE_EVENT_HOURS = 6
POST_EVENT_HOURS = 12
POST_EVENT_RESPONSE_HOURS = 12
MIN_EVENT_DURATION_HOURS = 2
MAX_EVENT_COLLAGE_COUNT = 9
ANALYSIS_START = "2025-12-01 00:00:00"
ANALYSIS_END = "2026-01-20 23:59:59"

CORE_COLUMNS = ["Timestamp", "Rainfall", "RainRate", "WaterLevel"]
