import pandas as pd

from hydrologic_pipeline.config import PREPROCESSED_DIR


def main() -> None:
    node_paths = [
        PREPROCESSED_DIR / "node1_hourly.csv",
        PREPROCESSED_DIR / "node2_hourly.csv",
    ]
    missing = [str(path) for path in node_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required hourly inputs: "
            + ", ".join(missing)
            + ". Run src/run_preprocessing_pipeline.py first."
        )

    print("Descriptive analysis input ready")
    for path in node_paths:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        node_id = path.stem.replace("_hourly", "").upper()
        summary = df[["rainfall", "rainrate", "waterlevel_clean"]].agg(["mean", "median"]).round(2)
        print(node_id)
        print(summary)


if __name__ == "__main__":
    main()
