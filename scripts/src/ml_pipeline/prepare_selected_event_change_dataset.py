import argparse
import json

import numpy as np
import pandas as pd

from .config import DATASET_DIR, REPO_ROOT


SOURCE = (
    REPO_ROOT
    / "output"
    / "figure_data"
    / "preprocessed-selected-events"
    / "normalized_rain_intensity"
    / "normalized_selected_events_all_merged_nodes.csv"
)

EVENT_FEATURES = [
    "rain_intensity",
    "rain_intensity_change",
    "total_rainfall",
    "cumulative_total_rainfall",
    "canal_water_level",
    "canal_water_level_change",
    "hours_since_event_start",
    "is_selected_event_window",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare selected-event water-level-change LSTM dataset.")
    parser.add_argument("--sequence-length", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=1)
    return parser.parse_args()


def node_frame(df: pd.DataFrame, node_id: str) -> pd.DataFrame:
    prefix = node_id.lower()
    result = pd.DataFrame(
        {
            "node_id": node_id,
            "selected_event_id": df["selected_event_id"],
            "original_shared_event_id": df["original_shared_event_id"],
            "timestamp": df["timestamp"],
            "event_start": df["event_start"],
            "event_end": df["event_end"],
            "is_selected_event_window": df["is_selected_event_window"].astype(int),
            "rain_intensity": df[f"{prefix}_rain_intensity_normalized"],
            "rain_intensity_change": df[f"{prefix}_rain_intensity_change"],
            "total_rainfall": df[f"{prefix}_total_rainfall"],
            "cumulative_total_rainfall": df[f"{prefix}_cumulative_total_rainfall"],
            "canal_water_level": df[f"{prefix}_canal_water_level_preprocessed"],
            "canal_water_level_change": df[f"{prefix}_canal_water_level_preprocessed_change"],
        }
    )
    result["hours_since_event_start"] = (
        pd.to_datetime(result["timestamp"]) - pd.to_datetime(result["event_start"])
    ).dt.total_seconds() / 3600
    return result


def split_name(event_id: int) -> str:
    if event_id in {9, 10}:
        return "test"
    if event_id in {7, 8}:
        return "validation"
    return "train"


def fit_scaler(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def build_sequences(df: pd.DataFrame, sequence_length: int, horizon: int, feature_mean: np.ndarray, feature_std: np.ndarray):
    sequences = []
    targets = []
    rows = []
    for _, event_df in df.groupby(["node_id", "selected_event_id"], sort=True):
        event_df = event_df.sort_values("timestamp").reset_index(drop=True)
        event_df["target_timestamp"] = event_df["timestamp"].shift(-horizon)
        event_df["target_change"] = event_df["canal_water_level"].shift(-horizon) - event_df["canal_water_level"]
        features = ((event_df[EVENT_FEATURES].to_numpy(dtype=float) - feature_mean) / feature_std).astype("float32")
        for row_index in range(sequence_length - 1, len(event_df) - horizon):
            start = row_index - sequence_length + 1
            sequences.append(features[start : row_index + 1])
            targets.append(float(event_df.loc[row_index, "target_change"]))
            rows.append(event_df.loc[row_index].to_dict())
    return np.asarray(sequences, dtype="float32"), np.asarray(targets, dtype="float32"), pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(SOURCE, parse_dates=["timestamp", "event_start", "event_end"])
    stacked = pd.concat([node_frame(raw, "Node1"), node_frame(raw, "Node2")], ignore_index=True)
    stacked[EVENT_FEATURES] = stacked[EVENT_FEATURES].ffill().bfill().fillna(0)
    stacked["split"] = stacked["selected_event_id"].apply(split_name)
    train_values = stacked.loc[stacked["split"] == "train", EVENT_FEATURES].to_numpy(dtype=float)
    feature_mean, feature_std = fit_scaler(train_values)
    x_all, y_all, rows = build_sequences(stacked, args.sequence_length, args.horizon, feature_mean, feature_std)
    arrays = {"feature_columns": np.asarray(EVENT_FEATURES)}
    for split in ["train", "validation", "test"]:
        mask = rows["split"].to_numpy() == split
        arrays[f"X_{split}"] = x_all[mask]
        arrays[f"y_{split}"] = y_all[mask]
        arrays[f"timestamp_{split}"] = rows.loc[mask, "timestamp"].astype(str).to_numpy()
        arrays[f"target_timestamp_{split}"] = rows.loc[mask, "target_timestamp"].astype(str).to_numpy()
        arrays[f"node_id_{split}"] = rows.loc[mask, "node_id"].to_numpy()
        arrays[f"selected_event_id_{split}"] = rows.loc[mask, "selected_event_id"].to_numpy()
    dataset_name = f"selected_events_change_seq{args.sequence_length}_h{args.horizon}"
    np.savez_compressed(DATASET_DIR / f"{dataset_name}.npz", **arrays)
    rows.to_csv(DATASET_DIR / f"{dataset_name}_supervised.csv", index=False)
    metadata = {
        "purpose": "Selected-event LSTM dataset for future canal water-level change prediction.",
        "sequence_length_hours": args.sequence_length,
        "forecast_horizon_hours": args.horizon,
        "target": "future canal_water_level - current canal_water_level",
        "features": EVENT_FEATURES,
        "split_rule": "Events 1-6 train, events 7-8 validation, events 9-10 test.",
        "counts": rows["split"].value_counts().to_dict(),
        "feature_scaler": {
            column: {"mean": float(mean), "std": float(std)}
            for column, mean, std in zip(EVENT_FEATURES, feature_mean, feature_std)
        },
    }
    (DATASET_DIR / f"{dataset_name}_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {dataset_name}: {rows['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
