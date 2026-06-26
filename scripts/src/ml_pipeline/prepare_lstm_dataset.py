import json
import argparse

import numpy as np
import pandas as pd

from .config import (
    DATASET_DIR,
    FEATURE_COLUMNS,
    NODE_FILES,
    TARGET_COLUMN,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
)


def split_labels(row_count: int) -> list[str]:
    train_end = int(row_count * TRAIN_FRACTION)
    validation_end = int(row_count * (TRAIN_FRACTION + VALIDATION_FRACTION))
    return [
        "train" if index < train_end else "validation" if index < validation_end else "test"
        for index in range(row_count)
    ]


def fit_scaler(train_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def make_sequences(
    df: pd.DataFrame,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = ((df[FEATURE_COLUMNS].to_numpy(dtype=float) - feature_mean) / feature_std).astype("float32")
    target = df["target_waterlevel"].to_numpy(dtype="float32")
    sequences = []
    targets = []
    for row_index in range(sequence_length - 1, len(df)):
        start = row_index - sequence_length + 1
        sequences.append(features[start : row_index + 1])
        targets.append(target[row_index])
    return np.asarray(sequences, dtype="float32"), np.asarray(targets, dtype="float32")


def prepare_node(node_id: str, path, sequence_length: int, forecast_horizon: int) -> dict[str, object]:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].ffill().bfill().fillna(0)
    df["target_timestamp"] = df["timestamp"].shift(-forecast_horizon)
    df["target_waterlevel"] = df[TARGET_COLUMN].shift(-forecast_horizon)
    df = df.dropna(subset=["target_timestamp", "target_waterlevel"]).reset_index(drop=True)
    df["split"] = split_labels(len(df))
    df["node_id"] = node_id

    train_values = df.loc[df["split"] == "train", FEATURE_COLUMNS].to_numpy(dtype=float)
    feature_mean, feature_std = fit_scaler(train_values)
    x_all, y_all = make_sequences(df, feature_mean, feature_std, sequence_length)
    sequence_rows = df.iloc[sequence_length - 1 :].reset_index(drop=True)

    arrays = {"feature_columns": np.asarray(FEATURE_COLUMNS)}
    for split in ["train", "validation", "test"]:
        mask = sequence_rows["split"].to_numpy() == split
        arrays[f"X_{split}"] = x_all[mask]
        arrays[f"y_{split}"] = y_all[mask]
        arrays[f"timestamp_{split}"] = sequence_rows.loc[mask, "timestamp"].astype(str).to_numpy()
        arrays[f"target_timestamp_{split}"] = sequence_rows.loc[mask, "target_timestamp"].astype(str).to_numpy()

    dataset_name = f"{node_id.lower()}_seq{sequence_length}_h{forecast_horizon}"
    np.savez_compressed(DATASET_DIR / f"{dataset_name}.npz", **arrays)
    sequence_rows.to_csv(DATASET_DIR / f"{dataset_name}_supervised.csv", index=False)

    return {
        "node_id": node_id,
        "source": str(path),
        "rows_after_target_shift": int(len(df)),
        "sequence_count": int(len(sequence_rows)),
        "sequence_length_hours": sequence_length,
        "forecast_horizon_hours": forecast_horizon,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "splits": sequence_rows["split"].value_counts().to_dict(),
        "feature_scaler": {
            column: {"mean": float(mean), "std": float(std)}
            for column, mean, std in zip(FEATURE_COLUMNS, feature_mean, feature_std)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LSTM-ready hydrologic sequence datasets.")
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "purpose": "LSTM-ready supervised hourly datasets for canal water-level forecasting.",
        "leakage_rule": "Only current and historical features are used. Future event peaks and lag labels are excluded.",
        "sequence_length_hours": args.sequence_length,
        "forecast_horizon_hours": args.horizon,
        "nodes": [],
    }
    for node_id, path in NODE_FILES.items():
        metadata["nodes"].append(prepare_node(node_id, path, args.sequence_length, args.horizon))
    (DATASET_DIR / f"lstm_dataset_metadata_seq{args.sequence_length}_h{args.horizon}.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote LSTM-ready datasets to {DATASET_DIR}")


if __name__ == "__main__":
    main()
