import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import DATASET_DIR, FEATURE_COLUMNS, FIGURE_DIR, MODEL_DIR, REPORT_DIR, REPO_ROOT
from .train_lstm import WaterLevelLSTM, metric_values


SELECTED_EVENTS = (
    REPO_ROOT
    / "output"
    / "figure_data"
    / "preprocessed-selected-events"
    / "merged_nodes"
    / "preprocessed_selected_events_all_merged_nodes.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LSTM and persistence on selected rainfall-event windows.")
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=1)
    return parser.parse_args()


def event_windows() -> pd.DataFrame:
    events = pd.read_csv(SELECTED_EVENTS, parse_dates=["plot_start", "plot_end", "event_start", "event_end"])
    return (
        events[
            [
                "selected_event_id",
                "original_shared_event_id",
                "plot_start",
                "plot_end",
                "event_start",
                "event_end",
            ]
        ]
        .drop_duplicates()
        .sort_values("selected_event_id")
        .reset_index(drop=True)
    )


def load_metadata(sequence_length: int, horizon: int) -> dict:
    path = DATASET_DIR / f"lstm_dataset_metadata_seq{sequence_length}_h{horizon}.json"
    return json.loads(path.read_text())


def current_waterlevel_from_sequences(
    x_values: np.ndarray,
    metadata: dict,
    node_id: str,
) -> np.ndarray:
    node_meta = next(node for node in metadata["nodes"] if node["node_id"] == node_id)
    feature_index = FEATURE_COLUMNS.index("waterlevel_clean")
    scaler = node_meta["feature_scaler"]["waterlevel_clean"]
    scaled_current = x_values[:, -1, feature_index]
    return scaled_current * scaler["std"] + scaler["mean"]


def load_model(node_id: str, sequence_length: int, horizon: int) -> tuple[WaterLevelLSTM, float, float]:
    checkpoint = torch.load(
        MODEL_DIR / f"{node_id.lower()}_lstm_seq{sequence_length}_h{horizon}.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = WaterLevelLSTM(input_size=checkpoint["input_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, float(checkpoint["target_mean"]), float(checkpoint["target_std"])


def predict(model: WaterLevelLSTM, x_values: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    with torch.no_grad():
        scaled = model(torch.tensor(x_values, dtype=torch.float32)).numpy()
    return scaled * target_std + target_mean


def all_split_predictions(node_id: str, sequence_length: int, horizon: int, metadata: dict) -> pd.DataFrame:
    arrays = dict(np.load(DATASET_DIR / f"{node_id.lower()}_seq{sequence_length}_h{horizon}.npz", allow_pickle=True))
    model, target_mean, target_std = load_model(node_id, sequence_length, horizon)
    rows = []
    for split in ["train", "validation", "test"]:
        x_values = arrays[f"X_{split}"]
        y_values = arrays[f"y_{split}"].astype(float)
        lstm_prediction = predict(model, x_values, target_mean, target_std)
        persistence_prediction = current_waterlevel_from_sequences(x_values, metadata, node_id)
        rows.append(
            pd.DataFrame(
                {
                    "node_id": node_id,
                    "split": split,
                    "timestamp": pd.to_datetime(arrays[f"timestamp_{split}"].astype(str)),
                    "target_timestamp": pd.to_datetime(arrays[f"target_timestamp_{split}"].astype(str)),
                    "observed_waterlevel": y_values,
                    "lstm_prediction": lstm_prediction,
                    "persistence_prediction": persistence_prediction,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def assign_events(predictions: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    matched = []
    for _, event in windows.iterrows():
        event_rows = predictions.loc[
            predictions["target_timestamp"].between(event["plot_start"], event["plot_end"], inclusive="both")
        ].copy()
        if event_rows.empty:
            continue
        for column in [
            "selected_event_id",
            "original_shared_event_id",
            "plot_start",
            "plot_end",
            "event_start",
            "event_end",
        ]:
            event_rows[column] = event[column]
        event_rows["is_core_event_window"] = event_rows["target_timestamp"].between(
            event["event_start"],
            event["event_end"],
            inclusive="both",
        )
        matched.append(event_rows)
    if not matched:
        return pd.DataFrame()
    return pd.concat(matched, ignore_index=True)


def summarize(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in events.groupby(["node_id", "selected_event_id", "original_shared_event_id"], sort=True):
        node_id, selected_event_id, original_shared_event_id = keys
        lstm_metrics = metric_values(group["observed_waterlevel"].to_numpy(), group["lstm_prediction"].to_numpy())
        persistence_metrics = metric_values(
            group["observed_waterlevel"].to_numpy(),
            group["persistence_prediction"].to_numpy(),
        )
        split_counts = group["split"].value_counts().to_dict()
        rows.append(
            {
                "node_id": node_id,
                "selected_event_id": selected_event_id,
                "original_shared_event_id": original_shared_event_id,
                "rows": len(group),
                "train_rows": split_counts.get("train", 0),
                "validation_rows": split_counts.get("validation", 0),
                "test_rows": split_counts.get("test", 0),
                "lstm_mae": lstm_metrics["mae"],
                "lstm_rmse": lstm_metrics["rmse"],
                "lstm_r2": lstm_metrics["r2"],
                "persistence_mae": persistence_metrics["mae"],
                "persistence_rmse": persistence_metrics["rmse"],
                "persistence_r2": persistence_metrics["r2"],
                "better_model": "LSTM" if lstm_metrics["rmse"] < persistence_metrics["rmse"] else "Persistence",
            }
        )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, sequence_length: int, horizon: int) -> None:
    aggregate = (
        summary.groupby("node_id")[["lstm_rmse", "persistence_rmse"]]
        .mean()
        .reindex(["Node1", "Node2"])
    )
    fig, axis = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(aggregate.index))
    axis.bar(positions - 0.18, aggregate["persistence_rmse"], width=0.36, label="Persistence")
    axis.bar(positions + 0.18, aggregate["lstm_rmse"], width=0.36, label="LSTM")
    axis.set_xticks(positions)
    axis.set_xticklabels([node.replace("Node", "Node ") for node in aggregate.index])
    axis.set_ylabel("Mean selected-event RMSE")
    axis.set_title(f"Selected-Event Evaluation: {sequence_length}h Input, {horizon}h Forecast")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"selected_event_eval_seq{sequence_length}_h{horizon}_rmse.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.sequence_length, args.horizon)
    windows = event_windows()
    predictions = pd.concat(
        [
            all_split_predictions("Node1", args.sequence_length, args.horizon, metadata),
            all_split_predictions("Node2", args.sequence_length, args.horizon, metadata),
        ],
        ignore_index=True,
    )
    selected_predictions = assign_events(predictions, windows)
    summary = summarize(selected_predictions)
    selected_predictions.to_csv(
        REPORT_DIR / f"selected_event_predictions_seq{args.sequence_length}_h{args.horizon}.csv",
        index=False,
    )
    summary.to_csv(
        REPORT_DIR / f"selected_event_eval_seq{args.sequence_length}_h{args.horizon}.csv",
        index=False,
    )
    plot_summary(summary, args.sequence_length, args.horizon)
    print(summary.groupby(["node_id", "better_model"]).size().to_string())


if __name__ == "__main__":
    main()
