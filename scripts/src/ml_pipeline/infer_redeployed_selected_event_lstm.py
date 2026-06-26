import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import DATASET_DIR, FIGURE_DIR, MODEL_DIR, REPORT_DIR, REPO_ROOT
from .prepare_selected_event_change_dataset import EVENT_FEATURES
from .train_lstm import WaterLevelLSTM, metric_values


SOURCE = REPO_ROOT / "preprocessed" / "redeployed" / "redeployed_preprocessed.csv"
OUTPUT_PREFIX = "redeployed_selected_event_change_inference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selected-event LSTM inference on redeployed data.")
    parser.add_argument("--sequence-length", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=1)
    return parser.parse_args()


def load_scaler(sequence_length: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    metadata_path = DATASET_DIR / f"selected_events_change_seq{sequence_length}_h{horizon}_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    scaler = metadata["feature_scaler"]
    mean = np.asarray([scaler[column]["mean"] for column in EVENT_FEATURES], dtype=float)
    std = np.asarray([scaler[column]["std"] for column in EVENT_FEATURES], dtype=float)
    std[std == 0] = 1.0
    return mean, std


def load_model(sequence_length: int, horizon: int) -> tuple[WaterLevelLSTM, float, float]:
    checkpoint_path = MODEL_DIR / f"final_event_based_lstm_seq{sequence_length}_h{horizon}.pt"
    if not checkpoint_path.exists():
        checkpoint_path = MODEL_DIR / f"selected_event_change_lstm_seq{sequence_length}_h{horizon}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = WaterLevelLSTM(input_size=int(checkpoint["input_size"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, float(checkpoint["target_mean"]), float(checkpoint["target_std"])


def hourly_node_frame(df: pd.DataFrame, node_id: str) -> pd.DataFrame:
    group = df.loc[df["node_id"] == node_id].copy()
    group = group.loc[group["timestamp"].notna()].sort_values("timestamp")
    hourly = (
        group.set_index("timestamp")
        .resample("1h")
        .agg(
            rain_intensity=("rain_intensity_preprocessed_mm_hr", "mean"),
            canal_water_level=("canal_water_level_preprocessed_cm", "last"),
        )
        .reset_index()
    )
    hourly["node_id"] = node_id
    hourly["rain_intensity"] = hourly["rain_intensity"].fillna(0)
    hourly["canal_water_level"] = hourly["canal_water_level"].ffill().bfill()
    hourly["rain_intensity_change"] = hourly["rain_intensity"].diff().fillna(0)
    hourly["total_rainfall"] = hourly["rain_intensity"]
    hourly["cumulative_total_rainfall"] = hourly["total_rainfall"].cumsum()
    hourly["canal_water_level_change"] = hourly["canal_water_level"].diff().fillna(0)
    hourly["hours_since_event_start"] = (
        hourly["timestamp"] - hourly["timestamp"].min()
    ).dt.total_seconds() / 3600
    hourly["is_selected_event_window"] = 0
    return hourly


def build_sequences(df: pd.DataFrame, sequence_length: int, horizon: int, feature_mean: np.ndarray, feature_std: np.ndarray):
    sequences = []
    rows = []
    for _, group in df.groupby("node_id", sort=True):
        group = group.sort_values("timestamp").reset_index(drop=True)
        group["target_timestamp"] = group["timestamp"].shift(-horizon)
        group["observed_future_level"] = group["canal_water_level"].shift(-horizon)
        group["observed_change"] = group["observed_future_level"] - group["canal_water_level"]
        features = ((group[EVENT_FEATURES].to_numpy(dtype=float) - feature_mean) / feature_std).astype("float32")
        for row_index in range(sequence_length - 1, len(group) - horizon):
            start = row_index - sequence_length + 1
            sequences.append(features[start : row_index + 1])
            rows.append(group.loc[row_index].to_dict())
    return np.asarray(sequences, dtype="float32"), pd.DataFrame(rows)


def predict(model: WaterLevelLSTM, x_values: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    with torch.no_grad():
        scaled = model(torch.tensor(x_values, dtype=torch.float32)).numpy()
    return scaled * target_std + target_mean


def make_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for node_id, group in predictions.groupby("node_id", sort=True):
        lstm = metric_values(group["observed_change"].to_numpy(), group["predicted_change_cm"].to_numpy())
        baseline = metric_values(group["observed_change"].to_numpy(), np.zeros(len(group)))
        rows.append(
            {
                "node_id": node_id,
                "rows": len(group),
                "actual_max_level_cm": float(group["observed_future_level"].max()),
                "predicted_max_level_cm": float(group["predicted_future_level_cm"].max()),
                "actual_max_change_cm": float(group["observed_change"].max()),
                "predicted_max_change_cm": float(group["predicted_change_cm"].max()),
                "lstm_mae": lstm["mae"],
                "lstm_rmse": lstm["rmse"],
                "baseline_mae": baseline["mae"],
                "baseline_rmse": baseline["rmse"],
                "large_false_rise_count": int(((group["predicted_change_cm"] >= 5) & (group["observed_change"] < 2)).sum()),
            }
        )
    return pd.DataFrame(rows)


def make_figure(predictions: pd.DataFrame, sequence_length: int, horizon: int) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    for node_id, group in predictions.groupby("node_id", sort=True):
        axes[0].plot(group["timestamp"], group["rain_intensity"], label=f"{node_id} rain intensity")
        axes[1].plot(group["timestamp"], group["canal_water_level"], label=f"{node_id} current level")
        axes[1].plot(group["target_timestamp"], group["predicted_future_level_cm"], linestyle="--", label=f"{node_id} predicted +{horizon}h")
    axes[0].set_ylabel("Rain Intensity (mm/hr)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_ylabel("Canal Water Level (cm)")
    axes[1].set_xlabel("Timestamp")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle(f"Redeployed Data Inference Using Selected-Event LSTM ({horizon}h Forecast)")
    fig.tight_layout()
    path = FIGURE_DIR / f"{OUTPUT_PREFIX}_seq{sequence_length}_h{horizon}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SOURCE, parse_dates=["timestamp"])
    hourly = pd.concat([hourly_node_frame(df, node_id) for node_id in sorted(df["node_id"].dropna().unique())], ignore_index=True)
    feature_mean, feature_std = load_scaler(args.sequence_length, args.horizon)
    x_values, rows = build_sequences(hourly, args.sequence_length, args.horizon, feature_mean, feature_std)
    model, target_mean, target_std = load_model(args.sequence_length, args.horizon)
    predicted_change = predict(model, x_values, target_mean, target_std)
    predictions = rows.copy()
    predictions["predicted_change_cm"] = predicted_change
    predictions["predicted_future_level_cm"] = predictions["canal_water_level"] + predictions["predicted_change_cm"]
    predictions["baseline_future_level_cm"] = predictions["canal_water_level"]

    name = f"{OUTPUT_PREFIX}_seq{args.sequence_length}_h{args.horizon}"
    predictions.to_csv(REPORT_DIR / f"{name}.csv", index=False)
    make_metrics(predictions).to_csv(REPORT_DIR / f"{name}_summary.csv", index=False)
    hourly.to_csv(REPORT_DIR / "redeployed_hourly_for_lstm_inference.csv", index=False)
    make_figure(predictions, args.sequence_length, args.horizon)
    print(make_metrics(predictions).to_string(index=False))


if __name__ == "__main__":
    main()
