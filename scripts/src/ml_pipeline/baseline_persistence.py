import json
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    DATASET_DIR,
    FIGURE_DIR,
    REPORT_DIR,
)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def evaluate_node(node_id: str, sequence_length: int, forecast_horizon: int) -> dict[str, object]:
    dataset_name = f"{node_id.lower()}_seq{sequence_length}_h{forecast_horizon}_supervised.csv"
    df = pd.read_csv(DATASET_DIR / dataset_name, parse_dates=["timestamp", "target_timestamp"])
    test = df.loc[df["split"] == "test"].copy()
    test["prediction_persistence"] = test["waterlevel_clean"]
    node_metrics = metrics(
        test["target_waterlevel"].to_numpy(dtype=float),
        test["prediction_persistence"].to_numpy(dtype=float),
    )
    test.to_csv(REPORT_DIR / f"{node_id.lower()}_seq{sequence_length}_h{forecast_horizon}_persistence_test_predictions.csv", index=False)

    fig, axis = plt.subplots(figsize=(14, 5))
    axis.plot(test["target_timestamp"], test["target_waterlevel"], label="Observed future canal water level", linewidth=1.4)
    axis.plot(test["target_timestamp"], test["prediction_persistence"], label="Persistence baseline", linewidth=1.2)
    axis.set_title(f"{node_id} Persistence Baseline: {forecast_horizon}-Hour Ahead Canal Water-Level Forecast")
    axis.set_ylabel("Canal water level")
    axis.set_xlabel("Target timestamp")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{node_id.lower()}_seq{sequence_length}_h{forecast_horizon}_persistence_baseline.png", dpi=220)
    plt.close(fig)

    return {
        "node_id": node_id,
        "baseline": "persistence_current_waterlevel",
        "test_rows": int(len(test)),
        "sequence_length_hours": sequence_length,
        "forecast_horizon_hours": forecast_horizon,
        **node_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate persistence baseline for LSTM-ready datasets.")
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        evaluate_node("Node1", args.sequence_length, args.horizon),
        evaluate_node("Node2", args.sequence_length, args.horizon),
    ]
    report = pd.DataFrame(rows)
    report.to_csv(REPORT_DIR / f"persistence_baseline_metrics_seq{args.sequence_length}_h{args.horizon}.csv", index=False)
    (REPORT_DIR / f"persistence_baseline_metrics_seq{args.sequence_length}_h{args.horizon}.json").write_text(json.dumps(rows, indent=2))
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
