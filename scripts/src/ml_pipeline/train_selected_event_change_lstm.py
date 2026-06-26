import argparse
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import DATASET_DIR, FIGURE_DIR, MODEL_DIR, REPORT_DIR
from .train_lstm import WaterLevelLSTM, metric_values


SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 160
PATIENCE = 20
LEARNING_RATE = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train selected-event LSTM for future water-level change.")
    parser.add_argument("--sequence-length", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=1)
    return parser.parse_args()


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def load_arrays(sequence_length: int, horizon: int) -> dict[str, np.ndarray]:
    name = f"selected_events_change_seq{sequence_length}_h{horizon}.npz"
    return dict(np.load(DATASET_DIR / name, allow_pickle=True))


def make_loader(x_values: np.ndarray, y_values: np.ndarray) -> DataLoader:
    dataset = TensorDataset(torch.tensor(x_values, dtype=torch.float32), torch.tensor(y_values, dtype=torch.float32))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def predict(model: nn.Module, x_values: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        scaled = model(torch.tensor(x_values, dtype=torch.float32)).numpy()
    return scaled * target_std + target_mean


def split_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for keys, group in rows.groupby(["split", "node_id"], sort=True):
        split, node_id = keys
        lstm = metric_values(group["observed_change"].to_numpy(), group["lstm_prediction"].to_numpy())
        baseline = metric_values(group["observed_change"].to_numpy(), group["baseline_prediction"].to_numpy())
        output.append(
            {
                "split": split,
                "node_id": node_id,
                "rows": len(group),
                "lstm_mae": lstm["mae"],
                "lstm_rmse": lstm["rmse"],
                "lstm_r2": lstm["r2"],
                "baseline_mae": baseline["mae"],
                "baseline_rmse": baseline["rmse"],
                "baseline_r2": baseline["r2"],
                "better_model": "LSTM" if lstm["rmse"] < baseline["rmse"] else "Zero-change baseline",
            }
        )
    for split, group in rows.groupby("split", sort=True):
        lstm = metric_values(group["observed_change"].to_numpy(), group["lstm_prediction"].to_numpy())
        baseline = metric_values(group["observed_change"].to_numpy(), group["baseline_prediction"].to_numpy())
        output.append(
            {
                "split": split,
                "node_id": "Overall",
                "rows": len(group),
                "lstm_mae": lstm["mae"],
                "lstm_rmse": lstm["rmse"],
                "lstm_r2": lstm["r2"],
                "baseline_mae": baseline["mae"],
                "baseline_rmse": baseline["rmse"],
                "baseline_r2": baseline["r2"],
                "better_model": "LSTM" if lstm["rmse"] < baseline["rmse"] else "Zero-change baseline",
            }
        )
    return pd.DataFrame(output)


def main() -> None:
    args = parse_args()
    set_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    arrays = load_arrays(args.sequence_length, args.horizon)
    x_train = arrays["X_train"]
    y_train = arrays["y_train"].astype("float32")
    x_validation = arrays["X_validation"]
    y_validation = arrays["y_validation"].astype("float32")
    x_test = arrays["X_test"]
    y_test = arrays["y_test"].astype("float32")

    target_mean = float(y_train.mean())
    target_std = float(y_train.std() or 1.0)
    y_train_scaled = (y_train - target_mean) / target_std
    y_validation_scaled = (y_validation - target_mean) / target_std

    model = WaterLevelLSTM(input_size=x_train.shape[-1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = make_loader(x_train, y_train_scaled)
    validation_x = torch.tensor(x_validation, dtype=torch.float32)
    validation_y = torch.tensor(y_validation_scaled, dtype=torch.float32)

    best_state = None
    best_validation_loss = float("inf")
    stale_epochs = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(validation_x), validation_y).item())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": validation_loss})
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    records = []
    for split, x_values, y_values in [
        ("train", x_train, y_train),
        ("validation", x_validation, y_validation),
        ("test", x_test, y_test),
    ]:
        predictions = predict(model, x_values, target_mean, target_std)
        records.append(
            pd.DataFrame(
                {
                    "split": split,
                    "node_id": arrays[f"node_id_{split}"],
                    "selected_event_id": arrays[f"selected_event_id_{split}"],
                    "timestamp": arrays[f"timestamp_{split}"].astype(str),
                    "target_timestamp": arrays[f"target_timestamp_{split}"].astype(str),
                    "observed_change": y_values,
                    "lstm_prediction": predictions,
                    "baseline_prediction": np.zeros_like(y_values),
                }
            )
        )
    prediction_df = pd.concat(records, ignore_index=True)
    metrics = split_metrics(prediction_df)
    name = f"selected_event_change_lstm_seq{args.sequence_length}_h{args.horizon}"
    prediction_df.to_csv(REPORT_DIR / f"{name}_predictions.csv", index=False)
    metrics.to_csv(REPORT_DIR / f"{name}_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(REPORT_DIR / f"{name}_history.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "target_mean": target_mean,
            "target_std": target_std,
            "input_size": int(x_train.shape[-1]),
            "sequence_length_hours": args.sequence_length,
            "forecast_horizon_hours": args.horizon,
        },
        MODEL_DIR / f"{name}.pt",
    )

    test_rows = prediction_df.loc[prediction_df["split"] == "test"].copy()
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.scatter(test_rows["observed_change"], test_rows["baseline_prediction"], label="Zero-change baseline", alpha=0.7)
    axis.scatter(test_rows["observed_change"], test_rows["lstm_prediction"], label="LSTM", alpha=0.7)
    limit = max(abs(test_rows["observed_change"]).max(), abs(test_rows["lstm_prediction"]).max(), 1)
    axis.plot([-limit, limit], [-limit, limit], color="black", linestyle="--", linewidth=1)
    axis.set_title(f"Selected-Event Water-Level Change Forecast ({args.horizon}h)")
    axis.set_xlabel("Observed water-level change")
    axis.set_ylabel("Predicted water-level change")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name}_test_scatter.png", dpi=220)
    plt.close(fig)
    print(metrics.loc[metrics["split"] == "test"].to_string(index=False))


if __name__ == "__main__":
    main()
