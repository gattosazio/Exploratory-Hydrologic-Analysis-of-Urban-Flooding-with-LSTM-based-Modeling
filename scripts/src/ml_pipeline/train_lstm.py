import json
import random
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import DATASET_DIR, FIGURE_DIR, MODEL_DIR, REPORT_DIR


SEED = 42
HIDDEN_SIZE = 32
DROPOUT = 0.20
BATCH_SIZE = 64
MAX_EPOCHS = 120
PATIENCE = 15
LEARNING_RATE = 0.001


class WaterLevelLSTM(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=HIDDEN_SIZE, batch_first=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.output = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence_output, _ = self.lstm(inputs)
        last_hidden = sequence_output[:, -1, :]
        return self.output(self.dropout(last_hidden)).squeeze(-1)


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def load_node_arrays(node_id: str, sequence_length: int, forecast_horizon: int) -> dict[str, np.ndarray]:
    dataset_name = f"{node_id.lower()}_seq{sequence_length}_h{forecast_horizon}.npz"
    return dict(np.load(DATASET_DIR / dataset_name, allow_pickle=True))


def make_loader(x_values: np.ndarray, y_values: np.ndarray, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.tensor(x_values, dtype=torch.float32), torch.tensor(y_values, dtype=torch.float32))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def predict(model: nn.Module, x_values: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        scaled = model(torch.tensor(x_values, dtype=torch.float32)).numpy()
    return scaled * target_std + target_mean


def train_node(node_id: str, sequence_length: int, forecast_horizon: int) -> dict[str, object]:
    arrays = load_node_arrays(node_id, sequence_length, forecast_horizon)
    x_train = arrays["X_train"]
    x_validation = arrays["X_validation"]
    x_test = arrays["X_test"]
    y_train = arrays["y_train"].astype("float32")
    y_validation = arrays["y_validation"].astype("float32")
    y_test = arrays["y_test"].astype("float32")
    target_mean = float(y_train.mean())
    target_std = float(y_train.std() or 1.0)
    y_train_scaled = (y_train - target_mean) / target_std
    y_validation_scaled = (y_validation - target_mean) / target_std

    model = WaterLevelLSTM(input_size=x_train.shape[-1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = make_loader(x_train, y_train_scaled, shuffle=True)
    validation_x = torch.tensor(x_validation, dtype=torch.float32)
    validation_y = torch.tensor(y_validation_scaled, dtype=torch.float32)

    best_state = None
    best_validation_loss = float("inf")
    stale_epochs = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(validation_x), validation_y).item())
        train_loss = float(np.mean(train_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
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

    train_predictions = predict(model, x_train, target_mean, target_std)
    validation_predictions = predict(model, x_validation, target_mean, target_std)
    test_predictions = predict(model, x_test, target_mean, target_std)
    metrics = {
        "train": metric_values(y_train, train_predictions),
        "validation": metric_values(y_validation, validation_predictions),
        "test": metric_values(y_test, test_predictions),
    }

    model_path = MODEL_DIR / f"{node_id.lower()}_lstm_seq{sequence_length}_h{forecast_horizon}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "target_mean": target_mean,
            "target_std": target_std,
            "input_size": int(x_train.shape[-1]),
            "sequence_length_hours": sequence_length,
            "forecast_horizon_hours": forecast_horizon,
        },
        model_path,
    )

    test_df = pd.DataFrame(
        {
            "timestamp": arrays["timestamp_test"].astype(str),
            "target_timestamp": arrays["target_timestamp_test"].astype(str),
            "observed_waterlevel": y_test,
            "predicted_waterlevel": test_predictions,
            "error": y_test - test_predictions,
        }
    )
    test_df.to_csv(REPORT_DIR / f"{node_id.lower()}_seq{sequence_length}_h{forecast_horizon}_lstm_test_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(REPORT_DIR / f"{node_id.lower()}_seq{sequence_length}_h{forecast_horizon}_lstm_training_history.csv", index=False)

    fig, axis = plt.subplots(figsize=(14, 5))
    axis.plot(pd.to_datetime(test_df["target_timestamp"]), test_df["observed_waterlevel"], label="Observed future canal water level", linewidth=1.4)
    axis.plot(pd.to_datetime(test_df["target_timestamp"]), test_df["predicted_waterlevel"], label="LSTM prediction", linewidth=1.2)
    axis.set_title(f"{node_id} LSTM: {forecast_horizon}-Hour Ahead Canal Water-Level Forecast")
    axis.set_ylabel("Canal water level")
    axis.set_xlabel("Target timestamp")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{node_id.lower()}_seq{sequence_length}_h{forecast_horizon}_lstm_test_forecast.png", dpi=220)
    plt.close(fig)

    return {
        "node_id": node_id,
        "model": "lstm",
        "epochs_trained": len(history),
        "best_validation_loss": best_validation_loss,
        "sequence_length_hours": sequence_length,
        "forecast_horizon_hours": forecast_horizon,
        "test_rows": int(len(y_test)),
        "train_mae": metrics["train"]["mae"],
        "train_rmse": metrics["train"]["rmse"],
        "train_r2": metrics["train"]["r2"],
        "validation_mae": metrics["validation"]["mae"],
        "validation_rmse": metrics["validation"]["rmse"],
        "validation_r2": metrics["validation"]["r2"],
        "test_mae": metrics["test"]["mae"],
        "test_rmse": metrics["test"]["rmse"],
        "test_r2": metrics["test"]["r2"],
        "model_path": str(model_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train per-node LSTM water-level forecasting models.")
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        train_node("Node1", args.sequence_length, args.horizon),
        train_node("Node2", args.sequence_length, args.horizon),
    ]
    report = pd.DataFrame(rows)
    report.to_csv(REPORT_DIR / f"lstm_metrics_seq{args.sequence_length}_h{args.horizon}.csv", index=False)
    (REPORT_DIR / f"lstm_metrics_seq{args.sequence_length}_h{args.horizon}.json").write_text(json.dumps(rows, indent=2))
    print(report[["node_id", "epochs_trained", "test_mae", "test_rmse", "test_r2"]].to_string(index=False))


if __name__ == "__main__":
    main()
