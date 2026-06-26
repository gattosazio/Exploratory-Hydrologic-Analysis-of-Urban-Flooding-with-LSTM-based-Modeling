from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import FIGURE_DIR, REPORT_DIR, REPO_ROOT


THESIS_DIR = REPO_ROOT / "output" / "thesis_revision"


def bar_labels(axis):
    for container in axis.containers:
        axis.bar_label(container, fmt="%.2f", fontsize=8)


def make_sequence_comparison(comparison: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for axis, horizon in zip(axes, [1, 2, 3]):
        subset = comparison.loc[comparison["forecast_horizon_hours"] == horizon]
        x_values = range(len(subset))
        axis.bar([x - 0.18 for x in x_values], subset["lstm_rmse"], width=0.36, label="LSTM")
        axis.bar([x + 0.18 for x in x_values], subset["baseline_rmse"], width=0.36, label="Zero-change baseline")
        axis.set_xticks(list(x_values))
        axis.set_xticklabels([f"{int(value)}h" for value in subset["sequence_length_hours"]])
        axis.set_title(f"{horizon}h Forecast")
        axis.set_xlabel("Input Sequence Length")
        axis.grid(axis="y", alpha=0.25)
        bar_labels(axis)
    axes[0].set_ylabel("RMSE (cm)")
    axes[0].legend()
    fig.suptitle("Sequence-Length Evaluation of the Event-Based LSTM Forecasting Model")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "final_lstm_sequence_horizon_rmse_comparison.png", dpi=220)
    plt.close(fig)


def make_best_horizon_figure(best: pd.DataFrame):
    fig, axis = plt.subplots(figsize=(10, 6))
    labels = [f"{int(row.forecast_horizon_hours)}h" for row in best.itertuples()]
    x_values = range(len(best))
    axis.bar([x - 0.18 for x in x_values], best["lstm_rmse"], width=0.36, label="LSTM")
    axis.bar([x + 0.18 for x in x_values], best["baseline_rmse"], width=0.36, label="Zero-change baseline")
    axis.set_xticks(list(x_values))
    axis.set_xticklabels(labels)
    axis.set_ylabel("RMSE (cm)")
    axis.set_xlabel("Forecast Horizon")
    axis.set_title("Final Event-Based LSTM Performance Compared with Zero-Change Baseline")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    bar_labels(axis)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "final_lstm_best_horizon_rmse.png", dpi=220)
    plt.close(fig)


def make_workflow_figure():
    fig, axis = plt.subplots(figsize=(13, 4))
    axis.axis("off")
    boxes = [
        ("Preprocessed\nSelected Events", 0.06),
        ("Separate by\nNode and Event", 0.27),
        ("Create 12h\nSequences", 0.47),
        ("Train Shared\nLSTM Model", 0.67),
        ("Forecast Water-\nLevel Change", 0.87),
    ]
    for text, x_value in boxes:
        axis.text(
            x_value,
            0.5,
            text,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f7f7f7", "edgecolor": "black"},
        )
    for start, end in zip(boxes[:-1], boxes[1:]):
        axis.annotate("", xy=(end[1] - 0.09, 0.5), xytext=(start[1] + 0.09, 0.5), arrowprops={"arrowstyle": "->", "lw": 1.5})
    axis.set_title("Methodological Workflow for Event-Based LSTM Forecasting")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "final_lstm_workflow.png", dpi=220)
    plt.close(fig)


def make_single_event_prediction_figure():
    predictions = pd.read_csv(REPORT_DIR / "selected_event_change_lstm_seq12_h1_predictions.csv", parse_dates=["timestamp", "target_timestamp"])
    supervised = pd.read_csv(
        REPO_ROOT / "output" / "ml" / "datasets" / "selected_events_change_seq12_h1_supervised.csv",
        parse_dates=["timestamp", "target_timestamp"],
    )
    merged = predictions.merge(
        supervised[
            [
                "node_id",
                "selected_event_id",
                "timestamp",
                "target_timestamp",
                "rain_intensity",
                "canal_water_level",
            ]
        ],
        on=["node_id", "selected_event_id", "timestamp", "target_timestamp"],
        how="left",
    )
    event = merged.loc[(merged["split"] == "test") & (merged["selected_event_id"] == 10)].copy()
    event["observed_future_level"] = event["canal_water_level"] + event["observed_change"]
    event["predicted_future_level"] = event["canal_water_level"] + event["lstm_prediction"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for node_id, group in event.groupby("node_id", sort=True):
        axes[0].plot(group["timestamp"], group["rain_intensity"], marker="o", label=f"{node_id} rain intensity")
        axes[1].plot(group["target_timestamp"], group["observed_future_level"], marker="o", label=f"{node_id} observed")
        axes[1].plot(group["target_timestamp"], group["predicted_future_level"], marker="x", linestyle="--", label=f"{node_id} LSTM forecast")
    axes[0].set_ylabel("Rain Intensity (mm/hr)")
    axes[0].set_title("Rainfall Forcing During Selected Event 10")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_ylabel("Canal Water Level (cm)")
    axes[1].set_xlabel("Timestamp")
    axes[1].set_title("Observed and 1-Hour Forecast Canal Water Level During Selected Event 10")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle("Single-Event LSTM Forecast Example for Selected Rainfall Event 10")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "final_lstm_single_event_prediction_example.png", dpi=220)
    plt.close(fig)


def write_markdown(best: pd.DataFrame, comparison: pd.DataFrame):
    best_rows = "\n".join(
        [
            f"| {int(row.forecast_horizon_hours)} h | {int(row.sequence_length_hours)} h | {row.rows} | {row.lstm_rmse:.2f} | {row.baseline_rmse:.2f} | {row.rmse_percent_improvement:.1f}% |"
            for row in best.itertuples()
        ]
    )
    text = f"""# LSTM Model Development and Results

## 4.X Event-Based LSTM Dataset Preparation

The machine-learning dataset was prepared from the preprocessed selected rainfall events rather than from the full raw dataset. This was done because the full monitoring record contains long stable periods, while the research objective requires modeling rainfall-response behavior during hydrologically meaningful events.

Each selected event was first separated by monitoring node. Therefore, the ten selected events produced node-specific event series for Node 1 and Node 2. These node-event series were then pooled into one shared LSTM dataset. The model was not trained as a separate Node 1 model and Node 2 model in the final setup. Instead, one event-based model was trained using both nodes so it could learn a general rainfall-response pattern while still receiving node-specific time-series behavior through the input variables.

## 4.X Input Sequence Methodology

The LSTM input window refers to the number of past hours used by the model before making a forecast. For example, a 12-hour input window means that the model reads the previous 12 hours of rainfall and water-level behavior before predicting the future water-level change. If the forecast horizon is 1 hour, the model uses the previous 12 hours to estimate how much the canal water level will change 1 hour later.

The model input variables were rain intensity, rain-intensity change, total rainfall, cumulative rainfall, canal water level, canal water-level change, hours since event start, and the selected-event window flag. The output variable was future canal water-level change. Peak-to-peak lag was not used as a direct input because it is calculated after an event is complete. Instead, lag behavior was represented through the historical sequence window.

Events 1-6 were used for training, Events 7-8 for validation, and Events 9-10 for testing.

## 5.X LSTM Model Improvement and Evaluation

The LSTM was improved by testing several input sequence lengths: 3 hours, 6 hours, 9 hours, and 12 hours. The 12-hour sequence length produced the lowest test RMSE across the 1-hour, 2-hour, and 3-hour forecast horizons.

| Forecast horizon | Best sequence length | Test rows | LSTM RMSE | Baseline RMSE | RMSE improvement |
|---:|---:|---:|---:|---:|---:|
{best_rows}

## 5.X Interpretation of Forecasting Results

The improved 12-hour selected-event LSTM outperformed the zero-change baseline for all tested forecast horizons. This supports the use of an event-based LSTM for forecasting canal water-level change during selected rainfall-response periods. However, because the model was trained on only ten selected events, the result should be presented as a prototype event-based forecasting model rather than a fully generalized operational model.

The redeployed dry/light-rain validation showed that the LSTM should not be used continuously during dry periods. For dashboard deployment, the system should first detect whether rainfall-event conditions are present. If no rainfall event is detected, the dashboard should use conservative baseline logic. If a rainfall event is detected, the LSTM can be activated to forecast future water-level change.

## Recommended Thesis Figures

1. `output/ml/figures/final_lstm_workflow.png` - event-based LSTM workflow.
2. `output/ml/figures/final_lstm_sequence_horizon_rmse_comparison.png` - sequence-length tuning results.
3. `output/ml/figures/final_lstm_best_horizon_rmse.png` - final LSTM versus baseline performance.
4. `output/ml/figures/final_lstm_single_event_prediction_example.png` - observed and predicted water-level response for one selected event.
5. `output/ml/figures/redeployed_selected_event_change_inference_seq12_h1.png` - redeployed validation example.
"""
    THESIS_DIR.mkdir(parents=True, exist_ok=True)
    (THESIS_DIR / "machine_learning_lstm_results_discussion.md").write_text(text, encoding="utf-8")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(REPORT_DIR / "selected_event_change_lstm_sequence_horizon_comparison.csv")
    best = pd.read_csv(REPORT_DIR / "selected_event_change_lstm_best_by_horizon.csv")
    make_sequence_comparison(comparison)
    make_best_horizon_figure(best)
    make_workflow_figure()
    make_single_event_prediction_figure()
    write_markdown(best, comparison)
    inventory = pd.DataFrame(
        [
            {
                "figure": "final_lstm_workflow.png",
                "purpose": "Explains the event-based LSTM data flow.",
                "paper_section": "Chapter 4 Machine Learning Methodology",
            },
            {
                "figure": "final_lstm_sequence_horizon_rmse_comparison.png",
                "purpose": "Shows sequence-length tuning across forecast horizons.",
                "paper_section": "Chapter 5 Model Evaluation",
            },
            {
                "figure": "final_lstm_best_horizon_rmse.png",
                "purpose": "Compares final LSTM performance with the zero-change baseline.",
                "paper_section": "Chapter 5 Model Evaluation",
            },
            {
                "figure": "final_lstm_single_event_prediction_example.png",
                "purpose": "Shows one selected event with observed and LSTM-predicted canal water level.",
                "paper_section": "Chapter 5 Model Evaluation",
            },
            {
                "figure": "redeployed_selected_event_change_inference_seq12_h1.png",
                "purpose": "Shows dry/light-rain deployment validation and supports gated LSTM activation.",
                "paper_section": "Chapter 5 Deployment Validation",
            },
        ]
    )
    inventory.to_csv(REPORT_DIR / "final_ml_figure_inventory.csv", index=False)
    print(inventory.to_string(index=False))


if __name__ == "__main__":
    main()
