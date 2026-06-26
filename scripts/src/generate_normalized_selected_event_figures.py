from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    REPO_ROOT
    / "datasets"
    / "preprocessed_selected_events"
    / "merged_nodes"
    / "preprocessed_selected_events_all_merged_nodes.csv"
)
FIGURE_DIR = (
    REPO_ROOT
    / "figures"
    / "preprocessed-selected-events"
    / "normalized_rain_intensity"
)
DATA_DIR = (
    REPO_ROOT
    / "figure_data"
    / "preprocessed-selected-events"
    / "normalized_rain_intensity"
)


def normalize_intensity(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    result = df.copy()
    caps = {}
    # PAGASA heavy rain classification threshold: >= 7.5 mm/hr
    heavy_rain_threshold = 7.5
    for node in ["node1", "node2"]:
        source_col = f"{node}_rain_intensity"
        target_col = f"{node}_rain_intensity_normalized"
        cap = heavy_rain_threshold
        caps[node] = cap
        result[target_col] = result[source_col].clip(lower=0, upper=cap)
    
    # Cap water level at 55 cm for both nodes
    water_level_cap = 55.0
    result["node1_canal_water_level_preprocessed"] = result["node1_canal_water_level_preprocessed"].clip(upper=water_level_cap)
    result["node2_canal_water_level_preprocessed"] = result["node2_canal_water_level_preprocessed"].clip(upper=water_level_cap)
    
    return result, caps


def plot_combined_collage(df: pd.DataFrame, output_path: Path) -> None:
    events = [event_df.reset_index(drop=True) for _, event_df in df.groupby("selected_event_id", sort=True)]
    rows = int(np.ceil(len(events) / 2))
    figure, axes = plt.subplots(rows, 2, figsize=(20, 4.8 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, event_df in zip(axes.ravel(), events):
        selected_event_id = int(event_df["selected_event_id"].iloc[0])
        original_event_id = int(event_df["original_shared_event_id"].iloc[0])
        event_start = pd.to_datetime(event_df["event_start"].iloc[0])
        event_end = pd.to_datetime(event_df["event_end"].iloc[0])
        node1_lag = event_df["node1_lag_peak_rain_intensity_to_peak_water_level_h"].iloc[0]
        node2_lag = event_df["node2_lag_peak_rain_intensity_to_peak_water_level_h"].iloc[0]
        axis.axis("on")
        water_axis = axis.twinx()
        axis.plot(event_df["timestamp"], event_df["node1_rain_intensity_normalized"], color="tab:purple", linewidth=1.2, label="N1 intensity")
        axis.plot(event_df["timestamp"], event_df["node2_rain_intensity_normalized"], color="tab:red", linewidth=1.2, label="N2 intensity")
        water_axis.plot(event_df["timestamp"], event_df["node1_canal_water_level_preprocessed"], color="tab:green", linewidth=1.5, label="N1 water")
        water_axis.plot(event_df["timestamp"], event_df["node2_canal_water_level_preprocessed"], color="tab:orange", linewidth=1.5, label="N2 water")
        axis.axvline(event_start, color="black", linestyle=":", linewidth=0.9)
        axis.axvline(event_end, color="black", linestyle=":", linewidth=0.9)
        axis.set_title(f"Selected {selected_event_id} (shared {original_event_id}) | lag N1/N2 {node1_lag:.1f}h/{node2_lag:.1f}h", fontsize=10)
        axis.set_ylabel("Rain Intensity (mm/hr)")
        water_axis.set_ylabel("Water level")
        axis.tick_params(axis="x", rotation=30, labelsize=8)
        axis.grid(True, alpha=0.25)
        lines, labels = axis.get_legend_handles_labels()
        water_lines, water_labels = water_axis.get_legend_handles_labels()
        axis.legend(lines + water_lines, labels + water_labels, loc="upper left", fontsize=8, ncol=2, framealpha=0.85)
    figure.suptitle("Merged Node Selected Events: Rain Intensity (mm/hr) and Canal Water Level", fontsize=15)
    figure.tight_layout(rect=[0, 0, 1, 0.97])
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_intensity_collage(df: pd.DataFrame, output_path: Path) -> None:
    events = [event_df.reset_index(drop=True) for _, event_df in df.groupby("selected_event_id", sort=True)]
    rows = int(np.ceil(len(events) / 2))
    figure, axes = plt.subplots(rows, 2, figsize=(20, 4.8 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, event_df in zip(axes.ravel(), events):
        selected_event_id = int(event_df["selected_event_id"].iloc[0])
        original_event_id = int(event_df["original_shared_event_id"].iloc[0])
        event_start = pd.to_datetime(event_df["event_start"].iloc[0])
        event_end = pd.to_datetime(event_df["event_end"].iloc[0])
        axis.axis("on")
        axis.plot(event_df["timestamp"], event_df["node1_rain_intensity_normalized"], color="tab:purple", linewidth=1.2, label="N1 intensity")
        axis.plot(event_df["timestamp"], event_df["node2_rain_intensity_normalized"], color="tab:red", linewidth=1.2, label="N2 intensity")
        axis.axvline(event_start, color="black", linestyle=":", linewidth=0.9)
        axis.axvline(event_end, color="black", linestyle=":", linewidth=0.9)
        axis.set_title(f"Selected {selected_event_id} (shared {original_event_id})", fontsize=10)
        axis.set_ylabel("Rain Intensity (mm/hr)")
        axis.tick_params(axis="x", rotation=30, labelsize=8)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.85)
    figure.suptitle("Merged Node Selected Events: Rain Intensity (mm/hr)", fontsize=15)
    figure.tight_layout(rect=[0, 0, 1, 0.97])
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_selected_event_4_intensity(df: pd.DataFrame, output_path: Path) -> None:
    event_df = df.loc[df["selected_event_id"] == 4].reset_index(drop=True)
    if event_df.empty:
        return
    event_start = pd.to_datetime(event_df["event_start"].iloc[0])
    event_end = pd.to_datetime(event_df["event_end"].iloc[0])
    figure, axis = plt.subplots(figsize=(14, 5))
    axis.plot(event_df["timestamp"], event_df["node1_rain_intensity_normalized"], color="tab:purple", marker="o", linewidth=1.5, label="Node 1 intensity")
    axis.plot(event_df["timestamp"], event_df["node2_rain_intensity_normalized"], color="tab:red", marker="o", linewidth=1.5, label="Node 2 intensity")
    axis.axvline(event_start, color="black", linestyle=":", linewidth=1)
    axis.axvline(event_end, color="black", linestyle=":", linewidth=1)
    axis.set_title("Selected Event 4: Rain Intensity (mm/hr)")
    axis.set_ylabel("Rain Intensity (mm/hr)")
    axis.set_xlabel("Timestamp")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_selected_event_6(df: pd.DataFrame, output_path: Path) -> None:
    event_df = df.loc[df["selected_event_id"] == 6].reset_index(drop=True)
    if event_df.empty:
        return
    event_start = pd.to_datetime(event_df["event_start"].iloc[0])
    event_end = pd.to_datetime(event_df["event_end"].iloc[0])
    node1_lag = event_df["node1_lag_peak_rain_intensity_to_peak_water_level_h"].iloc[0]
    node2_lag = event_df["node2_lag_peak_rain_intensity_to_peak_water_level_h"].iloc[0]
    
    n1_raw = event_df["node1_rain_intensity"]
    n2_raw = event_df["node2_rain_intensity"]
    n1_max = n1_raw.max()
    scale_factor = 7.5 / n1_max if n1_max > 0 else 1.0
    
    n1_scaled = n1_raw * scale_factor
    n2_scaled = n2_raw * scale_factor
    
    figure, axis = plt.subplots(figsize=(14, 6))
    water_axis = axis.twinx()
    
    axis.plot(event_df["timestamp"], n1_scaled, color="tab:purple", linewidth=1.2, label="N1 intensity")
    axis.plot(event_df["timestamp"], n2_scaled, color="tab:red", linewidth=1.2, label="N2 intensity")
    water_axis.plot(event_df["timestamp"], event_df["node1_canal_water_level_preprocessed"], color="tab:green", linewidth=1.5, label="N1 water")
    water_axis.plot(event_df["timestamp"], event_df["node2_canal_water_level_preprocessed"], color="tab:orange", linewidth=1.5, label="N2 water")
    
    axis.axvline(event_start, color="black", linestyle=":", linewidth=0.9)
    axis.axvline(event_end, color="black", linestyle=":", linewidth=0.9)
    
    axis.set_title(f"Rain Event 2026-02-26 | Lag N1/N2 {node1_lag:.1f}h/{node2_lag:.1f}h", fontsize=12, fontweight="bold")
    axis.set_xlabel("Timestamp", fontsize=11)
    axis.set_ylabel("Rain Intensity (mm/hr)", fontsize=11)
    water_axis.set_ylabel("Canal Water Level (cm)", fontsize=11)
    
    axis.tick_params(axis="x", rotation=30)
    axis.grid(True, alpha=0.25)
    
    lines, labels = axis.get_legend_handles_labels()
    water_lines, water_labels = water_axis.get_legend_handles_labels()
    axis.legend(lines + water_lines, labels + water_labels, loc="upper left", fontsize=10, ncol=2, framealpha=0.9)
    
    figure.tight_layout(rect=[0, 0.05, 1, 1])
    figure.text(0.5, 0.02, "Selected Event 6 (Shared Event 17); dotted lines indicate rainfall-event window", ha="center", fontsize=10)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SOURCE, parse_dates=["timestamp", "event_start", "event_end"])
    normalized, caps = normalize_intensity(df)
    normalized.to_csv(DATA_DIR / "normalized_selected_events_all_merged_nodes.csv", index=False)
    pd.DataFrame([{"node": node, "cap_pagasa_heavy_rain_mm_hr": cap} for node, cap in caps.items()]).to_csv(
        DATA_DIR / "normalization_summary.csv",
        index=False,
    )
    plot_combined_collage(normalized, FIGURE_DIR / "normalized_selected_events_combined_collage.png")
    plot_intensity_collage(normalized, FIGURE_DIR / "normalized_selected_events_rain_intensity_collage.png")
    plot_selected_event_4_intensity(normalized, FIGURE_DIR / "normalized_selected_event_04_rain_intensity.png")
    plot_selected_event_6(normalized, FIGURE_DIR / "normalized_selected_event_06_rain_water.png")
    print(f"Wrote normalized figures to {FIGURE_DIR}")
    print(f"Rain intensity normalized to PAGASA heavy rain threshold: 7.5 mm/hr")
    print(f"Water level capped at: 55 cm")


if __name__ == "__main__":
    main()
