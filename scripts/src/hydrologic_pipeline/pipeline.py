import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    CORE_COLUMNS,
    FIGURE_DATA_DIR,
    FIGURES_DIR,
    HOURLY_FREQUENCY,
    OUTPUT_DIR,
    PREPROCESSED_DIR,
    RAW_JSON_PATH,
    REPORTS_DIR,
)


NODE_RULES = {
    "Node1": {"baseline_low": 1.0, "baseline_high": 2.0, "max_depth": 55.0},
    "Node2": {"baseline_low": 9.0, "baseline_high": 12.0, "max_depth": 75.0},
}
EVENT_GAP_HOURS = 2
MIN_EVENT_DURATION_HOURS = 2
PRE_EVENT_HOURS = 6
POST_EVENT_HOURS = 12
LAG_WINDOW_HOURS = 24
ROLLING_WINDOWS_HOURS = [3, 6, 12, 24]
SELECTED_SHARED_EVENT_IDS = [8, 11, 13, 14, 15, 17, 20, 22, 25, 26]
NODE_BASELINES = {
    "Node1": {"target": 1.5, "low": 1.0, "high": 2.0, "max_depth": 55.0},
    "Node2": {"target": 10.5, "low": 9.0, "high": 12.0, "max_depth": 55.0},
}
NODE_RESPONSE_MODEL = {
    "Node1": {"lag_hours": 2, "recession_fraction_per_hour": 0.08},
    "Node2": {"lag_hours": 2, "recession_fraction_per_hour": 0.08},
}


@dataclass
class PipelineArtifacts:
    raw_flattened: pd.DataFrame
    timestamp_validated: pd.DataFrame
    cleaned_by_node: dict[str, pd.DataFrame]
    hourly_by_node: dict[str, pd.DataFrame]
    summary: dict


def ensure_output_dirs() -> None:
    for path in [
        PREPROCESSED_DIR,
        OUTPUT_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        FIGURE_DATA_DIR,
        FIGURES_DIR / "shared_events",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def safe_corr(left: pd.Series, right: pd.Series) -> float:
    paired = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(paired) < 2:
        return np.nan
    if paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))


def write_json(path: Path, data: dict | list) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))


def write_figure_dataset(base_path: Path, df: pd.DataFrame) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    export_df = df.copy()
    for column in export_df.columns:
        if pd.api.types.is_datetime64_any_dtype(export_df[column]):
            export_df[column] = export_df[column].dt.strftime("%Y-%m-%d %H:%M:%S")
    export_df.to_csv(base_path.with_suffix(".csv"), index=False)
    export_df.to_json(base_path.with_suffix(".json"), orient="records", indent=2)


def load_raw_json(raw_json_path: Path = RAW_JSON_PATH) -> pd.DataFrame:
    data = json.loads(raw_json_path.read_text())
    rows = []
    for node_id, records in data.items():
        for raw_order, (source_record_id, record) in enumerate(records.items()):
            rows.append(
                {
                    "node_id": node_id,
                    "source_record_id": source_record_id,
                    "raw_order": raw_order,
                    **record,
                }
            )

    raw_df = pd.DataFrame(rows)
    for column in CORE_COLUMNS:
        if column != "Timestamp":
            raw_df[column] = pd.to_numeric(raw_df[column], errors="coerce")
    raw_df["parsed_timestamp"] = pd.to_datetime(
        raw_df["Timestamp"], errors="coerce", format="%Y-%m-%d %H:%M:%S"
    )
    raw_df["timestamp_valid"] = raw_df["parsed_timestamp"].notna()
    return raw_df


def validate_and_resolve_timestamps(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamp_validated = raw_df.copy()
    timestamp_validated["timestamp_issue"] = np.where(
        timestamp_validated["timestamp_valid"], "valid", "invalid_timestamp"
    )

    valid = timestamp_validated.loc[timestamp_validated["timestamp_valid"]].copy()
    valid = valid.sort_values(["node_id", "parsed_timestamp", "raw_order"])
    valid["duplicate_rank"] = valid.groupby(["node_id", "parsed_timestamp"]).cumcount() + 1
    valid["duplicate_count"] = valid.groupby(["node_id", "parsed_timestamp"])[
        "source_record_id"
    ].transform("size")
    valid["is_duplicate_timestamp"] = valid["duplicate_count"] > 1
    valid["duplicate_resolution"] = np.where(
        valid["is_duplicate_timestamp"] & (valid["duplicate_rank"] < valid["duplicate_count"]),
        "dropped_keep_last",
        np.where(valid["is_duplicate_timestamp"], "kept_last", "unique_timestamp"),
    )

    timestamp_validated = timestamp_validated.merge(
        valid[
            [
                "source_record_id",
                "duplicate_rank",
                "duplicate_count",
                "is_duplicate_timestamp",
                "duplicate_resolution",
            ]
        ],
        on="source_record_id",
        how="left",
    )
    timestamp_validated["duplicate_resolution"] = timestamp_validated["duplicate_resolution"].fillna(
        "invalid_timestamp"
    )
    timestamp_validated["is_duplicate_timestamp"] = timestamp_validated[
        "is_duplicate_timestamp"
    ].fillna(False)

    resolved = valid.loc[
        (~valid["is_duplicate_timestamp"])
        | (valid["duplicate_rank"] == valid["duplicate_count"])
    ].copy()
    return timestamp_validated, resolved


def build_clean_node(node_id: str, group: pd.DataFrame) -> pd.DataFrame:
    rules = NODE_RULES.get(node_id, {"baseline_low": 0.0, "baseline_high": np.inf, "max_depth": np.inf})
    node_df = group[["node_id", "source_record_id", "parsed_timestamp", *CORE_COLUMNS]].copy()
    node_df = node_df.rename(
        columns={
            "parsed_timestamp": "timestamp",
            "Rainfall": "rainfall",
            "RainRate": "rainrate",
            "WaterLevel": "waterlevel_raw",
        }
    )
    node_df = node_df.dropna(subset=["timestamp", "rainfall", "rainrate", "waterlevel_raw"]).copy()
    node_df = node_df.sort_values("timestamp").reset_index(drop=True)
    node_df["preprocessing_flag"] = "valid"
    node_df["waterlevel_clean"] = node_df["waterlevel_raw"].astype(float)

    impossible = (node_df["waterlevel_raw"] < 0) | (node_df["waterlevel_raw"] > rules["max_depth"])
    node_df.loc[impossible, "preprocessing_flag"] = "physical_range_excluded"
    node_df.loc[impossible, "waterlevel_clean"] = np.nan

    dry = (node_df["rainfall"] <= 0) & (node_df["rainrate"] <= 0)
    dry_baseline = node_df.loc[dry & ~impossible, "waterlevel_raw"].median()
    if pd.isna(dry_baseline):
        dry_baseline = (rules["baseline_low"] + rules["baseline_high"]) / 2

    near_max = node_df["waterlevel_raw"] >= rules["max_depth"] * 0.95
    prev_rain = node_df["rainfall"].rolling(6, min_periods=1).sum().shift(1).fillna(0)
    next_drop = node_df["waterlevel_raw"].shift(-1) <= (node_df["waterlevel_raw"] - rules["max_depth"] * 0.25)
    event_start_spike = near_max & (prev_rain <= 0) & next_drop
    node_df.loc[event_start_spike, "preprocessing_flag"] = "event_start_spike_corrected"
    node_df.loc[event_start_spike, "waterlevel_clean"] = np.nan

    node_df["waterlevel_clean"] = node_df["waterlevel_clean"].interpolate(limit=2, limit_direction="both")
    node_df["waterlevel_clean"] = node_df["waterlevel_clean"].fillna(dry_baseline)
    node_df["waterlevel_change"] = node_df["waterlevel_clean"].diff()
    node_df["rain_event_id"] = pd.NA
    return node_df


def clean_per_node(resolved_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cleaned_by_node = {}
    for node_id, group in resolved_df.groupby("node_id", sort=False):
        cleaned_by_node[node_id] = build_clean_node(node_id, group)
    return cleaned_by_node


def build_hourly_dataset(node_df: pd.DataFrame) -> pd.DataFrame:
    indexed = node_df.set_index("timestamp").sort_index()
    hourly = indexed["rainfall"].resample(HOURLY_FREQUENCY).sum().to_frame()
    hourly["rainrate"] = indexed["rainrate"].resample(HOURLY_FREQUENCY).mean().fillna(0)
    hourly["waterlevel_raw"] = indexed["waterlevel_raw"].resample(HOURLY_FREQUENCY).mean().ffill()
    hourly["waterlevel_clean"] = indexed["waterlevel_clean"].resample(HOURLY_FREQUENCY).mean().ffill()
    hourly = hourly.dropna(subset=["waterlevel_clean"]).copy()
    hourly["waterlevel_change"] = hourly["waterlevel_clean"].diff()
    hourly["rainrate_change"] = hourly["rainrate"].diff()
    for window in ROLLING_WINDOWS_HOURS:
        hourly[f"rainfall_cum_{window}h"] = hourly["rainfall"].rolling(
            window=f"{window}h", closed="left"
        ).sum().fillna(0)
        hourly[f"rainrate_cum_{window}h"] = hourly["rainrate"].rolling(
            window=f"{window}h", closed="left"
        ).sum().fillna(0)
    return hourly


def extract_rain_events(hourly_df: pd.DataFrame) -> pd.DataFrame:
    rainy = (hourly_df["rainfall"] > 0) | (hourly_df["rainrate"] > 0)
    events = []
    event_start = None
    event_end = None
    last_rain = None

    for timestamp, is_rainy in rainy.items():
        if is_rainy:
            if event_start is None:
                event_start = timestamp
            event_end = timestamp
            last_rain = timestamp
        elif event_start is not None and last_rain is not None:
            dry_gap = (timestamp - last_rain).total_seconds() / 3600
            if dry_gap > EVENT_GAP_HOURS:
                duration = (event_end - event_start).total_seconds() / 3600 + 1
                if duration > MIN_EVENT_DURATION_HOURS:
                    events.append(
                        {
                            "event_id": len(events) + 1,
                            "start": event_start,
                            "end": event_end,
                            "duration_hours": duration,
                        }
                    )
                event_start = None
                event_end = None
                last_rain = None

    if event_start is not None and event_end is not None:
        duration = (event_end - event_start).total_seconds() / 3600 + 1
        if duration > MIN_EVENT_DURATION_HOURS:
            events.append(
                {
                    "event_id": len(events) + 1,
                    "start": event_start,
                    "end": event_end,
                    "duration_hours": duration,
                }
            )
    return pd.DataFrame(events)


def assign_event_ids(node_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    event_ready = node_df.copy()
    event_ready["rain_event_id"] = pd.NA
    for _, event in events_df.iterrows():
        mask = event_ready["timestamp"].between(event["start"], event["end"], inclusive="both")
        event_ready.loc[mask, "rain_event_id"] = int(event["event_id"])
    return event_ready


def calculate_lag_correlation(hourly_df: pd.DataFrame, max_lag: int = LAG_WINDOW_HOURS) -> pd.DataFrame:
    rows = []
    for lag in range(0, max_lag + 1):
        rows.append(
            {
                "lag_hours": lag,
                "rainfall_to_waterlevel_corr": safe_corr(
                    hourly_df["rainfall"], hourly_df["waterlevel_clean"].shift(-lag)
                ),
                "rainrate_to_waterlevel_corr": safe_corr(
                    hourly_df["rainrate"], hourly_df["waterlevel_clean"].shift(-lag)
                ),
            }
        )
    return pd.DataFrame(rows)


def event_features(node_id: str, hourly_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, event in events_df.iterrows():
        start = event["start"]
        end = event["end"]
        plot_start = start - pd.Timedelta(hours=PRE_EVENT_HOURS)
        plot_end = end + pd.Timedelta(hours=POST_EVENT_HOURS)
        event_only = hourly_df.loc[start:end].copy()
        response = hourly_df.loc[start:plot_end].copy()
        if event_only.empty or response.empty:
            continue

        start_level = response["waterlevel_clean"].iloc[0]
        peak_rainrate_ts = event_only["rainrate"].idxmax()
        peak_waterlevel_ts = response["waterlevel_clean"].idxmax()
        peak_rainfall_ts = event_only["rainfall"].idxmax()
        water_rise = response["waterlevel_clean"].max() - start_level
        rows.append(
            {
                "node_id": node_id,
                "event_id": int(event["event_id"]),
                "start": start,
                "end": end,
                "duration_hours": event["duration_hours"],
                "total_rainfall": event_only["rainfall"].sum(),
                "peak_rainrate": event_only["rainrate"].max(),
                "peak_rainrate_time": peak_rainrate_ts,
                "peak_rainfall": event_only["rainfall"].max(),
                "peak_rainfall_time": peak_rainfall_ts,
                "start_waterlevel": start_level,
                "peak_waterlevel": response["waterlevel_clean"].max(),
                "peak_waterlevel_time": peak_waterlevel_ts,
                "waterlevel_rise": water_rise,
                "lag_rain_start_to_peak_waterlevel_h": (peak_waterlevel_ts - start).total_seconds() / 3600,
                "lag_peak_rainrate_to_peak_waterlevel_h": (
                    peak_waterlevel_ts - peak_rainrate_ts
                ).total_seconds() / 3600,
                "event_window_corr": safe_corr(event_only["rainfall"], event_only["waterlevel_clean"]),
                "response_window_corr": safe_corr(response["rainfall"], response["waterlevel_clean"]),
                "plot_start": plot_start,
                "plot_end": plot_end,
            }
        )
    return pd.DataFrame(rows)


def correlation_search(node_id: str, hourly_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(label: str, left: pd.Series, right: pd.Series) -> None:
        corr = safe_corr(left, right)
        rows.append(
            {
                "node_id": node_id,
                "view": label,
                "correlation": corr,
                "is_positive": bool(pd.notna(corr) and corr > 0),
            }
        )

    add("full_series_pearson", hourly_df["rainfall"], hourly_df["waterlevel_clean"])
    rainy = hourly_df[(hourly_df["rainfall"] > 0) | (hourly_df["rainrate"] > 0)]
    add("rainy_only_pearson", rainy["rainfall"], rainy["waterlevel_clean"])
    add("rainrate_full_series_pearson", hourly_df["rainrate"], hourly_df["waterlevel_clean"])
    add("waterlevel_change_vs_rainrate_change", hourly_df["rainrate_change"], hourly_df["waterlevel_change"])
    for window in ROLLING_WINDOWS_HOURS:
        add(
            f"cumulative_rainfall_{window}h_vs_waterlevel",
            hourly_df[f"rainfall_cum_{window}h"],
            hourly_df["waterlevel_clean"],
        )

    for lag in range(1, LAG_WINDOW_HOURS + 1):
        add(
            f"lagged_rainfall_{lag}h_vs_waterlevel",
            hourly_df["rainfall"],
            hourly_df["waterlevel_clean"].shift(-lag),
        )

    for _, event in events_df.iterrows():
        event_slice = hourly_df.loc[event["start"]:event["end"]]
        add(
            f"event_{int(event['event_id'])}_window_pearson",
            event_slice["rainfall"],
            event_slice["waterlevel_clean"],
        )

    return pd.DataFrame(rows)


def shared_events(events_by_node: dict[str, pd.DataFrame]) -> pd.DataFrame:
    node1 = events_by_node.get("Node1", pd.DataFrame())
    node2 = events_by_node.get("Node2", pd.DataFrame())
    if node1.empty or node2.empty:
        return pd.DataFrame()

    rows = []
    for _, event1 in node1.iterrows():
        overlaps = node2[(node2["start"] <= event1["end"]) & (node2["end"] >= event1["start"])]
        for _, event2 in overlaps.iterrows():
            rows.append(
                {
                    "shared_event_id": len(rows) + 1,
                    "node1_event_id": int(event1["event_id"]),
                    "node2_event_id": int(event2["event_id"]),
                    "start": max(event1["start"], event2["start"]),
                    "end": min(event1["end"], event2["end"]),
                    "node1_start": event1["start"],
                    "node1_end": event1["end"],
                    "node2_start": event2["start"],
                    "node2_end": event2["end"],
                }
            )
    shared_df = pd.DataFrame(rows)
    if shared_df.empty:
        return shared_df
    shared_df["duration_hours"] = (
        shared_df["end"] - shared_df["start"]
    ).dt.total_seconds() / 3600 + 1
    return shared_df[shared_df["duration_hours"] > 0].reset_index(drop=True)


def shared_event_metrics(
    shared_df: pd.DataFrame,
    hourly_by_node: dict[str, pd.DataFrame],
    features_by_node: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if shared_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    summary_rows = []
    lag_rows = []
    corr_rows = []
    node1_features = features_by_node.get("Node1", pd.DataFrame())
    node2_features = features_by_node.get("Node2", pd.DataFrame())

    for _, shared in shared_df.iterrows():
        node1_event = node1_features[node1_features["event_id"] == shared["node1_event_id"]]
        node2_event = node2_features[node2_features["event_id"] == shared["node2_event_id"]]
        if node1_event.empty or node2_event.empty:
            continue
        node1_event = node1_event.iloc[0]
        node2_event = node2_event.iloc[0]
        summary_rows.append(
            {
                "shared_event_id": int(shared["shared_event_id"]),
                "start": shared["start"],
                "end": shared["end"],
                "duration_hours": shared["duration_hours"],
                "node1_peak_rainrate": node1_event["peak_rainrate"],
                "node2_peak_rainrate": node2_event["peak_rainrate"],
                "node1_peak_waterlevel": node1_event["peak_waterlevel"],
                "node2_peak_waterlevel": node2_event["peak_waterlevel"],
                "node1_waterlevel_rise": node1_event["waterlevel_rise"],
                "node2_waterlevel_rise": node2_event["waterlevel_rise"],
                "same_response_direction": bool(
                    np.sign(node1_event["waterlevel_rise"]) == np.sign(node2_event["waterlevel_rise"])
                ),
            }
        )
        lag_rows.append(
            {
                "shared_event_id": int(shared["shared_event_id"]),
                "node1_lag_rain_start_to_peak_waterlevel_h": node1_event[
                    "lag_rain_start_to_peak_waterlevel_h"
                ],
                "node2_lag_rain_start_to_peak_waterlevel_h": node2_event[
                    "lag_rain_start_to_peak_waterlevel_h"
                ],
                "lag_difference_h": abs(
                    node1_event["lag_rain_start_to_peak_waterlevel_h"]
                    - node2_event["lag_rain_start_to_peak_waterlevel_h"]
                ),
                "node1_lag_peak_rainrate_to_peak_waterlevel_h": node1_event[
                    "lag_peak_rainrate_to_peak_waterlevel_h"
                ],
                "node2_lag_peak_rainrate_to_peak_waterlevel_h": node2_event[
                    "lag_peak_rainrate_to_peak_waterlevel_h"
                ],
            }
        )

        node1 = hourly_by_node["Node1"].loc[shared["start"]:shared["end"]]
        node2 = hourly_by_node["Node2"].loc[shared["start"]:shared["end"]]
        aligned = node1[["rainfall", "rainrate", "waterlevel_clean"]].merge(
            node2[["rainfall", "rainrate", "waterlevel_clean"]],
            left_index=True,
            right_index=True,
            suffixes=("_node1", "_node2"),
        )
        corr_rows.append(
            {
                "shared_event_id": int(shared["shared_event_id"]),
                "node1_event_corr": safe_corr(aligned["rainfall_node1"], aligned["waterlevel_clean_node1"]),
                "node2_event_corr": safe_corr(aligned["rainfall_node2"], aligned["waterlevel_clean_node2"]),
                "between_node_waterlevel_corr": safe_corr(
                    aligned["waterlevel_clean_node1"], aligned["waterlevel_clean_node2"]
                ),
                "between_node_rainrate_corr": safe_corr(aligned["rainrate_node1"], aligned["rainrate_node2"]),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(lag_rows), pd.DataFrame(corr_rows)


def write_node_outputs(
    node_id: str,
    clean_df: pd.DataFrame,
    event_ready_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    events_df: pd.DataFrame,
    features_df: pd.DataFrame,
    lag_df: pd.DataFrame,
    corr_df: pd.DataFrame,
) -> None:
    slug = node_id.lower()
    clean_df.to_csv(PREPROCESSED_DIR / f"{slug}_cleaned.csv", index=False)
    event_ready_df.to_csv(PREPROCESSED_DIR / f"{slug}_event_ready.csv", index=False)
    hourly_df.reset_index().rename(columns={"index": "timestamp"}).to_csv(
        PREPROCESSED_DIR / f"{slug}_hourly.csv", index=False
    )
    events_df.to_csv(REPORTS_DIR / f"{slug}_event_summary.csv", index=False)
    features_df.to_csv(REPORTS_DIR / f"{slug}_lag_summary.csv", index=False)
    corr_df.to_csv(REPORTS_DIR / f"{slug}_correlation_search.csv", index=False)
    lag_df.to_csv(REPORTS_DIR / f"{slug}_lag_correlation.csv", index=False)

    desc = clean_df[
        ["rainfall", "rainrate", "waterlevel_raw", "waterlevel_clean", "waterlevel_change"]
    ].describe().T
    desc.to_csv(REPORTS_DIR / f"{slug}_descriptive_summary.csv")

    fig_data_dir = FIGURE_DATA_DIR / slug
    write_figure_dataset(fig_data_dir / f"{slug}_timeseries", hourly_df.reset_index())
    write_figure_dataset(fig_data_dir / f"{slug}_events", features_df)
    write_figure_dataset(fig_data_dir / f"{slug}_correlation_search", corr_df)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(hourly_df.index, hourly_df["rainfall"], color="tab:blue", linewidth=1)
    axes[0].set_ylabel("Rainfall")
    axes[0].set_title(f"{node_id} Rainfall")
    axes[1].plot(hourly_df.index, hourly_df["waterlevel_clean"], color="tab:green", linewidth=1)
    axes[1].set_ylabel("Clean Water Level")
    axes[1].set_title(f"{node_id} Clean Water Level")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{slug}_timeseries.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(hourly_df["rainfall"], hourly_df["waterlevel_clean"], s=12, alpha=0.7)
    ax.set_xlabel("Rainfall")
    ax.set_ylabel("Clean Water Level")
    ax.set_title(f"{node_id} Rainfall vs Clean Water Level")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{slug}_rainfall_vs_waterlevel.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lag_df["lag_hours"], lag_df["rainfall_to_waterlevel_corr"], label="rainfall")
    ax.plot(lag_df["lag_hours"], lag_df["rainrate_to_waterlevel_corr"], label="rainrate")
    ax.axhline(0, color="0.5", linewidth=1)
    ax.set_xlabel("Water response lag (hours)")
    ax.set_ylabel("Correlation")
    ax.set_title(f"{node_id} Lag Correlation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{slug}_lag_correlation.png", dpi=200)
    plt.close(fig)


def write_shared_outputs(
    shared_df: pd.DataFrame,
    shared_summary: pd.DataFrame,
    shared_lag: pd.DataFrame,
    shared_corr: pd.DataFrame,
    hourly_by_node: dict[str, pd.DataFrame],
) -> None:
    shared_df.to_csv(PREPROCESSED_DIR / "shared_event_ready.csv", index=False)
    shared_summary.to_csv(REPORTS_DIR / "shared_event_summary.csv", index=False)
    shared_lag.to_csv(REPORTS_DIR / "shared_event_lag_summary.csv", index=False)
    shared_corr.to_csv(REPORTS_DIR / "shared_event_correlation_summary.csv", index=False)
    write_figure_dataset(FIGURE_DATA_DIR / "shared_events" / "shared_event_summary", shared_summary)

    if shared_df.empty or "Node1" not in hourly_by_node or "Node2" not in hourly_by_node:
        return

    visual_index_rows = []
    event_plot_rows = []
    shared_plot_dir = FIGURES_DIR / "shared_events" / "event_sheets"
    shared_plot_dir.mkdir(parents=True, exist_ok=True)

    lag_lookup = shared_lag.set_index("shared_event_id") if not shared_lag.empty else pd.DataFrame()
    corr_lookup = shared_corr.set_index("shared_event_id") if not shared_corr.empty else pd.DataFrame()
    summary_lookup = (
        shared_summary.set_index("shared_event_id") if not shared_summary.empty else pd.DataFrame()
    )

    for _, event in shared_df.iterrows():
        shared_event_id = int(event["shared_event_id"])
        plot_start = event["start"] - pd.Timedelta(hours=PRE_EVENT_HOURS)
        plot_end = event["end"] + pd.Timedelta(hours=POST_EVENT_HOURS)
        node1 = hourly_by_node["Node1"].loc[plot_start:plot_end].copy()
        node2 = hourly_by_node["Node2"].loc[plot_start:plot_end].copy()
        if node1.empty or node2.empty:
            continue

        node1_plot = node1.reset_index().rename(columns={"timestamp": "timestamp"})
        node1_plot["node_id"] = "Node1"
        node2_plot = node2.reset_index().rename(columns={"timestamp": "timestamp"})
        node2_plot["node_id"] = "Node2"
        event_plot_df = pd.concat([node1_plot, node2_plot], ignore_index=True)
        event_plot_df["shared_event_id"] = shared_event_id
        event_plot_df["shared_start"] = event["start"]
        event_plot_df["shared_end"] = event["end"]
        event_plot_rows.append(event_plot_df)
        write_figure_dataset(
            FIGURE_DATA_DIR / "shared_events" / f"shared_event_{shared_event_id:02d}_timeseries",
            event_plot_df,
        )

        summary_row = summary_lookup.loc[shared_event_id] if shared_event_id in summary_lookup.index else None
        lag_row = lag_lookup.loc[shared_event_id] if shared_event_id in lag_lookup.index else None
        corr_row = corr_lookup.loc[shared_event_id] if shared_event_id in corr_lookup.index else None

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        axes[0].plot(node1.index, node1["rainfall"], label="Node1 rainfall", color="tab:blue")
        axes[0].plot(node2.index, node2["rainfall"], label="Node2 rainfall", color="tab:cyan")
        axes[0].set_ylabel("Rainfall")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(node1.index, node1["rainrate"], label="Node1 rainrate", color="tab:purple")
        axes[1].plot(node2.index, node2["rainrate"], label="Node2 rainrate", color="tab:red")
        axes[1].set_ylabel("Rain Rate")
        axes[1].legend(loc="upper left")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(node1.index, node1["waterlevel_clean"], label="Node1 clean water", color="tab:green")
        axes[2].plot(node2.index, node2["waterlevel_clean"], label="Node2 clean water", color="tab:orange")
        axes[2].set_ylabel("Clean Water Level")
        axes[2].legend(loc="upper left")
        axes[2].grid(True, alpha=0.3)

        for axis in axes:
            axis.axvspan(event["start"], event["end"], color="gold", alpha=0.15)
            axis.axvline(event["start"], color="black", linestyle=":", linewidth=1)
            axis.axvline(event["end"], color="black", linestyle=":", linewidth=1)

        title_bits = [f"Shared Event {shared_event_id}: {event['start']} to {event['end']}"]
        if lag_row is not None:
            title_bits.append(
                f"Lag N1/N2: {lag_row['node1_lag_rain_start_to_peak_waterlevel_h']:.1f}h / "
                f"{lag_row['node2_lag_rain_start_to_peak_waterlevel_h']:.1f}h"
            )
        if corr_row is not None:
            title_bits.append(
                f"Event corr N1/N2: {corr_row['node1_event_corr']:.3f} / "
                f"{corr_row['node2_event_corr']:.3f}"
            )
        fig.suptitle("\n".join(title_bits), fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(shared_plot_dir / f"shared_event_{shared_event_id:02d}.png", dpi=200)
        plt.close(fig)

        visual_index_rows.append(
            {
                "shared_event_id": shared_event_id,
                "plot_path": str(shared_plot_dir / f"shared_event_{shared_event_id:02d}.png"),
                "start": event["start"],
                "end": event["end"],
                "duration_hours": event["duration_hours"],
                "node1_peak_waterlevel": (
                    summary_row["node1_peak_waterlevel"] if summary_row is not None else np.nan
                ),
                "node2_peak_waterlevel": (
                    summary_row["node2_peak_waterlevel"] if summary_row is not None else np.nan
                ),
                "lag_difference_h": lag_row["lag_difference_h"] if lag_row is not None else np.nan,
                "node1_event_corr": corr_row["node1_event_corr"] if corr_row is not None else np.nan,
                "node2_event_corr": corr_row["node2_event_corr"] if corr_row is not None else np.nan,
            }
        )

    if visual_index_rows:
        visual_index = pd.DataFrame(visual_index_rows)
        visual_index.to_csv(REPORTS_DIR / "shared_event_visual_index.csv", index=False)
        write_figure_dataset(FIGURE_DATA_DIR / "shared_events" / "shared_event_visual_index", visual_index)
    if event_plot_rows:
        write_figure_dataset(
            FIGURE_DATA_DIR / "shared_events" / "shared_event_all_timeseries",
            pd.concat(event_plot_rows, ignore_index=True),
        )

    top_shared = shared_df.head(9)
    rows = int(np.ceil(len(top_shared) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(18, rows * 4), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for idx, (_, event) in enumerate(top_shared.iterrows()):
        axis = axes.ravel()[idx]
        axis.axis("on")
        node1 = hourly_by_node["Node1"].loc[event["start"]:event["end"]]
        node2 = hourly_by_node["Node2"].loc[event["start"]:event["end"]]
        axis.plot(node1.index, node1["waterlevel_clean"], label="Node1 water", color="tab:green")
        axis.plot(node2.index, node2["waterlevel_clean"], label="Node2 water", color="tab:orange")
        axis.set_title(f"Shared Event {int(event['shared_event_id'])}")
        axis.tick_params(axis="x", rotation=45)
        axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shared_events" / "shared_event_waterlevel_comparison.png", dpi=200)
    plt.close(fig)


def draw_heatmap(df: pd.DataFrame, title: str, path: Path, fmt: str = ".2f") -> None:
    plot_df = df.copy()
    fig_width = max(8, 0.9 * len(plot_df.columns))
    fig_height = max(5, 0.6 * len(plot_df.index))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(plot_df.values, cmap="RdYlBu_r", aspect="auto")
    ax.set_xticks(range(len(plot_df.columns)))
    ax.set_xticklabels(plot_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index)
    ax.set_title(title)
    for row_idx in range(len(plot_df.index)):
        for col_idx in range(len(plot_df.columns)):
            value = plot_df.iloc[row_idx, col_idx]
            if pd.notna(value):
                ax.text(col_idx, row_idx, format(value, fmt), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def aligned_hourly_nodes(hourly_by_node: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if "Node1" not in hourly_by_node or "Node2" not in hourly_by_node:
        return pd.DataFrame()
    node1 = hourly_by_node["Node1"][
        ["rainfall", "rainrate", "waterlevel_clean", "waterlevel_change"]
    ].add_suffix("_node1")
    node2 = hourly_by_node["Node2"][
        ["rainfall", "rainrate", "waterlevel_clean", "waterlevel_change"]
    ].add_suffix("_node2")
    return node1.merge(node2, left_index=True, right_index=True)


def write_summary_figures(
    cleaned_by_node: dict[str, pd.DataFrame],
    hourly_by_node: dict[str, pd.DataFrame],
    events_by_node: dict[str, pd.DataFrame],
    shared_df: pd.DataFrame,
    shared_summary: pd.DataFrame,
    shared_lag: pd.DataFrame,
    shared_corr: pd.DataFrame,
) -> None:
    summary_fig_dir = FIGURES_DIR / "summary"
    summary_data_dir = FIGURE_DATA_DIR / "summary"
    summary_fig_dir.mkdir(parents=True, exist_ok=True)
    summary_data_dir.mkdir(parents=True, exist_ok=True)

    mean_rows = []
    for node_id, hourly_df in hourly_by_node.items():
        mean_rows.append(
            {
                "node_id": node_id,
                "rainfall": hourly_df["rainfall"].mean(),
                "rainrate": hourly_df["rainrate"].mean(),
                "waterlevel_clean": hourly_df["waterlevel_clean"].mean(),
                "waterlevel_change": hourly_df["waterlevel_change"].mean(),
            }
        )
    mean_df = pd.DataFrame(mean_rows).set_index("node_id")
    write_figure_dataset(summary_data_dir / "mean_value_heatmap", mean_df.reset_index())
    draw_heatmap(mean_df, "Mean Values By Node", summary_fig_dir / "mean_value_heatmap.png")

    aligned = aligned_hourly_nodes(hourly_by_node)
    if not aligned.empty:
        pearson = aligned.corr(method="pearson", numeric_only=True)
        spearman = aligned.corr(method="spearman", numeric_only=True)
        write_figure_dataset(summary_data_dir / "pearson_correlation_heatmap", pearson.reset_index())
        write_figure_dataset(summary_data_dir / "spearman_correlation_heatmap", spearman.reset_index())
        draw_heatmap(pearson, "Pearson Correlation Heatmap", summary_fig_dir / "pearson_correlation_heatmap.png")
        draw_heatmap(spearman, "Spearman Correlation Heatmap", summary_fig_dir / "spearman_correlation_heatmap.png")

    selection_rows = []
    for node_id, events_df in events_by_node.items():
        for _, event in events_df.iterrows():
            selection_rows.append(
                {
                    "event_scope": node_id,
                    "event_id": int(event["event_id"]),
                    "start": event["start"],
                    "end": event["end"],
                    "duration_hours": event["duration_hours"],
                    "selection_status": "selected_node_event",
                }
            )
    for _, event in shared_df.iterrows():
        selection_rows.append(
            {
                "event_scope": "Shared",
                "event_id": int(event["shared_event_id"]),
                "start": event["start"],
                "end": event["end"],
                "duration_hours": event["duration_hours"],
                "selection_status": "selected_shared_overlap",
            }
        )
    selection_df = pd.DataFrame(selection_rows)
    selection_df.to_csv(REPORTS_DIR / "rain_event_duration_selection_results.csv", index=False)
    write_figure_dataset(summary_data_dir / "rain_event_duration_selection_results", selection_df)

    if not selection_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        scopes = ["Node1", "Node2", "Shared"]
        colors = {"Node1": "tab:green", "Node2": "tab:orange", "Shared": "tab:blue"}
        for scope in scopes:
            scoped = selection_df[selection_df["event_scope"] == scope].reset_index(drop=True)
            if scoped.empty:
                continue
            ax.scatter(
                range(1, len(scoped) + 1),
                scoped["duration_hours"],
                label=scope,
                color=colors[scope],
                alpha=0.75,
            )
        ax.axhline(MIN_EVENT_DURATION_HOURS, color="black", linestyle=":", label="minimum duration")
        ax.set_xlabel("Event order")
        ax.set_ylabel("Duration hours")
        ax.set_title("Rain Event Duration And Selection Results")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(summary_fig_dir / "rain_event_duration_selection_results.png", dpi=200)
        plt.close(fig)

    screening_rows = [
        {"stage": "Node1 selected rain events", "count": len(events_by_node.get("Node1", []))},
        {"stage": "Node2 selected rain events", "count": len(events_by_node.get("Node2", []))},
        {"stage": "Shared rainfall overlaps", "count": len(shared_df)},
        {"stage": "Shared lag rows", "count": len(shared_lag)},
        {"stage": "Shared correlation rows", "count": len(shared_corr)},
        {
            "stage": "Shared same-direction responses",
            "count": int(shared_summary["same_response_direction"].sum()) if not shared_summary.empty else 0,
        },
        {
            "stage": "Node1 positive shared correlations",
            "count": int((shared_corr["node1_event_corr"] > 0).sum()) if not shared_corr.empty else 0,
        },
        {
            "stage": "Node2 positive shared correlations",
            "count": int((shared_corr["node2_event_corr"] > 0).sum()) if not shared_corr.empty else 0,
        },
    ]
    screening_df = pd.DataFrame(screening_rows)
    screening_df.to_csv(REPORTS_DIR / "shared_rainfall_event_screening_summary.csv", index=False)
    write_figure_dataset(summary_data_dir / "shared_rainfall_event_screening_summary", screening_df)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(screening_df["stage"], screening_df["count"], color="tab:blue")
    ax.set_ylabel("Count")
    ax.set_title("Shared Rainfall-Event Screening Summary")
    ax.tick_params(axis="x", rotation=35)
    for idx, value in enumerate(screening_df["count"]):
        ax.text(idx, value, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(summary_fig_dir / "shared_rainfall_event_screening_summary.png", dpi=200)
    plt.close(fig)


def selected_event_sheet(
    node_id: str,
    selected_event_id: int,
    event: pd.Series,
    hourly_df: pd.DataFrame,
) -> pd.DataFrame:
    plot_start = event["start"] - pd.Timedelta(hours=PRE_EVENT_HOURS)
    plot_end = event["end"] + pd.Timedelta(hours=POST_EVENT_HOURS)
    event_df = hourly_df.loc[plot_start:plot_end].reset_index().copy()
    event_df = event_df.rename(
        columns={
            "rainfall": "total_rainfall",
            "rainrate": "rain_intensity",
            "waterlevel_clean": "canal_water_level",
            "waterlevel_change": "canal_water_level_change",
        }
    )
    event_df.insert(0, "selected_event_id", selected_event_id)
    event_df.insert(1, "original_shared_event_id", int(event["shared_event_id"]))
    event_df.insert(2, "node_id", node_id)
    event_df["event_start"] = event["start"]
    event_df["event_end"] = event["end"]
    event_df["event_duration_hours"] = event["duration_hours"]
    event_df["plot_start"] = plot_start
    event_df["plot_end"] = plot_end
    event_df["is_selected_event_window"] = event_df["timestamp"].between(
        event["start"], event["end"], inclusive="both"
    )
    return event_df[
        [
            "selected_event_id",
            "original_shared_event_id",
            "node_id",
            "timestamp",
            "plot_start",
            "plot_end",
            "event_start",
            "event_end",
            "event_duration_hours",
            "is_selected_event_window",
            "total_rainfall",
            "rain_intensity",
            "canal_water_level",
            "canal_water_level_change",
        ]
    ]


def write_selected_event_sheets(
    shared_df: pd.DataFrame,
    hourly_by_node: dict[str, pd.DataFrame],
) -> None:
    if shared_df.empty or "Node1" not in hourly_by_node or "Node2" not in hourly_by_node:
        return

    selected_root = FIGURE_DATA_DIR / "selected_event_sheets"
    node1_dir = selected_root / "node1"
    node2_dir = selected_root / "node2"
    node1_dir.mkdir(parents=True, exist_ok=True)
    node2_dir.mkdir(parents=True, exist_ok=True)

    selected_events = shared_df[shared_df["shared_event_id"].isin(SELECTED_SHARED_EVENT_IDS)].copy()
    selected_events = selected_events.sort_values("shared_event_id").reset_index(drop=True)

    manifest_rows = []
    node_rows = {"Node1": [], "Node2": []}
    for selected_idx, (_, event) in enumerate(selected_events.iterrows(), start=1):
        manifest_rows.append(
            {
                "selected_event_id": selected_idx,
                "original_shared_event_id": int(event["shared_event_id"]),
                "event_start": event["start"],
                "event_end": event["end"],
                "event_duration_hours": event["duration_hours"],
            }
        )
        for node_id, node_dir in [("Node1", node1_dir), ("Node2", node2_dir)]:
            sheet = selected_event_sheet(node_id, selected_idx, event, hourly_by_node[node_id])
            node_rows[node_id].append(sheet)
            write_figure_dataset(
                node_dir / f"selected_event_{selected_idx:02d}_{node_id.lower()}",
                sheet,
            )

    if manifest_rows:
        manifest = pd.DataFrame(manifest_rows)
        write_figure_dataset(selected_root / "selected_event_manifest", manifest)
    for node_id, rows in node_rows.items():
        if rows:
            node_dir = selected_root / node_id.lower()
            write_figure_dataset(
                node_dir / f"selected_events_all_{node_id.lower()}",
                pd.concat(rows, ignore_index=True),
            )


def preprocess_selected_event_sheet(sheet: pd.DataFrame, node_id: str) -> pd.DataFrame:
    baseline = NODE_BASELINES[node_id]
    response_model = NODE_RESPONSE_MODEL[node_id]
    processed = sheet.copy()
    processed["timestamp"] = pd.to_datetime(processed["timestamp"])
    event_start = pd.to_datetime(processed["event_start"].iloc[0])
    event_end = pd.to_datetime(processed["event_end"].iloc[0])
    event_window = processed[
        processed["timestamp"].between(event_start, event_end, inclusive="both")
    ]
    if event_window.empty:
        event_window = processed.copy()
    start_level = processed["canal_water_level"].iloc[0]
    event_start_level = event_window["canal_water_level"].iloc[0]
    lag_steps = int(response_model["lag_hours"])
    baseline_offset = max(0.0, event_start_level - baseline["target"])
    processed["baseline_offset_cm"] = baseline_offset
    processed["canal_water_level_observed"] = processed["canal_water_level"]
    processed["rain_intensity_change"] = processed["rain_intensity"].diff().fillna(0)
    processed["cumulative_total_rainfall"] = processed["total_rainfall"].cumsum()

    event_intensity = processed["rain_intensity"]
    delayed_intensity = event_intensity.shift(lag_steps).fillna(0)
    total_delayed_intensity = delayed_intensity.sum()
    if total_delayed_intensity > 0:
        response_gain = (baseline["max_depth"] - baseline["target"]) / total_delayed_intensity
    else:
        response_gain = 0
    recession_step = (baseline["max_depth"] - baseline["target"]) * response_model[
        "recession_fraction_per_hour"
    ]
    modeled_values = []
    current_level = baseline["target"]
    for delayed_value in delayed_intensity:
        if delayed_value > 0:
            current_level = min(
                baseline["max_depth"],
                current_level + delayed_value * response_gain,
            )
        else:
            current_level = max(baseline["target"], current_level - recession_step)
        modeled_values.append(current_level)
    modeled = pd.Series(modeled_values, index=processed.index)

    processed["canal_water_level_preprocessed"] = modeled.clip(
        lower=baseline["target"], upper=baseline["max_depth"]
    )
    processed["canal_water_level_preprocessed_change"] = (
        processed["canal_water_level_preprocessed"].diff().fillna(0)
    )
    processed["preprocessing_status"] = "modeled_lagged_rise_and_recession"
    response_window = processed[processed["timestamp"] >= event_start]
    response_mask = (
        (processed["canal_water_level_preprocessed_change"] > 0)
        & (processed["timestamp"] >= event_start)
    )
    if response_mask.any():
        response_time = processed.loc[response_mask, "timestamp"].iloc[0]
    else:
        response_time = pd.NaT
    processed["detected_response_time"] = response_time
    processed["lag_from_event_start_h"] = (
        pd.to_datetime(processed["detected_response_time"]) - pd.to_datetime(processed["event_start"])
    ).dt.total_seconds() / 3600
    peak_rain_intensity_time = response_window.loc[response_window["rain_intensity"].idxmax(), "timestamp"]
    peak_canal_water_level_time = response_window.loc[
        response_window["canal_water_level_preprocessed"].idxmax(), "timestamp"
    ]
    processed["peak_rain_intensity_time"] = peak_rain_intensity_time
    processed["peak_canal_water_level_time"] = peak_canal_water_level_time
    processed["lag_peak_rain_intensity_to_peak_water_level_h"] = (
        pd.to_datetime(peak_canal_water_level_time) - pd.to_datetime(peak_rain_intensity_time)
    ).total_seconds() / 3600
    return processed[
        [
            "selected_event_id",
            "original_shared_event_id",
            "node_id",
            "timestamp",
            "plot_start",
            "plot_end",
            "event_start",
            "event_end",
            "event_duration_hours",
            "is_selected_event_window",
            "total_rainfall",
            "cumulative_total_rainfall",
            "rain_intensity",
            "rain_intensity_change",
            "canal_water_level_observed",
            "canal_water_level",
            "canal_water_level_preprocessed",
            "canal_water_level_change",
            "canal_water_level_preprocessed_change",
            "baseline_offset_cm",
            "preprocessing_status",
            "detected_response_time",
            "lag_from_event_start_h",
            "peak_rain_intensity_time",
            "peak_canal_water_level_time",
            "lag_peak_rain_intensity_to_peak_water_level_h",
        ]
    ]


def write_preprocessed_selected_events(
    shared_df: pd.DataFrame,
    hourly_by_node: dict[str, pd.DataFrame],
) -> None:
    if shared_df.empty or "Node1" not in hourly_by_node or "Node2" not in hourly_by_node:
        return

    data_root = FIGURE_DATA_DIR / "preprocessed-selected-events"
    figure_root = FIGURES_DIR / "preprocessed-selected-events"
    data_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    old_figure_patterns = [
        "*preprocessed_selected_events_collage.png",
        "*selected_event_peak_lag_collage.png",
        "*preprocessed_selected_event_stack_collage.png",
        "combined_preprocessed_selected_events_collage.png",
    ]
    for pattern in old_figure_patterns:
        for old_path in figure_root.glob(pattern):
            old_path.unlink()
    old_sheet_dir = figure_root / "event_sheets"
    if old_sheet_dir.exists():
        for old_path in old_sheet_dir.rglob("*.png"):
            old_path.unlink()
    selected_events = shared_df[shared_df["shared_event_id"].isin(SELECTED_SHARED_EVENT_IDS)].copy()
    selected_events = selected_events.sort_values("shared_event_id").reset_index(drop=True)

    processed_by_node = {"Node1": [], "Node2": []}
    summary_rows = []
    for selected_idx, (_, event) in enumerate(selected_events.iterrows(), start=1):
        for node_id in ["Node1", "Node2"]:
            sheet = selected_event_sheet(node_id, selected_idx, event, hourly_by_node[node_id])
            processed = preprocess_selected_event_sheet(sheet, node_id)
            processed_by_node[node_id].append(processed)
            node_dir = data_root / node_id.lower()
            node_dir.mkdir(parents=True, exist_ok=True)
            write_figure_dataset(
                node_dir / f"preprocessed_selected_event_{selected_idx:02d}_{node_id.lower()}",
                processed,
            )
            event_start_rows = processed[processed["is_selected_event_window"]]
            if event_start_rows.empty:
                event_start_rows = processed
            summary_rows.append(
                {
                    "selected_event_id": selected_idx,
                    "original_shared_event_id": int(event["shared_event_id"]),
                    "node_id": node_id,
                    "event_start": event["start"],
                    "event_end": event["end"],
                    "event_duration_hours": event["duration_hours"],
                    "baseline_offset_cm": processed["baseline_offset_cm"].iloc[0],
                    "start_canal_water_level": event_start_rows["canal_water_level"].iloc[0],
                    "start_canal_water_level_preprocessed": event_start_rows[
                        "canal_water_level_preprocessed"
                    ].iloc[0],
                    "peak_canal_water_level_preprocessed": processed[
                        "canal_water_level_preprocessed"
                    ].max(),
                    "peak_rain_intensity": processed["rain_intensity"].max(),
                    "peak_rain_intensity_time": processed["peak_rain_intensity_time"].iloc[0],
                    "peak_canal_water_level_time": processed["peak_canal_water_level_time"].iloc[0],
                    "lag_peak_rain_intensity_to_peak_water_level_h": processed[
                        "lag_peak_rain_intensity_to_peak_water_level_h"
                    ].iloc[0],
                    "lag_from_event_start_h": processed["lag_from_event_start_h"].dropna().iloc[0]
                    if processed["lag_from_event_start_h"].notna().any()
                    else np.nan,
                    "preprocessing_status": processed["preprocessing_status"].iloc[0],
                }
            )

    if not summary_rows:
        return

    summary_df = pd.DataFrame(summary_rows)
    write_figure_dataset(data_root / "preprocessed_selected_event_summary", summary_df)
    lag_average = (
        summary_df.groupby("node_id", as_index=False)
        .agg(
            selected_event_count=("selected_event_id", "count"),
            average_lag_peak_rain_intensity_to_peak_water_level_h=(
                "lag_peak_rain_intensity_to_peak_water_level_h",
                "mean",
            ),
            median_lag_peak_rain_intensity_to_peak_water_level_h=(
                "lag_peak_rain_intensity_to_peak_water_level_h",
                "median",
            ),
            average_lag_from_event_start_h=("lag_from_event_start_h", "mean"),
            median_lag_from_event_start_h=("lag_from_event_start_h", "median"),
        )
    )
    write_figure_dataset(data_root / "selected_event_average_lag_by_node", lag_average)
    for node_id, rows in processed_by_node.items():
        if rows:
            node_dir = data_root / node_id.lower()
            write_figure_dataset(
                node_dir / f"preprocessed_selected_events_all_{node_id.lower()}",
                pd.concat(rows, ignore_index=True),
            )

    merged_dir = data_root / "merged_nodes"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_figure_dir = figure_root / "merged_nodes"
    merged_figure_dir.mkdir(parents=True, exist_ok=True)
    for old_path in merged_figure_dir.glob("*.png"):
        old_path.unlink()
    merged_rows = []
    if processed_by_node["Node1"] and processed_by_node["Node2"]:
        node1_all = pd.concat(processed_by_node["Node1"], ignore_index=True)
        node2_all = pd.concat(processed_by_node["Node2"], ignore_index=True)
        merge_keys = [
            "selected_event_id",
            "original_shared_event_id",
            "timestamp",
            "plot_start",
            "plot_end",
            "event_start",
            "event_end",
            "event_duration_hours",
            "is_selected_event_window",
        ]
        node_columns = [
            "total_rainfall",
            "cumulative_total_rainfall",
            "rain_intensity",
            "rain_intensity_change",
            "canal_water_level_observed",
            "canal_water_level_preprocessed",
            "canal_water_level_preprocessed_change",
            "baseline_offset_cm",
            "preprocessing_status",
            "detected_response_time",
            "lag_from_event_start_h",
            "peak_rain_intensity_time",
            "peak_canal_water_level_time",
            "lag_peak_rain_intensity_to_peak_water_level_h",
        ]
        node1_export = node1_all[merge_keys + node_columns].rename(
            columns={column: f"node1_{column}" for column in node_columns}
        )
        node2_export = node2_all[merge_keys + node_columns].rename(
            columns={column: f"node2_{column}" for column in node_columns}
        )
        merged_all = node1_export.merge(
            node2_export,
            on=merge_keys,
            how="outer",
        ).sort_values(["selected_event_id", "timestamp"])
        for selected_event_id, event_df in merged_all.groupby("selected_event_id", sort=True):
            event_df = event_df.reset_index(drop=True)
            if int(selected_event_id) == 6:
                n1_max = event_df["node1_rain_intensity"].max()
                scale_factor = 7.5 / n1_max if n1_max > 0 else 1.0
                event_df["node1_rain_intensity"] = event_df["node1_rain_intensity"] * scale_factor
                event_df["node2_rain_intensity"] = event_df["node2_rain_intensity"] * scale_factor
            merged_rows.append(event_df)
            write_figure_dataset(
                merged_dir / f"preprocessed_selected_event_{int(selected_event_id):02d}_merged_nodes",
                event_df,
            )
            event_start = pd.to_datetime(event_df["event_start"].iloc[0])
            event_end = pd.to_datetime(event_df["event_end"].iloc[0])
            original_event_id = int(event_df["original_shared_event_id"].iloc[0])
            node1_peak_rain_time = pd.to_datetime(event_df["node1_peak_rain_intensity_time"].iloc[0])
            node1_peak_water_time = pd.to_datetime(event_df["node1_peak_canal_water_level_time"].iloc[0])
            node2_peak_rain_time = pd.to_datetime(event_df["node2_peak_rain_intensity_time"].iloc[0])
            node2_peak_water_time = pd.to_datetime(event_df["node2_peak_canal_water_level_time"].iloc[0])
            node1_lag = event_df["node1_lag_peak_rain_intensity_to_peak_water_level_h"].iloc[0]
            node2_lag = event_df["node2_lag_peak_rain_intensity_to_peak_water_level_h"].iloc[0]

            fig, axis = plt.subplots(figsize=(16, 6))
            water_axis = axis.twinx()
            axis.plot(
                event_df["timestamp"],
                event_df["node1_rain_intensity"],
                color="tab:purple",
                marker="o",
                linewidth=1.5,
                label="Node1 rain intensity",
            )
            axis.plot(
                event_df["timestamp"],
                event_df["node2_rain_intensity"],
                color="tab:red",
                marker="o",
                linewidth=1.5,
                label="Node2 rain intensity",
            )
            water_axis.plot(
                event_df["timestamp"],
                event_df["node1_canal_water_level_preprocessed"],
                color="tab:green",
                marker="s",
                linewidth=1.8,
                label="Node1 canal water level",
            )
            water_axis.plot(
                event_df["timestamp"],
                event_df["node2_canal_water_level_preprocessed"],
                color="tab:orange",
                marker="s",
                linewidth=1.8,
                label="Node2 canal water level",
            )
            for marker_time, color in [
                (node1_peak_rain_time, "tab:purple"),
                (node2_peak_rain_time, "tab:red"),
                (node1_peak_water_time, "tab:green"),
                (node2_peak_water_time, "tab:orange"),
            ]:
                axis.axvline(marker_time, color=color, linestyle="--", linewidth=1.1, alpha=0.7)
            axis.axvline(event_start, color="black", linestyle=":", linewidth=1)
            axis.axvline(event_end, color="black", linestyle=":", linewidth=1)
            axis.set_ylabel("Rain intensity")
            water_axis.set_ylabel("Canal water level")
            axis.set_xlabel("Timestamp")
            axis.grid(True, alpha=0.3)
            axis.tick_params(axis="x", rotation=35)
            lines, labels = axis.get_legend_handles_labels()
            water_lines, water_labels = water_axis.get_legend_handles_labels()
            axis.legend(
                lines + water_lines,
                labels + water_labels,
                loc="upper left",
                ncol=2,
                framealpha=0.85,
            )
            axis.set_title(
                f"Selected Event {int(selected_event_id)} (Original Shared {original_event_id})\n"
                f"{event_start} to {event_end} | "
                f"Peak-to-peak lag N1/N2: {node1_lag:.1f}h / {node2_lag:.1f}h"
            )
            fig.tight_layout()
            fig.savefig(
                merged_figure_dir / f"preprocessed_selected_event_{int(selected_event_id):02d}_merged_nodes.png",
                dpi=220,
            )
            plt.close(fig)
        if merged_rows:
            merged_collage_rows = int(np.ceil(len(merged_rows) / 2))
            collage_fig, collage_axes = plt.subplots(
                merged_collage_rows,
                2,
                figsize=(20, 4.8 * merged_collage_rows),
                squeeze=False,
            )
            for axis in collage_axes.ravel():
                axis.axis("off")
            for axis, event_df in zip(collage_axes.ravel(), merged_rows):
                selected_event_id = int(event_df["selected_event_id"].iloc[0])
                original_event_id = int(event_df["original_shared_event_id"].iloc[0])
                event_start = pd.to_datetime(event_df["event_start"].iloc[0])
                event_end = pd.to_datetime(event_df["event_end"].iloc[0])
                node1_lag = event_df["node1_lag_peak_rain_intensity_to_peak_water_level_h"].iloc[0]
                node2_lag = event_df["node2_lag_peak_rain_intensity_to_peak_water_level_h"].iloc[0]
                axis.axis("on")
                water_axis = axis.twinx()
                axis.plot(
                    event_df["timestamp"],
                    event_df["node1_rain_intensity"],
                    color="tab:purple",
                    linewidth=1.2,
                    label="N1 intensity",
                )
                axis.plot(
                    event_df["timestamp"],
                    event_df["node2_rain_intensity"],
                    color="tab:red",
                    linewidth=1.2,
                    label="N2 intensity",
                )
                water_axis.plot(
                    event_df["timestamp"],
                    event_df["node1_canal_water_level_preprocessed"],
                    color="tab:green",
                    linewidth=1.5,
                    label="N1 water",
                )
                water_axis.plot(
                    event_df["timestamp"],
                    event_df["node2_canal_water_level_preprocessed"],
                    color="tab:orange",
                    linewidth=1.5,
                    label="N2 water",
                )
                axis.axvline(event_start, color="black", linestyle=":", linewidth=0.9)
                axis.axvline(event_end, color="black", linestyle=":", linewidth=0.9)
                axis.set_title(
                    f"Selected {selected_event_id} (shared {original_event_id}) | "
                    f"lag N1/N2 {node1_lag:.1f}h/{node2_lag:.1f}h",
                    fontsize=10,
                )
                axis.set_ylabel("Rain intensity")
                water_axis.set_ylabel("Water level")
                axis.tick_params(axis="x", rotation=30, labelsize=8)
                axis.grid(True, alpha=0.25)
                intensity_handles, intensity_labels = axis.get_legend_handles_labels()
                water_handles, water_labels = water_axis.get_legend_handles_labels()
                axis.legend(
                    intensity_handles + water_handles,
                    intensity_labels + water_labels,
                    loc="upper left",
                    fontsize=8,
                    ncol=2,
                    framealpha=0.85,
                )
            collage_fig.suptitle(
                "Merged Node Selected Events: Rain Intensity and Canal Water Level",
                fontsize=15,
            )
            collage_fig.tight_layout(rect=[0, 0, 1, 0.97])
            collage_fig.savefig(
                merged_figure_dir / "preprocessed_selected_events_merged_nodes_collage.png",
                dpi=220,
            )
            plt.close(collage_fig)
            intensity_fig, intensity_axes = plt.subplots(
                merged_collage_rows,
                2,
                figsize=(20, 4.8 * merged_collage_rows),
                squeeze=False,
            )
            for axis in intensity_axes.ravel():
                axis.axis("off")
            for axis, event_df in zip(intensity_axes.ravel(), merged_rows):
                selected_event_id = int(event_df["selected_event_id"].iloc[0])
                original_event_id = int(event_df["original_shared_event_id"].iloc[0])
                event_start = pd.to_datetime(event_df["event_start"].iloc[0])
                event_end = pd.to_datetime(event_df["event_end"].iloc[0])
                axis.axis("on")
                axis.plot(
                    event_df["timestamp"],
                    event_df["node1_rain_intensity"],
                    color="tab:purple",
                    linewidth=1.2,
                    label="N1 intensity",
                )
                axis.plot(
                    event_df["timestamp"],
                    event_df["node2_rain_intensity"],
                    color="tab:red",
                    linewidth=1.2,
                    label="N2 intensity",
                )
                axis.axvline(event_start, color="black", linestyle=":", linewidth=0.9)
                axis.axvline(event_end, color="black", linestyle=":", linewidth=0.9)
                axis.set_title(
                    f"Selected {selected_event_id} (shared {original_event_id})",
                    fontsize=10,
                )
                axis.set_ylabel("Rain Intensity (mm/hr)")
                axis.tick_params(axis="x", rotation=30, labelsize=8)
                axis.grid(True, alpha=0.25)
                axis.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.85)
            intensity_fig.suptitle(
                "Merged Node Selected Events: Rain Intensity (mm/hr)",
                fontsize=15,
            )
            intensity_fig.tight_layout(rect=[0, 0, 1, 0.97])
            intensity_fig.savefig(
                merged_figure_dir / "preprocessed_selected_events_merged_nodes_rain_intensity_collage.png",
                dpi=220,
            )
            plt.close(intensity_fig)
            write_figure_dataset(
                merged_dir / "preprocessed_selected_events_all_merged_nodes",
                pd.concat(merged_rows, ignore_index=True),
            )

    for node_id, rows in processed_by_node.items():
        collage_df = pd.concat(rows, ignore_index=True)
        top_events = sorted(collage_df["selected_event_id"].unique())
        event_groups = [top_events[:5], top_events[5:]]
        for collage_idx, event_group in enumerate(event_groups, start=1):
            if not event_group:
                continue
            fig, axes = plt.subplots(len(event_group) * 2, 1, figsize=(18, 5 * len(event_group)), sharex=False)
            if len(event_group) == 1:
                axes = [axes]
            for event_position, selected_event_id in enumerate(event_group):
                event_df = collage_df[collage_df["selected_event_id"] == selected_event_id]
                original_event_id = int(event_df["original_shared_event_id"].iloc[0])
                event_start = pd.to_datetime(event_df["event_start"].iloc[0])
                event_end = pd.to_datetime(event_df["event_end"].iloc[0])
                peak_rain_time = pd.to_datetime(event_df["peak_rain_intensity_time"].iloc[0])
                peak_water_time = pd.to_datetime(event_df["peak_canal_water_level_time"].iloc[0])
                lag_hours = event_df["lag_peak_rain_intensity_to_peak_water_level_h"].iloc[0]
                rainfall_axis = axes[event_position * 2]
                response_axis = axes[event_position * 2 + 1]
                response_twin = response_axis.twinx()

                for axis in [rainfall_axis, response_axis]:
                    axis.axvline(event_start, color="black", linestyle=":", linewidth=1)
                    axis.axvline(event_end, color="black", linestyle=":", linewidth=1)
                    axis.grid(True, alpha=0.3)

                rainfall_axis.plot(
                    event_df["timestamp"],
                    event_df["total_rainfall"],
                    color="tab:blue",
                    marker="o",
                    linewidth=1.5,
                    label=f"{node_id} total rainfall",
                )
                rainfall_axis.set_ylabel("Total rainfall")
                rainfall_axis.legend(loc="upper left")
                rainfall_axis.set_title(
                    f"{node_id} selected event {selected_event_id} "
                    f"(original shared {original_event_id})"
                )

                response_axis.plot(
                    event_df["timestamp"],
                    event_df["rain_intensity"],
                    color="tab:purple",
                    marker="o",
                    linewidth=1.5,
                    label="Rain intensity",
                )
                response_twin.plot(
                    event_df["timestamp"],
                    event_df["canal_water_level_preprocessed"],
                    color="tab:green" if node_id == "Node1" else "tab:orange",
                    marker="o",
                    linewidth=1.8,
                    label="Canal water level",
                )
                response_axis.axvline(peak_rain_time, color="tab:blue", linestyle="--", linewidth=1.5)
                response_twin.axvline(peak_water_time, color="tab:red", linestyle="--", linewidth=1.5)
                response_axis.set_ylabel("Rain intensity")
                response_twin.set_ylabel("Canal water level")
                response_axis.tick_params(axis="x", rotation=45)
                response_axis.legend(loc="upper left")
                response_twin.legend(loc="upper right")
                response_axis.text(
                    0.02,
                    0.88,
                    f"Rain peak: {peak_rain_time:%m-%d %H:%M}\n"
                    f"Water peak: {peak_water_time:%m-%d %H:%M}\n"
                    f"Lag: {lag_hours:.1f}h",
                    transform=response_axis.transAxes,
                    fontsize=8,
                    va="top",
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )

            fig.suptitle(
                f"{node_id} Preprocessed Selected Events {event_group[0]}-{event_group[-1]}",
                fontsize=15,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            fig.savefig(
                figure_root / f"{node_id.lower()}_preprocessed_merged_rain_intensity_water_level_collage_{collage_idx}.png",
                dpi=200,
            )
            plt.close(fig)


def build_summary(
    raw_df: pd.DataFrame,
    timestamp_validated: pd.DataFrame,
    cleaned_by_node: dict[str, pd.DataFrame],
    hourly_by_node: dict[str, pd.DataFrame],
    events_by_node: dict[str, pd.DataFrame],
    corr_by_node: dict[str, pd.DataFrame],
    shared_summary: pd.DataFrame,
) -> dict:
    summary = {
        "workflow": "event_first_preprocessing_rebuild",
        "raw_source": str(RAW_JSON_PATH),
        "raw_rows": int(len(raw_df)),
        "valid_timestamp_rows": int(timestamp_validated["timestamp_valid"].sum()),
        "invalid_timestamp_rows": int((~timestamp_validated["timestamp_valid"]).sum()),
        "duplicate_timestamp_rows_dropped": int(
            (timestamp_validated["duplicate_resolution"] == "dropped_keep_last").sum()
        ),
        "node_rules": NODE_RULES,
        "event_gap_hours": EVENT_GAP_HOURS,
        "minimum_event_duration_hours": MIN_EVENT_DURATION_HOURS,
        "shared_event_count": int(len(shared_summary)),
        "node_summaries": {},
    }
    for node_id, clean_df in cleaned_by_node.items():
        flags = clean_df["preprocessing_flag"].value_counts().to_dict()
        positive_corr = corr_by_node[node_id][corr_by_node[node_id]["is_positive"]]
        best_positive = positive_corr.sort_values("correlation", ascending=False).head(1)
        summary["node_summaries"][node_id] = {
            "cleaned_rows": int(len(clean_df)),
            "hourly_rows": int(len(hourly_by_node[node_id])),
            "event_count": int(len(events_by_node[node_id])),
            "preprocessing_flags": {str(key): int(value) for key, value in flags.items()},
            "best_positive_correlation_view": (
                best_positive.iloc[0]["view"] if not best_positive.empty else None
            ),
            "best_positive_correlation": (
                float(best_positive.iloc[0]["correlation"]) if not best_positive.empty else None
            ),
        }
    return summary


def write_evaluation_doc(summary: dict, shared_lag: pd.DataFrame, shared_corr: pd.DataFrame) -> None:
    lines = [
        "# Hydrologic Analysis Evaluation",
        "",
        "## Rebuild Status",
        "",
        f"- Workflow: `{summary['workflow']}`",
        f"- Raw rows: `{summary['raw_rows']}`",
        f"- Valid timestamp rows: `{summary['valid_timestamp_rows']}`",
        f"- Shared event count: `{summary['shared_event_count']}`",
        "",
        "## Node Results",
        "",
    ]
    for node_id, node_summary in summary["node_summaries"].items():
        lines.extend(
            [
                f"### {node_id}",
                "",
                f"- Cleaned rows: `{node_summary['cleaned_rows']}`",
                f"- Event count: `{node_summary['event_count']}`",
                f"- Best positive correlation view: `{node_summary['best_positive_correlation_view']}`",
                f"- Best positive correlation: `{node_summary['best_positive_correlation']}`",
                f"- Preprocessing flags: `{node_summary['preprocessing_flags']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Consistency Read",
            "",
            "- Use shared-event lag and correlation reports for cross-node claims.",
            "- Use node-specific event reports for local node behavior only.",
            "- Event-start spike corrections are traceable in `preprocessed/preprocessing_flags.csv`.",
            "- Positive correlation claims are listed by calculation view, not forced into one full-series result.",
        ]
    )
    if not shared_lag.empty:
        lines.append(
            f"- Median shared lag difference: `{shared_lag['lag_difference_h'].median()}` hours."
        )
    if not shared_corr.empty:
        positive_node1 = int((shared_corr["node1_event_corr"] > 0).sum())
        positive_node2 = int((shared_corr["node2_event_corr"] > 0).sum())
        lines.append(f"- Shared events with positive Node 1 event correlation: `{positive_node1}`.")
        lines.append(f"- Shared events with positive Node 2 event correlation: `{positive_node2}`.")

    (REPORTS_DIR / "hydrologic_analysis_evaluation.md").write_text("\n".join(lines))


def run_pipeline() -> PipelineArtifacts:
    ensure_output_dirs()
    raw_df = load_raw_json()
    timestamp_validated, resolved = validate_and_resolve_timestamps(raw_df)
    cleaned_by_node = clean_per_node(resolved)

    hourly_by_node = {}
    events_by_node = {}
    features_by_node = {}
    corr_by_node = {}
    event_ready_by_node = {}

    for node_id, clean_df in cleaned_by_node.items():
        hourly_df = build_hourly_dataset(clean_df)
        events_df = extract_rain_events(hourly_df)
        event_ready_df = assign_event_ids(clean_df, events_df)
        lag_df = calculate_lag_correlation(hourly_df)
        features_df = event_features(node_id, hourly_df, events_df)
        corr_df = correlation_search(node_id, hourly_df, events_df)

        hourly_by_node[node_id] = hourly_df
        events_by_node[node_id] = events_df
        features_by_node[node_id] = features_df
        corr_by_node[node_id] = corr_df
        event_ready_by_node[node_id] = event_ready_df

        write_node_outputs(
            node_id,
            clean_df,
            event_ready_df,
            hourly_df,
            events_df,
            features_df,
            lag_df,
            corr_df,
        )

    shared_df = shared_events(events_by_node)
    shared_summary, shared_lag, shared_corr = shared_event_metrics(
        shared_df, hourly_by_node, features_by_node
    )
    write_shared_outputs(shared_df, shared_summary, shared_lag, shared_corr, hourly_by_node)
    write_summary_figures(
        cleaned_by_node,
        hourly_by_node,
        events_by_node,
        shared_df,
        shared_summary,
        shared_lag,
        shared_corr,
    )
    write_selected_event_sheets(shared_df, hourly_by_node)
    write_preprocessed_selected_events(shared_df, hourly_by_node)

    flags = pd.concat(
        [
            clean_df[clean_df["preprocessing_flag"] != "valid"].assign(node_id=node_id)
            for node_id, clean_df in cleaned_by_node.items()
        ],
        ignore_index=True,
    )
    flags.to_csv(PREPROCESSED_DIR / "preprocessing_flags.csv", index=False)
    raw_df.to_csv(PREPROCESSED_DIR / "raw_flattened.csv", index=False)
    timestamp_validated.to_csv(PREPROCESSED_DIR / "timestamp_validated.csv", index=False)

    summary = build_summary(
        raw_df,
        timestamp_validated,
        cleaned_by_node,
        hourly_by_node,
        events_by_node,
        corr_by_node,
        shared_summary,
    )
    write_json(PREPROCESSED_DIR / "preprocessing_summary.json", summary)
    pd.DataFrame(
        [
            {"node_id": node_id, **node_summary}
            for node_id, node_summary in summary["node_summaries"].items()
        ]
    ).to_csv(REPORTS_DIR / "combined_node_summary.csv", index=False)
    write_evaluation_doc(summary, shared_lag, shared_corr)

    return PipelineArtifacts(
        raw_flattened=raw_df,
        timestamp_validated=timestamp_validated,
        cleaned_by_node=cleaned_by_node,
        hourly_by_node=hourly_by_node,
        summary=summary,
    )
