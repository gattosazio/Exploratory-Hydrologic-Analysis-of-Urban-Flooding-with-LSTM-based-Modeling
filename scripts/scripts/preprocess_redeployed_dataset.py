from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "datasets" / "raw" / "re-reployed_dataset.json"
PREPROCESSED_DIR = ROOT / "datasets" / "preprocessed_selected_events" / "redeployed"
FIGURE_DATA_DIR = ROOT / "figure_data" / "redeployed_preprocessed"
FIGURES_DIR = ROOT / "figures" / "redeployed_preprocessed"
REPORTS_DIR = ROOT / "reports" / "redeployed_preprocessed"

NODE_MAX_DEPTH = {"Node1": 55.0, "Node2": 75.0}
DRY_PERIOD_WATER_CAP = {"Node1": 5.0, "Node2": 15.0}
NEIGHBOR_WINDOW = pd.Timedelta(minutes=15)
LIGHT_RAIN_MM_HR = 2.5
MODERATE_RAIN_MM_HR = 7.5
OBSERVER_VALIDATED_MAX_RAIN_MM_HR = 2.49
OBSERVER_VALID_LIGHT_RAIN_DATES = {pd.Timestamp("2026-05-27").date()}


def get_nested(record, keys, default=None):
    value = record
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def flatten_raw(data):
    rows = []
    for node_id, records in data.items():
        if not isinstance(records, dict):
            continue
        for record_id, record in records.items():
            rows.append(
                {
                    "node_id": node_id,
                    "record_id": record_id,
                    "timestamp": record.get("timestamp"),
                    "send_reason": record.get("send_reason"),
                    "rain_intensity_raw_mm_hr": get_nested(record, ["rain_gauge", "processed", "rate_mm_per_hr"]),
                    "total_rainfall_raw_mm": get_nested(record, ["rain_gauge", "processed", "total_mm"]),
                    "mm_per_tip": get_nested(record, ["rain_gauge", "processed", "mm_per_tip"]),
                    "tips_in_window": get_nested(record, ["rain_gauge", "raw", "tips_in_window"]),
                    "tip_count": get_nested(record, ["rain_gauge", "raw", "tip_count"]),
                    "ignored_tip_count": get_nested(record, ["rain_gauge", "raw", "ignored_tip_count"]),
                    "is_raining_raw": get_nested(record, ["rain_gauge", "raw", "is_raining"]),
                    "last_tip_gap_ms": get_nested(record, ["rain_gauge", "raw", "last_tip_gap_ms"]),
                    "canal_water_level_raw_cm": get_nested(record, ["ultrasonic", "processed", "water_level_cm"]),
                    "filtered_distance_cm": get_nested(record, ["ultrasonic", "processed", "filtered_distance_cm"]),
                    "empty_distance_cm": get_nested(record, ["ultrasonic", "processed", "empty_distance_cm"]),
                    "max_water_level_cm": get_nested(record, ["ultrasonic", "processed", "max_water_level_cm"]),
                    "distance_cm": get_nested(record, ["ultrasonic", "raw", "distance_cm"]),
                    "valid_ultrasonic_raw": get_nested(record, ["ultrasonic", "raw", "valid"]),
                    "valid_samples": get_nested(record, ["ultrasonic", "raw", "valid_samples"]),
                }
            )
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["timestamp_valid"] = df["timestamp"].notna()
    df = df.sort_values(["node_id", "timestamp", "record_id"]).reset_index(drop=True)
    return df


def classify_rain(value):
    if pd.isna(value) or value <= 0:
        return "Dry"
    if value < LIGHT_RAIN_MM_HR:
        return "Light Rain"
    if value < MODERATE_RAIN_MM_HR:
        return "Moderate Rain"
    return "Heavy Rain"


def preprocess_node(group):
    group = group.copy().sort_values("timestamp").reset_index(drop=True)
    raw = pd.to_numeric(group["rain_intensity_raw_mm_hr"], errors="coerce").fillna(0)
    tips = pd.to_numeric(group["tips_in_window"], errors="coerce").fillna(0)
    is_raining = group["is_raining_raw"].fillna(False).astype(bool)
    valid_light_rain_date = group["timestamp"].dt.date.isin(OBSERVER_VALID_LIGHT_RAIN_DATES).fillna(False)
    cleaned = raw.copy()
    flags = pd.Series("kept", index=group.index, dtype="object")

    observer_dry_date = (raw > 0) & (~valid_light_rain_date)
    cleaned.loc[observer_dry_date] = 0
    flags.loc[observer_dry_date] = "observer_dry_period_rain_zeroed"

    dry_mask = (raw > 0) & (~is_raining) & (tips <= 0)
    cleaned.loc[dry_mask] = 0
    flags.loc[dry_mask] = "dry_period_zeroed"

    for idx, row in group.iterrows():
        if cleaned.loc[idx] <= 0 or pd.isna(row["timestamp"]):
            continue
        nearby = group["timestamp"].between(row["timestamp"] - NEIGHBOR_WINDOW, row["timestamp"] + NEIGHBOR_WINDOW)
        nearby.loc[idx] = False
        nearby_positive = (raw[nearby] > 0).any()
        if not nearby_positive and tips.loc[idx] <= 1:
            cleaned.loc[idx] = 0
            flags.loc[idx] = "isolated_spike_zeroed"
            continue
        if cleaned.loc[idx] >= MODERATE_RAIN_MM_HR and tips.loc[idx] <= 1:
            neighbor_values = raw[nearby & (raw > 0) & (raw < MODERATE_RAIN_MM_HR)]
            if len(neighbor_values) > 0:
                cleaned.loc[idx] = float(neighbor_values.median())
                flags.loc[idx] = "single_tip_high_spike_smoothed"

    observer_invalidated_moderate_heavy = cleaned >= LIGHT_RAIN_MM_HR
    cleaned.loc[observer_invalidated_moderate_heavy] = OBSERVER_VALIDATED_MAX_RAIN_MM_HR
    flags.loc[observer_invalidated_moderate_heavy] = "observer_invalidated_moderate_heavy_spike_capped"

    water_raw = pd.to_numeric(group["canal_water_level_raw_cm"], errors="coerce")
    node_id = str(group["node_id"].iloc[0])
    max_depth = pd.to_numeric(group["max_water_level_cm"], errors="coerce").fillna(NODE_MAX_DEPTH.get(node_id, 75.0))
    max_depth = max_depth.clip(lower=0)
    water_clean = water_raw.clip(lower=0)
    water_flags = pd.Series("kept", index=group.index, dtype="object")
    high_mask = water_clean > max_depth
    water_clean.loc[high_mask] = max_depth.loc[high_mask]
    water_flags.loc[high_mask] = "physical_range_capped"
    dry_water_cap = DRY_PERIOD_WATER_CAP.get(node_id, 15.0)
    dry_water_spike = (~valid_light_rain_date) & (water_clean > dry_water_cap)
    water_clean.loc[dry_water_spike] = dry_water_cap
    water_flags.loc[dry_water_spike] = "observer_dry_period_water_spike_capped"

    rolling_median = water_clean.rolling(window=5, center=True, min_periods=3).median()
    residual = (water_clean - rolling_median).abs()
    isolated_water_spike = residual > 8
    water_clean.loc[isolated_water_spike] = rolling_median.loc[isolated_water_spike]
    water_flags.loc[isolated_water_spike] = "isolated_water_spike_smoothed"

    group["rain_intensity_preprocessed_mm_hr"] = cleaned.round(3)
    group["rain_intensity_flag"] = flags
    group["rain_class"] = group["rain_intensity_preprocessed_mm_hr"].map(classify_rain)
    group["canal_water_level_preprocessed_cm"] = water_clean.round(3)
    group["water_level_flag"] = water_flags
    group["time_delta_min"] = group["timestamp"].diff().dt.total_seconds().div(60).round(3)
    return group


def write_json_records(df, path):
    records = df.copy()
    for column in records.columns:
        if pd.api.types.is_datetime64_any_dtype(records[column]):
            records[column] = records[column].dt.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(records.to_dict(orient="records"), indent=2), encoding="utf-8")


def make_figures(df):
    for node_id, group in df.groupby("node_id"):
        group = group.sort_values("timestamp")
        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        axes[0].plot(group["timestamp"], group["rain_intensity_raw_mm_hr"], label="Raw rain intensity", color="#d62728", alpha=0.45)
        axes[0].plot(group["timestamp"], group["rain_intensity_preprocessed_mm_hr"], label="Preprocessed rain intensity", color="#1f77b4")
        axes[0].set_ylabel("Rain Intensity (mm/hr)")
        axes[0].legend()
        axes[0].grid(alpha=0.25)

        axes[1].plot(group["timestamp"], group["canal_water_level_raw_cm"], label="Raw canal water level", color="#ff7f0e", alpha=0.45)
        axes[1].plot(group["timestamp"], group["canal_water_level_preprocessed_cm"], label="Preprocessed canal water level", color="#2ca02c")
        axes[1].set_ylabel("Canal Water Level (cm)")
        axes[1].set_xlabel("Timestamp")
        axes[1].legend()
        axes[1].grid(alpha=0.25)

        fig.suptitle(f"{node_id} Redeployed Dataset: Raw and Preprocessed Hydrologic Variables")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"{node_id.lower()}_redeployed_raw_vs_preprocessed.png", dpi=200)
        plt.close(fig)

    rain_counts = df.groupby(["node_id", "rain_class"]).size().unstack(fill_value=0)
    rain_counts = rain_counts.reindex(columns=["Dry", "Light Rain", "Moderate Rain", "Heavy Rain"], fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    rain_counts.plot(kind="bar", ax=ax)
    ax.set_title("Redeployed Dataset Rainfall Intensity Classification")
    ax.set_ylabel("Number of Records")
    ax.set_xlabel("Node")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "redeployed_rain_class_counts.png", dpi=200)
    plt.close(fig)


def make_reports(df):
    summary = (
        df.groupby("node_id")
        .agg(
            rows=("record_id", "count"),
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            raw_rain_max_mm_hr=("rain_intensity_raw_mm_hr", "max"),
            preprocessed_rain_max_mm_hr=("rain_intensity_preprocessed_mm_hr", "max"),
            raw_water_max_cm=("canal_water_level_raw_cm", "max"),
            preprocessed_water_max_cm=("canal_water_level_preprocessed_cm", "max"),
            timestamp_valid_count=("timestamp_valid", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(REPORTS_DIR / "redeployed_preprocessing_summary.csv", index=False)
    write_json_records(summary, REPORTS_DIR / "redeployed_preprocessing_summary.json")

    rain_flags = df.groupby(["node_id", "rain_intensity_flag"]).size().reset_index(name="records")
    rain_flags.to_csv(REPORTS_DIR / "redeployed_rain_intensity_flag_summary.csv", index=False)

    water_flags = df.groupby(["node_id", "water_level_flag"]).size().reset_index(name="records")
    water_flags.to_csv(REPORTS_DIR / "redeployed_water_level_flag_summary.csv", index=False)

    rain_classes = df.groupby(["node_id", "rain_class"]).size().reset_index(name="records")
    rain_classes.to_csv(REPORTS_DIR / "redeployed_rain_class_summary.csv", index=False)
    method = [
        "# Redeployed Dataset Preprocessing Method",
        "",
        "Source dataset: `datasets/raw/re-reployed_dataset.json`.",
        "",
        "Raw rainfall and water-level values were preserved. Preprocessed fields were added for analysis and visualization.",
        "",
        "## Rain Intensity Rules",
        "",
        "- Positive rain intensity during non-raining records with zero tips was set to 0.",
        "- Isolated one-record rain spikes without nearby rainfall within 15 minutes were set to 0.",
        "- Single-tip high spikes were smoothed only when nearby lower rain values supported correction.",
        "- Moderate- and heavy-rain values were capped to light-rain intensity because field observation confirmed that only dry to light-rain conditions occurred during the redeployed observation period.",
        "- May 27, 2026 was treated as the only observer-validated light-rain date.",
        "- Rainfall sensor tips outside May 27, 2026 were treated as dry-period tipping noise and set to 0.",
        "- Rain classes used: Dry = 0, Light Rain = <2.5 mm/hr, Moderate Rain = 2.5 to <7.5 mm/hr, Heavy Rain = >=7.5 mm/hr.",
        "",
        "## Canal Water-Level Rules",
        "",
        "- Negative water levels were clipped to 0 cm.",
        "- Water levels above the node maximum depth were capped using the sensor-reported maximum depth.",
        "- Dry-period water spikes were capped using node-specific dry-period limits: Node 1 = 5 cm and Node 2 = 15 cm.",
        "- Isolated water-level spikes greater than 8 cm from the local rolling median were smoothed.",
        "- Node 1 maximum depth is 55 cm; Node 2 maximum depth is 75 cm.",
        "",
        "## Output Summary",
        "",
        "```",
        summary.to_string(index=False),
        "```",
    ]
    (REPORTS_DIR / "redeployed_preprocessing_method.md").write_text("\n".join(method), encoding="utf-8")
    return summary


def main():
    for directory in [PREPROCESSED_DIR, FIGURE_DATA_DIR, FIGURES_DIR, REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    data = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    raw_df = flatten_raw(data)
    processed = pd.concat([preprocess_node(group) for _, group in raw_df.groupby("node_id")], ignore_index=True)
    processed = processed.sort_values(["node_id", "timestamp", "record_id"]).reset_index(drop=True)

    processed.to_csv(PREPROCESSED_DIR / "redeployed_preprocessed.csv", index=False)
    processed.to_csv(FIGURE_DATA_DIR / "redeployed_preprocessed.csv", index=False)
    write_json_records(processed, PREPROCESSED_DIR / "redeployed_preprocessed.json")
    write_json_records(processed, FIGURE_DATA_DIR / "redeployed_preprocessed.json")

    for node_id, group in processed.groupby("node_id"):
        safe_node = node_id.lower()
        group.to_csv(PREPROCESSED_DIR / f"{safe_node}_redeployed_preprocessed.csv", index=False)
        write_json_records(group, PREPROCESSED_DIR / f"{safe_node}_redeployed_preprocessed.json")

    make_figures(processed)
    summary = make_reports(processed)
    print(summary.to_string(index=False))
    print(f"Wrote: {PREPROCESSED_DIR / 'redeployed_preprocessed.csv'}")
    print(f"Wrote figures: {FIGURES_DIR}")
    print(f"Wrote reports: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
