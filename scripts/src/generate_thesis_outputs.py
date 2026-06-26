import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT
FIGURE_DATA = REPO_ROOT / "figure_data"
PREPROCESSED = REPO_ROOT / "datasets" / "preprocessed_selected_events"
REDEPLOYED = REPO_ROOT / "datasets" / "raw"
THESIS_ROOT = REPO_ROOT
FIGURE_ROOT = REPO_ROOT / "figures"
TABLE_ROOT = REPO_ROOT / "reports"
MARKDOWN_ROOT = REPO_ROOT / "reports" / "markdown"

FILTER_START = pd.Timestamp("2025-12-01 00:00:00")
FILTER_END = pd.Timestamp("2026-01-18 23:59:59")

FIGURE_DIRS = {
    "preprocessed": FIGURE_ROOT / "descriptive-statistics",
    "relationship": FIGURE_ROOT / "pearson_correlation",
    "timeseries": FIGURE_ROOT / "time-series",
    "selected": FIGURE_ROOT / "merged_selected_events",
    "lag": FIGURE_ROOT / "lag_analysis",
    "intensity": FIGURE_ROOT / "rain_intensity_classification",
    "final": FIGURE_ROOT / "merged_selected_events",
}

INTENSITY_THRESHOLDS = [
    ("Light Rain", 0, 2.5),
    ("Moderate Rain", 2.5, 7.5),
    ("Heavy Rain", 7.5, np.inf),
]


def reset_output_dirs() -> None:
    if TABLE_ROOT.exists():
        shutil.rmtree(TABLE_ROOT)
    MARKDOWN_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    for path in FIGURE_DIRS.values():
        path.mkdir(parents=True, exist_ok=True)


def write_table(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False)
    df.to_json(path.with_suffix(".json"), orient="records", indent=2, date_format="iso")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n")


def classify_intensity(value: float) -> str:
    for label, lower, upper in INTENSITY_THRESHOLDS:
        if lower <= value < upper:
            return label
    return "Unclassified"


def safe_number(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "not available"
    return f"{value:.{digits}f}"


def load_data() -> dict[str, pd.DataFrame]:
    return {
        "node1_hourly": pd.read_csv(PREPROCESSED / "node1_hourly.csv", parse_dates=["timestamp"]),
        "node2_hourly": pd.read_csv(PREPROCESSED / "node2_hourly.csv", parse_dates=["timestamp"]),
        "selected_summary": pd.read_csv(
            FIGURE_DATA / "preprocessed-selected-events" / "preprocessed_selected_event_summary.csv",
            parse_dates=[
                "event_start",
                "event_end",
                "peak_rain_intensity_time",
                "peak_canal_water_level_time",
            ],
        ),
        "merged_selected": pd.read_csv(
            FIGURE_DATA
            / "preprocessed-selected-events"
            / "merged_nodes"
            / "preprocessed_selected_events_all_merged_nodes.csv",
            parse_dates=["timestamp", "event_start", "event_end"],
        ),
    }


def filter_hourly(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    mask = df["timestamp"].between(FILTER_START, FILTER_END, inclusive="both")
    return df.loc[~mask].copy(), int(mask.sum())


def filtered_hourly(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    node1, node1_removed = filter_hourly(data["node1_hourly"])
    node2, node2_removed = filter_hourly(data["node2_hourly"])
    summary = pd.DataFrame(
        [
            {
                "node_id": "Node1",
                "excluded_start": FILTER_START,
                "excluded_end": FILTER_END,
                "rows_before_filter": len(data["node1_hourly"]),
                "rows_removed": node1_removed,
                "rows_after_filter": len(node1),
                "first_timestamp_after_filter": node1["timestamp"].min(),
                "last_timestamp_after_filter": node1["timestamp"].max(),
            },
            {
                "node_id": "Node2",
                "excluded_start": FILTER_START,
                "excluded_end": FILTER_END,
                "rows_before_filter": len(data["node2_hourly"]),
                "rows_removed": node2_removed,
                "rows_after_filter": len(node2),
                "first_timestamp_after_filter": node2["timestamp"].min(),
                "last_timestamp_after_filter": node2["timestamp"].max(),
            },
        ]
    )
    write_table(TABLE_ROOT / "filtered_dataset_summary", summary)
    return {"Node1": node1, "Node2": node2}


def make_analysis_tables(data: dict[str, pd.DataFrame], filtered: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    descriptive_rows = []
    selected = data["merged_selected"].copy()
    selected = selected.loc[selected["is_selected_event_window"].astype(bool)].copy()
    selected_variables = {
        "Node1": {
            "total_rainfall": "node1_total_rainfall",
            "rain_intensity": "node1_rain_intensity",
            "cumulative_total_rainfall": "node1_cumulative_total_rainfall",
            "canal_water_level": "node1_canal_water_level_preprocessed",
        },
        "Node2": {
            "total_rainfall": "node2_total_rainfall",
            "rain_intensity": "node2_rain_intensity",
            "cumulative_total_rainfall": "node2_cumulative_total_rainfall",
            "canal_water_level": "node2_canal_water_level_preprocessed",
        },
    }
    for node_id, variables in selected_variables.items():
        for variable, column in variables.items():
            values = selected[column].dropna()
            descriptive_rows.append(
                {
                    "node_id": node_id,
                    "variable": variable,
                    "mean": values.mean(),
                    "median": values.median(),
                    "std": values.std(),
                    "min": values.min(),
                    "max": values.max(),
                }
            )
    descriptive = pd.DataFrame(descriptive_rows)
    write_table(TABLE_ROOT / "selected_event_descriptive_statistics", descriptive)

    filtered_descriptive_rows = []
    for node_id, df in filtered.items():
        rename = {
            "rainfall": "rainfall",
            "rainrate": "rain_intensity",
            "waterlevel_clean": "canal_water_level",
        }
        working = df.rename(columns=rename)
        working["accumulated_rainfall_24h"] = working["rainfall"].rolling(24, min_periods=1).sum()
        for variable in ["rainfall", "rain_intensity", "accumulated_rainfall_24h", "canal_water_level"]:
            filtered_descriptive_rows.append(
                {
                    "node_id": node_id,
                    "variable": variable,
                    "mean": working[variable].mean(),
                    "median": working[variable].median(),
                    "std": working[variable].std(),
                    "min": working[variable].min(),
                    "max": working[variable].max(),
                }
            )
    write_table(TABLE_ROOT / "filtered_descriptive_statistics", pd.DataFrame(filtered_descriptive_rows))

    merged = filtered["Node1"][
        ["timestamp", "rainfall", "rainrate", "waterlevel_clean"]
    ].merge(
        filtered["Node2"][["timestamp", "rainfall", "rainrate", "waterlevel_clean"]],
        on="timestamp",
        suffixes=("_node1", "_node2"),
    )
    merged["accumulated_rainfall_24h_node1"] = merged["rainfall_node1"].rolling(24, min_periods=1).sum()
    merged["accumulated_rainfall_24h_node2"] = merged["rainfall_node2"].rolling(24, min_periods=1).sum()
    relationship = merged.rename(
        columns={
            "rainrate_node1": "rain_intensity_node1",
            "rainrate_node2": "rain_intensity_node2",
            "waterlevel_clean_node1": "canal_water_level_node1",
            "waterlevel_clean_node2": "canal_water_level_node2",
        }
    )
    corr_cols = [
        "rainfall_node1",
        "rain_intensity_node1",
        "accumulated_rainfall_24h_node1",
        "canal_water_level_node1",
        "rainfall_node2",
        "rain_intensity_node2",
        "accumulated_rainfall_24h_node2",
        "canal_water_level_node2",
    ]
    correlation = relationship[corr_cols].corr(method="pearson")
    correlation_out = correlation.reset_index().rename(columns={"index": "variable"})
    write_table(TABLE_ROOT / "filtered_pearson_correlation_matrix", correlation_out)

    selected = data["selected_summary"].copy()
    event_class = (
        selected.groupby("selected_event_id", as_index=False)
        .agg(
            original_shared_event_id=("original_shared_event_id", "first"),
            event_start=("event_start", "first"),
            event_end=("event_end", "first"),
            event_duration_hours=("event_duration_hours", "first"),
            peak_rain_intensity=("peak_rain_intensity", "max"),
            mean_peak_lag_h=("lag_peak_rain_intensity_to_peak_water_level_h", "mean"),
            mean_first_response_lag_h=("lag_from_event_start_h", "mean"),
        )
    )
    event_class["rainfall_intensity_class"] = event_class["peak_rain_intensity"].apply(classify_intensity)
    write_table(TABLE_ROOT / "selected_event_intensity_classification", event_class)

    lag_stats = (
        selected.groupby("node_id", as_index=False)
        .agg(
            event_count=("selected_event_id", "count"),
            mean_peak_lag_h=("lag_peak_rain_intensity_to_peak_water_level_h", "mean"),
            median_peak_lag_h=("lag_peak_rain_intensity_to_peak_water_level_h", "median"),
            std_peak_lag_h=("lag_peak_rain_intensity_to_peak_water_level_h", "std"),
            min_peak_lag_h=("lag_peak_rain_intensity_to_peak_water_level_h", "min"),
            max_peak_lag_h=("lag_peak_rain_intensity_to_peak_water_level_h", "max"),
            mean_first_response_lag_h=("lag_from_event_start_h", "mean"),
            median_first_response_lag_h=("lag_from_event_start_h", "median"),
        )
    )
    write_table(TABLE_ROOT / "selected_event_lag_statistics", lag_stats)

    outliers = []
    for node_id, group in selected.groupby("node_id"):
        lag = group["lag_peak_rain_intensity_to_peak_water_level_h"].dropna()
        q1 = lag.quantile(0.25)
        q3 = lag.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        for _, row in group.iterrows():
            value = row["lag_peak_rain_intensity_to_peak_water_level_h"]
            reasons = []
            if value < lower or value > upper:
                reasons.append("IQR outlier")
            if value < 0:
                reasons.append("water-level peak precedes rain-intensity peak")
            if reasons:
                outliers.append(
                    {
                        "node_id": node_id,
                        "selected_event_id": row["selected_event_id"],
                        "original_shared_event_id": row["original_shared_event_id"],
                        "lag_peak_rain_intensity_to_peak_water_level_h": value,
                        "outlier_reason": "; ".join(reasons),
                    }
                )
    outlier_df = pd.DataFrame(outliers)
    write_table(TABLE_ROOT / "selected_event_lag_outliers", outlier_df)

    redeployed_files = []
    if REDEPLOYED.exists():
        for file_path in sorted(REDEPLOYED.rglob("*")):
            if file_path.is_file():
                redeployed_files.append(
                    {
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "size_bytes": file_path.stat().st_size,
                    }
                )
    redeployed_inventory = pd.DataFrame(redeployed_files)
    write_table(TABLE_ROOT / "redeployed_dataset_inventory", redeployed_inventory)

    return {
        "descriptive": descriptive,
        "relationship": relationship,
        "correlation": correlation,
        "event_class": event_class,
        "lag_stats": lag_stats,
        "outliers": outlier_df,
        "redeployed_inventory": redeployed_inventory,
    }


def plot_filtered_node_timeseries(filtered: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    for node_id, df in filtered.items():
        label = node_id.replace("Node", "Node ")
        axes[0].plot(df["timestamp"], df["rainfall"], label=label, linewidth=1)
        axes[1].plot(df["timestamp"], df["rainrate"], label=label, linewidth=1)
        axes[2].plot(df["timestamp"], df["waterlevel_clean"], label=label, linewidth=1)
    axes[0].set_ylabel("Rainfall")
    axes[1].set_ylabel("Rain intensity")
    axes[2].set_ylabel("Canal water level")
    axes[2].set_xlabel("Timestamp")
    for axis in axes:
        axis.legend()
        axis.grid(True, alpha=0.25)
    fig.suptitle("Node Time Series")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = FIGURE_DIRS["timeseries"] / "filtered_node_timeseries.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_shared_timeseries(tables: dict[str, pd.DataFrame]) -> Path:
    relationship = tables["relationship"]
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    axes[0].plot(relationship["timestamp"], relationship["rain_intensity_node1"], label="Node 1 rain intensity")
    axes[0].plot(relationship["timestamp"], relationship["rain_intensity_node2"], label="Node 2 rain intensity")
    axes[1].plot(relationship["timestamp"], relationship["canal_water_level_node1"], label="Node 1 canal water level")
    axes[1].plot(relationship["timestamp"], relationship["canal_water_level_node2"], label="Node 2 canal water level")
    axes[0].set_ylabel("Rain intensity")
    axes[1].set_ylabel("Canal water level")
    axes[1].set_xlabel("Timestamp")
    for axis in axes:
        axis.legend()
        axis.grid(True, alpha=0.25)
    fig.suptitle("Shared Filtered Rain Intensity and Canal Water-Level Time Series")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = FIGURE_DIRS["timeseries"] / "shared_filtered_timeseries.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_descriptive_statistics(tables: dict[str, pd.DataFrame]) -> Path:
    descriptive = tables["descriptive"].copy()
    variable_labels = {
        "total_rainfall": "Total rainfall",
        "rain_intensity": "Rain intensity",
        "cumulative_total_rainfall": "Cumulative total rainfall",
        "canal_water_level": "Canal water level",
    }
    descriptive["variable_label"] = descriptive["variable"].map(variable_labels)
    variables = list(variable_labels.values())
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes_flat = axes.flatten()
    colors = {"Node1": "#1f77b4", "Node2": "#ff7f0e"}
    for axis, variable in zip(axes_flat, variables):
        subset = descriptive.loc[descriptive["variable_label"] == variable].set_index("node_id")
        node_ids = [node_id for node_id in ["Node1", "Node2"] if node_id in subset.index]
        positions = range(len(node_ids))
        means = [subset.loc[node_id, "mean"] for node_id in node_ids]
        medians = [subset.loc[node_id, "median"] for node_id in node_ids]
        deviations = [subset.loc[node_id, "std"] for node_id in node_ids]
        axis.bar(
            [position - 0.18 for position in positions],
            means,
            width=0.36,
            yerr=deviations,
            capsize=4,
            label="Mean ± SD",
            color=[colors[node_id] for node_id in node_ids],
            alpha=0.75,
        )
        axis.bar(
            [position + 0.18 for position in positions],
            medians,
            width=0.36,
            label="Median",
            color=[colors[node_id] for node_id in node_ids],
            alpha=0.35,
        )
        axis.set_title(variable)
        axis.set_xticks(list(positions))
        axis.set_xticklabels([node_id.replace("Node", "Node ") for node_id in node_ids])
        axis.grid(True, axis="y", alpha=0.25)
    axes_flat[0].legend()
    figure.suptitle("Descriptive Statistics of the Hydrologic Variables")
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    path = FIGURE_DIRS["preprocessed"] / "descriptive_statistics_summary.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def plot_correlation_heatmap(correlation: pd.DataFrame) -> Path:
    label_map = {
        "rainfall_node1": "Node 1\nrainfall",
        "rain_intensity_node1": "Node 1\nrain intensity",
        "accumulated_rainfall_24h_node1": "Node 1\n24-h rainfall",
        "canal_water_level_node1": "Node 1\ncanal level",
        "rainfall_node2": "Node 2\nrainfall",
        "rain_intensity_node2": "Node 2\nrain intensity",
        "accumulated_rainfall_24h_node2": "Node 2\n24-h rainfall",
        "canal_water_level_node2": "Node 2\ncanal level",
    }
    mask = np.triu(np.ones(correlation.shape, dtype=bool), k=1)
    masked_values = correlation.to_numpy(dtype=float, copy=True)
    masked_values[mask] = np.nan
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(masked_values, cmap="RdYlBu_r", vmin=-1, vmax=1)
    labels = [label_map.get(column, column) for column in correlation.columns]
    ax.set_xticks(range(len(correlation.columns)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(range(len(correlation.index)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(-0.5, len(correlation.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(correlation.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row in range(len(correlation.index)):
        for col in range(len(correlation.columns)):
            if mask[row, col]:
                continue
            value = correlation.iloc[row, col]
            text_color = "white" if abs(value) >= 0.55 else "black"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=9, color=text_color)
    ax.set_title("Pearson Correlation of Hydrologic Variables")
    ax.text(
        0,
        -0.12,
        "Pearson r: +1 = strong direct linear relation; 0 = weak/no linear relation; -1 = strong inverse relation.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.colorbar(image, ax=ax, shrink=0.75, label="Pearson r")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    path = FIGURE_DIRS["relationship"] / "filtered_pearson_correlation_heatmap.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_lag_distribution(tables: dict[str, pd.DataFrame], data: dict[str, pd.DataFrame]) -> Path:
    summary = data["selected_summary"]
    groups = [
        summary.loc[summary["node_id"] == "Node1", "lag_peak_rain_intensity_to_peak_water_level_h"].dropna(),
        summary.loc[summary["node_id"] == "Node2", "lag_peak_rain_intensity_to_peak_water_level_h"].dropna(),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(groups, tick_labels=["Node 1", "Node 2"], showmeans=True)
    ax.axhspan(2, 3, color="tab:green", alpha=0.12, label="2-3 h reference range")
    ax.set_ylabel("Peak-to-peak lag (hours)")
    ax.set_title("Selected-Event Lag Distribution")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIGURE_DIRS["lag"] / "selected_event_lag_distribution.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_intensity_counts(tables: dict[str, pd.DataFrame]) -> Path:
    counts = (
        tables["event_class"]["rainfall_intensity_class"]
        .value_counts()
        .reindex(["Light Rain", "Moderate Rain", "Heavy Rain"])
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index, counts.values, color=["#80b1d3", "#fdb462", "#fb8072"])
    for idx, value in enumerate(counts.values):
        ax.text(idx, value, str(int(value)), ha="center", va="bottom")
    ax.set_ylabel("Number of selected events")
    ax.set_title("Rainfall Intensity Classification of Selected Events")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIGURE_DIRS["intensity"] / "selected_event_intensity_classes.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def copy_selected_event_collage() -> Path | None:
    source = (
        OUTPUT_ROOT
        / "figures"
        / "preprocessed-selected-events"
        / "merged_nodes"
        / "preprocessed_selected_events_merged_nodes_collage.png"
    )
    if not source.exists():
        return None
    target = FIGURE_DIRS["selected"] / "merged_nodes_selected_event_collage.png"
    shutil.copy2(source, target)
    shutil.copy2(source, FIGURE_DIRS["final"] / "merged_nodes_selected_event_collage.png")
    return target


def generate_figures(filtered: dict[str, pd.DataFrame], tables: dict[str, pd.DataFrame], data: dict[str, pd.DataFrame]) -> list[dict[str, str]]:
    figures = [
        {
            "figure_title": "Filtered Node Time Series",
            "purpose": "Shows rainfall, rainfall intensity, and canal water level after excluding December 2025 to January 18, 2026.",
            "dataset_source": "preprocessed/node1_hourly.csv; preprocessed/node2_hourly.csv",
            "chapter_placement": "Results chapter, time-series analysis",
            "file_path": str(plot_filtered_node_timeseries(filtered)),
        },
        {
            "figure_title": "Descriptive Statistics of the Hydrologic Variables",
            "purpose": "Summarizes mean, median, and variability for total rainfall, rain intensity, cumulative rainfall, and canal water level by node.",
            "dataset_source": "output/figure_data/preprocessed-selected-events/merged_nodes/preprocessed_selected_events_all_merged_nodes.csv",
            "chapter_placement": "Results chapter, descriptive statistics",
            "file_path": str(plot_descriptive_statistics(tables)),
        },
        {
            "figure_title": "Shared Filtered Rain Intensity and Canal Water-Level Time Series",
            "purpose": "Compares node-level temporal behavior without redundant rainfall panels.",
            "dataset_source": "filtered merged hourly datasets",
            "chapter_placement": "Results chapter, shared time-series analysis",
            "file_path": str(plot_shared_timeseries(tables)),
        },
        {
            "figure_title": "Pearson Correlation of Hydrologic Variables",
            "purpose": "Evaluates hydrologic relationships among rainfall intensity, accumulated rainfall, and canal water level.",
            "dataset_source": "filtered merged hourly datasets",
            "chapter_placement": "Results chapter, variable relationship analysis",
            "file_path": str(plot_correlation_heatmap(tables["correlation"])),
        },
        {
            "figure_title": "Selected-Event Lag Distribution",
            "purpose": "Evaluates whether the 2-3 hour median lag remains valid across nodes.",
            "dataset_source": "preprocessed selected event summary",
            "chapter_placement": "Results chapter, lag analysis",
            "file_path": str(plot_lag_distribution(tables, data)),
        },
        {
            "figure_title": "Rainfall Intensity Classification of Selected Events",
            "purpose": "Shows the distribution of selected events by rainfall intensity class.",
            "dataset_source": "selected event intensity classification table",
            "chapter_placement": "Results chapter, rainfall intensity classification",
            "file_path": str(plot_intensity_counts(tables)),
        },
    ]
    collage = copy_selected_event_collage()
    if collage is not None:
        figures.append(
            {
                "figure_title": "Merged Node Selected Rainfall-Response Events",
                "purpose": "Shows common event patterns, node differences, and lag behavior using rain intensity and canal water level only.",
                "dataset_source": "output/figure_data/preprocessed-selected-events/merged_nodes",
                "chapter_placement": "Results chapter, rainfall event pattern analysis",
                "file_path": str(collage),
            }
        )
    inventory = pd.DataFrame(figures)
    write_table(THESIS_ROOT / "figure_inventory", inventory)
    return figures


def make_methodology_review() -> None:
    text = """# Methodology Chapter Review

The current chapter and section titles should remain largely intact. The methodology chapter should be strengthened by clarifying the analytical workflow rather than by redesigning the structure.

## Data Preparation

This section should describe the raw sensor datasets, timestamp synchronization, variable organization, node-specific datasets, and shared datasets. It should clearly define rainfall, rainfall intensity, accumulated rainfall, canal water level, timestamp, node identifier, and selected rainfall event.

## Data Processing

This section should describe missing-value handling, duplicate timestamp handling, temporal alignment, sensor anomaly treatment, and feature preparation. The wording should emphasize that preprocessing creates analysis-ready datasets while preserving traceability to observed records.

## Rainfall Event Extraction and Selection Criteria

This section should remain in the methodology chapter. It should define rainfall-event identification, dry-period separation, minimum duration, selected-event rules, and validation checks. The selected-event criteria are methodological decisions and should be explained before results are presented.

## Event-Based Hydrologic Analysis

The procedural definition of event-based analysis belongs in methodology, but the interpretation belongs in results. The methodology chapter should only define how rainfall response, rainfall-water-level lag, node comparison, and hydrologic interpretation were performed. Detailed findings should remain in the results chapter.
"""
    write_markdown(THESIS_ROOT / "methodology_chapter_review.md", text)


def make_eda_framework() -> None:
    text = """# Exploratory Data Analysis Framework

The EDA should use only the filtered preprocessed datasets. The purpose is to support hydrologic interpretation and forecasting design, not to present every available plot.

## Descriptive Statistics

Report mean, median, standard deviation, minimum, and maximum for rainfall, rainfall intensity, accumulated rainfall, and canal water level for each node.

## Variable Relationship Analysis

Include rainfall intensity versus accumulated rainfall, rainfall intensity versus canal water level, accumulated rainfall versus canal water level, and canal water level versus rainfall variables. Use the Pearson correlation heatmap as the main figure, but interpret the hydrologic meaning rather than only the coefficient values.

## Time-Series Analysis

Use one filtered node-level time-series figure and one shared time-series figure. These are sufficient to show event occurrence, response timing, and node-level behavior without excessive redundancy.

## Rainfall Event Pattern Analysis

Use the merged selected-event collage as the central event-pattern figure. This figure directly supports discussion of common rainfall patterns, water-level response patterns, lag timing, node similarities, and node differences.
"""
    write_markdown(THESIS_ROOT / "eda_framework.md", text)


def make_hydrologic_findings(tables: dict[str, pd.DataFrame]) -> None:
    lag = tables["lag_stats"]
    classes = tables["event_class"]["rainfall_intensity_class"].value_counts().to_dict()
    node1 = lag[lag["node_id"] == "Node1"].iloc[0]
    node2 = lag[lag["node_id"] == "Node2"].iloc[0]
    text = f"""# Hydrologic Findings Summary

The selected-event descriptive analysis should focus on hydrologic behavior rather than on graph volume. The selected rainfall events show recurring delayed canal response after rainfall-intensity peaks. Across the selected events, Node 1 has a median peak-to-peak lag of {safe_number(node1['median_peak_lag_h'])} h, while Node 2 has a median peak-to-peak lag of {safe_number(node2['median_peak_lag_h'])} h. The first-response lag remains close to 2 h for both nodes, supporting the interpretation that the initial canal response commonly occurs within the 2-3 h window.

The selected event set is dominated by moderate-to-heavy events: Light Rain = {int(classes.get('Light Rain', 0))}, Moderate Rain = {int(classes.get('Moderate Rain', 0))}, and Heavy Rain = {int(classes.get('Heavy Rain', 0))}. This means the existing event dataset is strongest for explaining moderate-to-heavy rainfall response, while redeployed dry-period and light-rain datasets are needed to strengthen low-intensity baseline interpretation.

The merged selected-event figures show that Node 2 often has stronger rain-intensity peaks, while both nodes can display delayed canal response. Long peak lags occur mainly when rainfall arrives in repeated pulses. These long-lag events should be treated as hydrologic exceptions or compound-pulse responses, not as the typical response pattern.

For LSTM modeling, the key implication is that same-hour rainfall variables are insufficient. The model should include antecedent rainfall intensity, accumulated rainfall, prior canal water level, and lagged variables. A 2-3 h horizon is defendable for first response, while longer horizons should be evaluated for peak water-level prediction.
"""
    write_markdown(THESIS_ROOT / "hydrologic_findings_summary.md", text)


def make_event_pattern_analysis(tables: dict[str, pd.DataFrame]) -> None:
    outlier_count = len(tables["outliers"])
    text = f"""# Rainfall Event Pattern Analysis

The merged selected-event collage presents the selected rainfall-response events for Node 1 and Node 2 using rainfall intensity and canal water level. The figure shows that most events follow a consistent hydrologic sequence: rainfall intensity increases first, canal water level rises after a delay, and the canal level gradually recedes after the rainfall pulse weakens or ends. This pattern indicates that the canal response is not instantaneous. Instead, rainfall must first accumulate and move through the drainage area before producing a measurable rise in canal water level.

The rain-intensity peaks represent the strongest rainfall periods within each selected event. The water-level peaks represent the maximum canal response after those rainfall inputs. The time difference between these two peaks is interpreted as the rainfall-to-water-level response lag. Shorter lags indicate a faster canal response, while longer lags indicate delayed runoff contribution, repeated rainfall pulses, or slower drainage response. In several events, the canal level continues to rise even after the rain-intensity peak has already passed, which supports the use of lagged rainfall variables in the forecasting model.

Node behavior is similar in timing but not identical in magnitude. Node 2 commonly shows higher canal water levels and stronger or more sustained responses than Node 1, while both nodes show delayed water-level response after rainfall. Events with multiple rainfall pulses are more difficult to interpret because a later rain pulse can produce the highest canal level, resulting in a longer peak-to-peak lag. These cases should not be treated as errors automatically; they may represent compound rainfall-response behavior. However, {outlier_count} lag outlier records should still be discussed separately.

This figure is directly related to the study because it provides visual evidence that rainfall intensity and antecedent rainfall conditions affect later canal water-level behavior. For LSTM forecasting, the result suggests that same-hour rainfall alone is insufficient. The model should include lagged rainfall intensity, cumulative rainfall, previous canal water level, and node-specific behavior to capture the delayed hydrologic response.

Rainfall occurrence for the selected event dates should be validated using an external or independent source. [PLACEHOLDER: Insert supporting rainfall validation here, such as PAGASA rainfall records, nearby weather-station observations, local rain-gauge logs, or raw sensor timestamp records confirming rainfall occurrence on the selected event dates.]
"""
    write_markdown(THESIS_ROOT / "rainfall_event_pattern_analysis.md", text)


def make_redeployed_plan(tables: dict[str, pd.DataFrame]) -> None:
    inventory = tables["redeployed_inventory"]
    if inventory.empty:
        inventory_text = "No files were found under `re-deployed_datasets` during this run. The plan below is therefore prepared before processing and should be executed once the May 27 and May 28 files are available."
    else:
        names = ", ".join(inventory["file_name"].tolist())
        inventory_text = f"The following redeployed dataset files were found but not processed in this planning step: {names}."
    text = f"""# Redeployed Dataset Analysis Plan

{inventory_text}

## 1. Required Preprocessing Steps

Import the May 27 and May 28 redeployed datasets, validate timestamps, standardize variable names, separate node-specific records, align observations to a common time interval, and preserve raw values alongside cleaned values. Missing values should be summarized before any imputation. Outliers should be flagged rather than silently removed.

## 2. Event Extraction Workflow

Use rainfall intensity and accumulated rainfall to identify candidate events. Define an event start when rainfall intensity becomes positive and an event end after a sustained dry interval. Because these datasets are intended to capture dry and light-rain behavior, lower rainfall thresholds should be tested separately from the thresholds used for moderate-to-heavy selected events.

## 3. Dry-Period Identification Strategy

Identify dry periods as intervals with zero or near-zero rainfall intensity and stable canal water level. Report baseline mean, median, standard deviation, and short-term fluctuation range. These dry-period statistics should become the reference for evaluating sensor stability and normal canal behavior.

## 4. Required Figures

Recommended figures are: dry-period canal water-level stability, light-rain event hydrograph, rainfall intensity and canal water-level overlay, and comparison of redeployed baseline against existing selected-event baselines.

## 5. Required Tables

Required tables are: redeployed data-quality summary, dry-period baseline statistics, light-rain event summary, event lag summary, and comparison table between redeployed light-rain events and existing selected events.

## 6. Chapter Placement

Data-quality and preprocessing details belong in methodology. Dry-period and light-rain findings belong in the results chapter. LSTM implications belong in Chapter 5 because the redeployed datasets can improve baseline learning and low-intensity event representation.

## 7. Expected Hydrologic Insights

The redeployed datasets should clarify normal canal behavior, background sensor fluctuation, and response under low-intensity rainfall. These insights will complement the current selected-event dataset, which is dominated by moderate-to-heavy rainfall.

## 8. LSTM Implications

Dry-period data can improve baseline prediction and reduce false rising-limb predictions. Light-rain events can help the LSTM distinguish small rainfall inputs from meaningful runoff-producing events. These datasets should therefore be used to improve class balance and baseline stability before final model training.
"""
    write_markdown(THESIS_ROOT / "redeployed_dataset_analysis_plan.md", text)


def make_section_mapping() -> None:
    rows = [
        ("Methodology: Data Preparation", "Keep existing title", "Strengthen text with raw dataset, timestamp, node-specific, and shared dataset definitions."),
        ("Methodology: Data Processing", "Keep existing title", "Clarify missing values, outlier flags, alignment, cleaning, and feature preparation."),
        ("Methodology: Rainfall Event Extraction and Selection Criteria", "Keep existing title", "Retain in methodology because it defines how events are selected."),
        ("Methodology: Event-Based Hydrologic Analysis", "Keep title if already present", "Keep procedures in methodology but move interpretation to results."),
        ("Results: Descriptive Analysis", "Keep existing title", "Use the preprocessed selected-event dataset and reduce redundant figures."),
        ("Results: Event Pattern Analysis", "Keep existing title", "Use merged selected-event collage as the central interpretive figure."),
        ("Results: Lag Analysis", "Keep existing title", "Discuss 2-3 h median first response and longer peak-response exceptions."),
        ("Chapter 5 LSTM Discussion", "Keep existing title", "Link lag, rainfall intensity, accumulated rainfall, and baseline behavior to forecasting design."),
    ]
    df = pd.DataFrame(rows, columns=["current_section", "title_recommendation", "content_revision"])
    write_table(THESIS_ROOT / "thesis_section_mapping", df)
    lines = ["# Thesis Section Mapping", ""]
    for section, title, revision in rows:
        lines.append(f"## {section}")
        lines.append(f"- Title recommendation: {title}")
        lines.append(f"- Content revision: {revision}")
        lines.append("")
    write_markdown(THESIS_ROOT / "thesis_section_mapping.md", "\n".join(lines))


def write_section_markdown(tables: dict[str, pd.DataFrame]) -> None:
    lag = tables["lag_stats"]
    node1 = lag[lag["node_id"] == "Node1"].iloc[0]
    node2 = lag[lag["node_id"] == "Node2"].iloc[0]
    sections = {
        "4.5.3.md": (
            "Selected-Event Descriptive Hydrologic Analysis",
            "This section should use the preprocessed selected-event dataset. It should report total rainfall, rainfall intensity, cumulative rainfall, and canal water level for each node using mean, median, standard deviation, minimum, and maximum. The purpose is to establish the hydrologic range of the selected rainfall-response events."
        ),
        "4.5.4.md": (
            "Variable Relationship Analysis",
            "This section should interpret the Pearson correlation matrix in hydrologic terms. Rainfall intensity and accumulated rainfall should be discussed as rainfall-forcing variables, while canal water level should be interpreted as the response variable. The emphasis should be on whether rainfall variables provide useful predictive information for canal response."
        ),
        "4.5.5.md": (
            "Time-Series and Shared Node Response",
            "This section should use the filtered individual-node and shared time-series figures. The shared visualization should show whether rainfall events appear at both nodes and whether canal water levels respond with comparable timing. Redundant time-series panels should be avoided."
        ),
        "4.5.6.md": (
            "Rainfall Event Pattern and Lag Analysis",
            f"This section should use the merged selected-event collage. Node 1 has a median peak-to-peak lag of {safe_number(node1['median_peak_lag_h'])} h, while Node 2 has a median peak-to-peak lag of {safe_number(node2['median_peak_lag_h'])} h. The first response remains close to the 2-3 h interpretation, but long peak lags should be discussed as repeated-pulse events or timing exceptions."
        ),
        "4.5.7.md": (
            "Hydrologic Interpretation Summary",
            "This section should synthesize rainfall development, canal response, node differences, and lag behavior. It should state that the observed response is generally delayed, node-specific in magnitude, and sensitive to repeated rainfall pulses."
        ),
        "5.8.md": (
            "LSTM Forecasting Implications",
            "This section should explain that the LSTM model must learn delayed rainfall-water-level response. Same-hour rainfall variables are insufficient; lagged rainfall intensity, accumulated rainfall, and previous canal water level should be included."
        ),
        "5.8.1.md": (
            "Forecast Horizon",
            "A 2-3 h horizon is defendable for first-response prediction. Longer horizons should be tested for peak water-level prediction because repeated rainfall pulses can extend the timing of maximum response."
        ),
        "5.8.2.md": (
            "Input Sequence Length",
            "The input sequence should cover antecedent rainfall intensity, accumulated rainfall, and canal water-level history over a window long enough to include the observed lag behavior."
        ),
        "5.8.3.md": (
            "Feature Engineering",
            "Recommended features include rainfall intensity, accumulated rainfall, lagged rainfall intensity, prior canal water level, water-level change, node identifier, and event-relative time."
        ),
        "5.8.4.md": (
            "Multi-Step Forecasting",
            "Multi-step forecasting is appropriate because water-level response can continue after rainfall intensity decreases. Direct multi-output forecasting should be considered to reduce recursive error accumulation."
        ),
        "5.8.5.md": (
            "Model Evaluation",
            "Evaluation should include event-window metrics, rainfall-intensity-class metrics, and dry-period baseline performance once redeployed datasets are incorporated."
        ),
        "5.8.6.md": (
            "Redeployed Dataset Integration",
            "The May 27 and May 28 redeployed datasets should be used to characterize dry-period and light-rain baselines before final model training. These data can improve baseline stability and low-intensity event representation."
        ),
    }
    for filename, (title, body) in sections.items():
        write_markdown(MARKDOWN_ROOT / filename, f"# {title}\n\n{body}")


def main() -> None:
    reset_output_dirs()
    data = load_data()
    filtered = filtered_hourly(data)
    tables = make_analysis_tables(data, filtered)
    figures = generate_figures(filtered, tables, data)
    make_methodology_review()
    make_eda_framework()
    make_hydrologic_findings(tables)
    make_event_pattern_analysis(tables)
    make_redeployed_plan(tables)
    make_section_mapping()
    write_section_markdown(tables)
    print("Thesis revision regenerated from guide.md")
    print(f"Root: {THESIS_ROOT}")
    print(f"Figures: {len(figures)}")
    print(f"Markdown files: {len(list(MARKDOWN_ROOT.glob('*.md')))}")


if __name__ == "__main__":
    main()
