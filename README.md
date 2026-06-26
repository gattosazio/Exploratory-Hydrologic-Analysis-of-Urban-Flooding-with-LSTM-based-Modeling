# Hydrologic Analysis & LSTM-Based Flood Forecasting

This repository contains supporting datasets, figures, data analysis pipelines, and machine-learning code for the thesis study:
**"Development of an IoT-Based Near Real-Time Precipitation Monitoring System with Machine Learning for Hydrologic Analysis of Urban Flooding in Del Carmen, Iligan City"**

---

## 1. Project Overview

Urban flooding in Del Carmen, Iligan City represents a significant hazard that requires effective monitoring and predictive early-warning tools. Due to local infrastructure constraints, physical mitigation measures cannot be immediately deployed. This study establishes a real-time IoT monitoring system coupled with a machine-learning model to forecast short-term water-level changes, providing civil defense teams with critical lead time for emergency response.

---

## 2. Directory Structure

```text
.
├── datasets/
│   ├── raw/                             # Original raw datasets (JSON & CSV formats)
│   └── preprocessed_selected_events/    # Cleaned, baseline-aligned selected storm events
├── figure_data/                         # Cleaned CSV and JSON data tables mapped to figures
├── figures/                             # Categorized thesis figures (collages and close-ups)
│   ├── descriptive-statistics/          # Node-level statistics bar charts
│   ├── lag_analysis/                    # Peak-to-peak lag boxplots
│   ├── merged_selected_events/          # Merged event rainfall-response collages
│   ├── pearson_correlation/             # Correlation heatmaps
│   └── time-series/                     # Full monitoring overview plots
├── machine_learning/                    # Machine learning pipeline outputs
│   ├── datasets/                        # Training/validation/test sequence arrays
│   ├── models/                          # Trained PyTorch LSTM models (.pt)
│   ├── reports/                         # LSTM performance reports (RMSE, MAE, R²)
│   └── figures/                         # Observed vs. Predicted prediction plots
├── reports/                             # Tabular summaries, evaluations, and inventory reports
└── scripts/                             # Runnable preprocessing, plotting, and training scripts
```

---

## 3. Data Preprocessing Pipeline

The sensor data collected by the IoT system contains high-frequency noise typical of ultrasonic distance sensors in open-canal environments, along with dry-period background fluctuations. A strict, reproducible preprocessing pipeline is implemented to generate clean, physically consistent datasets for analysis and model training:

1. **Timestamp Synchronization & Deduplication**: Timestamps are parsed, synchronized to 1-hour intervals, and sorted chronologically, keeping the latest valid reading in case of duplication.
2. **Outlier & Anomaly Correction**: High-frequency measurement spikes (e.g., sudden jumps to maximum depth unsupported by antecedent rainfall) are flagged and corrected using nearest-neighbor interpolation.
3. **Baseline Offset Alignment**: Water-level measurements are aligned to a standardized dry-period baseline (1.5 cm for Node 1, 10.5 cm for Node 2) to eliminate static measurement discrepancies.
4. **Noise Reduction**: Signal-cleaning filters are applied to isolate the storm runoff response (hydrograph) from background fluctuations, ensuring the LSTM learns true hydrologic dynamics.

---

## 4. LSTM Forecasting Framework

A parsimonious **Long Short-Term Memory (LSTM)** neural network is designed to forecast future water-level change ($\Delta h$) rather than absolute values, forcing the model to learn directional rainfall-response limb dynamics.

* **Architecture**: Single recurrent LSTM layer with **32 hidden units** and a dropout layer, followed by a fully connected output layer. The model contains **5,409 trainable parameters**, making it lightweight enough to run on resource-constrained edge devices and robust against overfitting on small datasets.
* **Input Sequence**: 12 hours of historical antecedent sequence window (rainfall intensity, total rainfall, cumulative rainfall, water level, and their rates of change).
* **Operational Lead Times**: 1, 2, and 3-hour forecast horizons to cover the urban flood hydrograph lifecycle (early rising limb detection, trend escalation, and evacuation planning).

### Forecasting Performance Results
Percentage improvements are computed relative to the zero-change persistence baseline:

| Horizon | LSTM RMSE | Persistence RMSE | RMSE Improvement | Operational Purpose |
| :---: | :---: | :---: | :---: | :---: |
| **1 Hour** | 2.57 cm | 4.98 cm | **48.3%** | Immediate flash flood detection |
| **2 Hours** | 3.01 cm | 9.34 cm | **67.7%** | Trend escalation tracking |
| **3 Hours** | 3.62 cm | 13.89 cm | **74.0%** | Evacuation & civil defense lead time |

---

## 5. Usage & Execution Guide

All scripts are located in the `scripts` folder and configured to run within the self-contained repository environment.

### A. Run Preprocessing Pipeline
Rebuild the clean hourly datasets, extract rain events, and output event summaries to the `reports` directory:
```bash
python scripts/src/run_preprocessing_pipeline.py
```

### B. Run Descriptive & Thesis Output Generation
Compute descriptive statistics, Pearson correlation heatmaps, lag distribution boxplots, and event classification counts:
```bash
python scripts/src/run_descriptive_analysis.py
python scripts/src/generate_thesis_outputs.py
```

### C. Preprocess Redeployed Dataset
Process dry-period and light-rain redeployed validation datasets:
```bash
python scripts/scripts/preprocess_redeployed_dataset.py
```

### D. Generate Selected Event Figures
Generate the storm hydrograph collages:
```bash
python scripts/src/generate_normalized_selected_event_figures.py
```

### E. Prepare LSTM Dataset & Train Model
Prepare the supervised sequence arrays and train the PyTorch LSTM model:
```bash
# Prepare dataset
python scripts/src/ml_pipeline/prepare_lstm_dataset.py --sequence-length 12 --horizon 3

# Train model
python scripts/src/ml_pipeline/train_lstm.py --sequence-length 12 --horizon 3
```
