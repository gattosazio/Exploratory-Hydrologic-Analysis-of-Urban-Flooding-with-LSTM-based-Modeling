# ML Script Entry Points

Run these from the repository root.

```powershell
python scripts\ml\prepare_lstm_dataset.py
python scripts\ml\run_persistence_baseline.py
.\.venv-ml\Scripts\python.exe scripts\ml\train_lstm.py
.\.venv-ml\Scripts\python.exe scripts\ml\evaluate_selected_events.py --sequence-length 12 --horizon 1
.\.venv-ml\Scripts\python.exe scripts\ml\prepare_selected_event_change_dataset.py --sequence-length 6 --horizon 1
.\.venv-ml\Scripts\python.exe scripts\ml\train_selected_event_change_lstm.py --sequence-length 6 --horizon 1
.\.venv-ml\Scripts\python.exe scripts\ml\infer_redeployed_selected_event_lstm.py --sequence-length 6 --horizon 1
.\.venv-ml\Scripts\python.exe scripts\ml\generate_final_ml_outputs.py
```

Final selected-event model uses sequence length `12` and horizons `1`, `2`, and `3`.

Final model files:

- `output/ml/models/final_event_based_lstm_seq12_h1.pt`
- `output/ml/models/final_event_based_lstm_seq12_h2.pt`
- `output/ml/models/final_event_based_lstm_seq12_h3.pt`

The reusable code remains in `src/ml_pipeline/`. This folder only contains runnable scripts so the workflow is easier to follow.
