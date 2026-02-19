# Pricing Analysis

Regression pipeline for pricing data: clean → summarize → audit → model → report.

## Run the Web UI

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

- **/** – Upload CSV or PDF, run pipeline. Shows "Resume last run" link when a prior run exists.
- **/models** – Model status and optimization progress
- **/models/config** – Model config form, run Bayesian optimization (Optuna)
- **/results** – Run summary for the latest run

Upload a CSV first, then go to /models/config to run optimization. The best params are used when you run the pipeline. Click "Back to Run Summary" or "Resume last run" to revisit results without clearing the uploaded dataset.

## Run from CLI

```bash
python main.py path/to/data.csv [options]
```

### Target and transform options

- `--target <colname>` – Override auto-inferred target column
- `--target-transform {auto,none,log1p}` – Target transform: `auto` (heuristic), `none`, or `log1p`
- `--force-log1p` – Force log1p transform (alias for `--target-transform log1p`)
- `--no-log1p` – Disable log1p (alias for `--target-transform none`)

The auto heuristic uses log1p when skewness > 1.0 or p99/p50 > 20 (for nonnegative targets).

### Other options

- `--dry-run` – Run clean/audit only, skip training
- `--keep-negatives` – Keep rows with negative target
- `--no-report` – Disable feature report PDF
- `--summarize-columns` – Use LLM for column descriptions (requires OPENAI_API_KEY)

## Artifacts per run

Each pipeline run gets a unique `run_id` (timestamp + random suffix, e.g. `20250217_1432_a7f2`). Artifacts are written under:

```
artifacts/runs/{run_id}/
  ui_summary.csv
  ui_summary.md
  audit_report.pdf
  run_config.json
  metrics.json          # train/test metrics
  predictions.csv       # y_true, y_pred, residual, split
  residuals_report.pdf  # residual diagnostics
  permutation_importance.csv
  report.pdf            # feature plots (unless --no-report)
```

`artifacts/latest.txt` stores the most recent `run_id` so the UI can resume the last run.

## Stop optimization safely

While Optuna is running, click "Stop Optimization" on the /models page. The stop is processed at the next trial boundary (at most one trial delay). Status shows "Stopping..." until the run finishes.

## Smoke test

```bash
chmod +x scripts/smoke_test.sh
./scripts/smoke_test.sh
```

Runs dry-run and full pipeline on minimal data and verifies expected artifacts exist.

## PDF support

For PDF uploads, install a table extraction library:

```bash
pip install pdfplumber   # recommended
# or
pip install 'camelot-py[cv]'
```

The app works with CSV even if PDF libraries are not installed.
