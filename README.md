# MLForecasting

A probabilistic forecasting system for multi-product time series prediction using GRU-based neural networks with energy scoring. This repo implements a 3-layer GRU architecture generating 20-member probabilistic ensembles, trained on M5 sales data with a 60/20/20 temporal split for training/validation/testing.

## Data Requirements

This project requires two CSV files in the `src/` directory:

### 1. `sales_train_validation.csv`
Historical sales data matrix with product × day columns.

### 2. `calendar.csv`
Calendar metadata providing temporal context for feature engineering.

Both these datasets are available on kaggle at https://www.kaggle.com/datasets/kyakovlev/m5-aux-models

Used for: optional calendar covariate features during training and evaluation (see `--calendar-feature-set`).

---

## Setup

### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then verify setup:
```powershell
cd src
python train.py --help
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependency workflow: when package imports change, update `requirements.txt` first, then reinstall with `pip install -r requirements.txt`.

Then verify setup:
```bash
cd src
python train.py --help
```

---

## Training

Train the GRU-based probabilistic forecasting model on the first 60% of temporal data using energy or kernel scoring rules.

### Quick Start

```powershell
cd src
python train.py
```

This runs the default configuration from `config.yaml`:
- Loss function: energy
- Epochs: 40
- Batch size: 81,920 (This uses approximately 21GB of VRAM, scale the batch size to your hardware accordingly)
- Model: GRU hidden size 64, 8-dim noise, 20 ensemble members
- Calendar features: rich calendar (DOW + MOY + SNAP + events)

### Training Flags

**Model & Loss:**
- `--loss-name {ensemble_nll, energy, kernel, energy_kernel}` — Loss function (default: energy)
- `--epochs N` — Training passes (default: 40)
- `--window-size N` — sliding window in days (default: 28)

**Performance & Hardware:**
- `--batch-size N` — Samples per update (default: 81,920; reduce if CUDA OOM)
- `--learning-rate LR` — Optimizer step size (default: 0.001)
- `--progress-every N` — Print updates every N batches (default: 50)

**Data Scope:**
- `--mode {quick, full}` — **quick**: 100 products, 365 days (fast); **full**: all data (default: full)
- `--max-products N` — Cap products (default: 1,000)
- `--max-day-columns N` — Cap recent days (default: 365; must be ≥ window_size + 1)
- `--config PATH` — Override config file location (default: config.yaml)

### Example Commands

**Fast iteration (100 products, 20 epochs):**
```bash
python train.py --mode quick --epochs 20 --max-products 100
```

**Full training with custom loss:**
```bash
python train.py --mode full --loss-name kernel --epochs 50 --batch-size 4096
```

**Custom window size and learning rate:**
```bash
python train.py --window-size 14 --learning-rate 0.0005 --epochs 30
```

Outputs:
- Model checkpoint: `artifacts/model_checkpoint.pt`
- Model metadata: `artifacts/model_metadata.json`
- Loss plot: `artifacts/loss_vs_epoch.png`

---

## Validation

Evaluate the model on the **20%** of temporal data (validation split). This assesses model generalization without using test data.

### Quick Start

```bash
python fullpredict.py --eval-split val
```

### Validation Flags

- `--eval-split {val, test, all}` — Temporal split to evaluate on (default: test; use **val** for validation)
- `--mode {quick, full}` — Data scope: **quick** = 100 products, **full** = all (default: quick)
- `--batch-size N` — Products per inference batch (default: 2,048; reduce for slower GPUs)
- `--max-origins N` — Cap number of rolling origins (e.g., last 50 days only)
- `--horizon N` — Forecast horizon in days (default: 1; only 1-day-ahead supported)

### Example Commands

**Validate on 20% of data:**
```bash
python fullpredict.py --eval-split val --mode quick
```

**Full validation on all products:**
```bash
python fullpredict.py --eval-split val --mode full --batch-size 1024
```

**Validation with fewer rolling origins (faster):**
```bash
python fullpredict.py --eval-split val --max-origins 50 --batch-size 4096
```

Outputs:
- Per-product metrics: printed to console
- Summary statistics: MAE, RMSE, R² aggregated across all products

---

## Testing

Evaluate the model on the **final witheld 20%** of temporal data (test split). Tests are conducted as a rolling 1-day-ahead backtest with continuous model updates.

### Quick Start

```bash
python fullpredict.py --eval-split test
```

### Testing Flags

Same as Validation, but with `--eval-split test` (final 20%):
- `--eval-split test` — Use final 20% for testing (default)
- `--mode {quick, full}` — **quick** = 100 products, **full** = all (default: quick)
- `--batch-size N` — Products per batch (default: 2,048)
- `--max-origins N` — Cap rolling test origins

### Example Commands

**Test on final 20% (quick run):**
```bash
python fullpredict.py --eval-split test --mode quick
```

**Full test suite on all products:**
```bash
python fullpredict.py --eval-split test --mode full --batch-size 2048
```

**Test with custom window over final 30-day period:**
```bash
python fullpredict.py --eval-split test --max-origins 30 --window-size 14
```

Outputs:
- Per-origin metrics: rolling predictions for each forecast date
- Aggregated test metrics: MAE, RMSE, R² over the entire 20% test window

---

## Temporal Split Strategy

The data is split **temporally** (not randomly) to respect the time-series structure:

```
Day 1 ─────────────── Day N
[60% TRAINING]──────[VAL (20%)]──[TEST (20%)]
```

- **Training (0.6)**: Model learns from the first 60% of days
- **Validation (0.2)**: Middle 20% of days used for hyperparameter tuning and early stopping
- **Testing (0.2)**: Final 20% of days used for final model evaluation

All sliding window backtest origins are computed **after** the temporal split is applied, ensuring no data leakage.

## Baseline Comparisons

For benchmarking, use `compare_models.py` to compare against simple baselines:

```bash
python compare_models.py --eval-split test --mode full --models neural,seasonal7,seasonal28,ma7,ma28
```

Supported baselines:
- `seasonal7`: 7-day seasonal naïve
- `seasonal28`: 28-day seasonal naïve
- `ma7`: 7-day moving average
- `ma28`: 28-day moving average

Outputs:
- Leaderboard CSV: `artifacts/compare_models_metrics.csv`
- Per-origin metrics: `artifacts/compare_models_per_origin.csv`
- Comparison plots: `artifacts/compare_models_leaderboard.png`, `artifacts/compare_models_per_origin_rmse.png`

---

## Configuration

All hyperparameters are centralized in `config.yaml` and overridden via CLI flags:

```yaml
paths:
  data_path: src/sales_train_validation.csv
  calendar_path: src/calendar.csv
  model_file: artifacts/model_checkpoint.pt

training:
  loss_name: energy
  epochs: 40
  batch_size: 81920
  learning_rate: 0.001
  train_split: 0.6
  val_split: 0.2

model:
  gru_hidden_size: 64
  noise_size: 8
  number_generations_per_forward_call: 20
```

CLI overrides take precedence: `python train.py --epochs 50 --loss-name kernel` (overrides config.yaml)

---

## Quick Integration Example

```bash
# 1. Setup
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
pip install torch pandas numpy pyyaml

# 2. Train (15 min on GPU)
cd src
python train.py --mode quick --epochs 20

# 3. Validate
python fullpredict.py --eval-split val --mode quick

# 4. Test
python fullpredict.py --eval-split test --mode quick

# 5. Compare with baselines
python compare_models.py --eval-split test --mode quick
```

---

## Troubleshooting

**CUDA Out of Memory (OOM):**
- Reduce `--batch-size` (e.g., 4,096 or 2,048)
- Reduce `--max-products` or use `--mode quick`

**Slow training:**
- Use `--mode quick` for iteration
- Reduce `--epochs` for feedback before long runs