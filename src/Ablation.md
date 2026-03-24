# This document will explai how to run ablation studies

## 1) Setup

From the repo root:

```powershell
& .\.venv\Scripts\Activate.ps1
cd .\Forecasting\src
```

## 2) Pick the checkpoint

`ablation.py` supports three checkpoint formats:

1. Suffix token:

```powershell
--checkpoint _Energy_fullCalendar
```

This resolves to:

```text
Forecasting/artifacts/model_checkpoint_Energy_fullCalendar.pt
```

2. File name:

```powershell
--checkpoint model_checkpoint_Energy_fullCalendar.pt
```

3. Full/relative path:

```powershell
--checkpoint ..\artifacts\model_checkpoint_Energy_fullCalendar.pt
```

Metadata is auto-inferred when possible:

```text
model_checkpoint_SUFFIX.pt -> model_metadata_SUFFIX.json
```

You can override it explicitly with:

```powershell
--metadata ..\artifacts\model_metadata_Energy_fullCalendar.json
```

## 3) Ablation types and commands

### A) Feature-group ablation (`feature_zero`)

Zero out selected calendar feature groups at inference time.

Run one group:

```powershell
python .\ablation.py --checkpoint _Energy_fullCalendar feature_zero --zero-groups snap
```

Run all groups one-by-one:

```powershell
python .\ablation.py --checkpoint _Energy_fullCalendar feature_zero
```

Supported groups:

```text
event, snap, dow, month, weekend, day_idx
```

Output file:

```text
Forecasting/artifacts/ablation_feature_zero.csv
```

### B) Ensemble-size ablation (`ensemble_size`)

Change number of stochastic draws per forecast.

```powershell
python .\ablation.py --checkpoint _Energy_fullCalendar ensemble_size --num-generations 5 10 20 50
```

Output file:

```text
Forecasting/artifacts/ablation_ensemble_size.csv
```

### C) Freeze/fine-tune ablation (`freeze_finetune`)

Freeze one part of the model and fine-tune the rest.

Freeze GRU, tune FC:

```powershell
python .\ablation.py --checkpoint _Energy_fullCalendar freeze_finetune --freeze gru --finetune-epochs 3
```

Freeze FC, tune GRU:

```powershell
python .\ablation.py --checkpoint _Energy_fullCalendar freeze_finetune --freeze fc --finetune-epochs 3
```

### D) Loss-swap ablation (`loss_swap`)

Fine-tune the checkpoint with a different loss.

```powershell
python .\ablation.py --checkpoint _Energy_fullCalendar loss_swap --loss energy --finetune-epochs 2
```

Supported losses:

```text
ensemble_nll, energy, kernel, energy_kernel
```

## 4) Common global options

These can be used with any mode:

```text
--mode quick|full
--eval-split test|val|all
--batch-size <int>
--max-origins <int>
--save
--checkpoint <value>
--metadata <path>
```

Example with eval controls:

```powershell
python .\ablation.py --checkpoint _Energy_fullCalendar --mode quick --eval-split test --max-origins 100 feature_zero --zero-groups event
```

## 5) What this does not change

Checkpoint-only ablations do not change core architecture choices like:

```text
window size, GRU hidden size, GRU layer count, noise size
```

Those require training a new model configuration.