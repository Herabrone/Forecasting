
## 1) Setup

From the repo root:

```powershell
& .\src\.venv\Scripts\Activate.ps1
```

Optional dependency fix (only needed if you see `No module named 'yaml'`):

```powershell
python -m pip install pyyaml
```

Sanity check:

```powershell
python .\src\ablation.py --help
```

Or, if you prefer to run from `src`:

```powershell
cd .\src
python .\ablation.py --help
```

## 2) Pick the checkpoint

`ablation.py` supports three checkpoint formats:

Normal repo naming (recommended):

```text
artifacts/model_checkpoint.pt
artifacts/model_metadata.json
```

1. Suffix token:

```powershell
--checkpoint _Energy_fullCalendar
```

This resolves to:

```text
artifacts/model_checkpoint_Energy_fullCalendar.pt
```

2. File name:

```powershell
--checkpoint model_checkpoint.pt
```

3. Full/relative path:

```powershell
--checkpoint ..\artifacts\model_checkpoint.pt  # when running from src
--checkpoint .\artifacts\model_checkpoint.pt   # when running from repo root
```

Metadata is auto-inferred when possible:

```text
model_checkpoint_SUFFIX.pt -> model_metadata_SUFFIX.json
```

You can override it explicitly with:

```powershell
--metadata ..\artifacts\model_metadata.json
```

## 3) Ablation types and commands

### A) Feature-group ablation (`feature_zero`)

Zero out selected calendar feature groups at inference time.

Run one group:

```powershell
python .\src\ablation.py --checkpoint model_checkpoint.pt feature_zero --zero-groups snap
```

Run all groups one-by-one:

```powershell
python .\src\ablation.py --checkpoint model_checkpoint.pt feature_zero
```

Supported groups:

```text
event, snap, dow, month, weekend, day_idx
```

Output file:

```text
artifacts/ablation_feature_zero.csv
```

### B) Ensemble-size ablation (`ensemble_size`)

Change number of ensemble draws per forecast.

```powershell
python .\src\ablation.py --checkpoint model_checkpoint.pt ensemble_size --num-generations 5 10 20 50
```

Output file:

```text
artifacts/ablation_ensemble_size.csv
```

### C) Freeze/fine-tune ablation (`freeze_finetune`)

Freeze one part of the model and fine-tune the rest.

Freeze GRU, tune FC:

```powershell
python .\src\ablation.py --checkpoint model_checkpoint.pt freeze_finetune --freeze gru --finetune-epochs 3
```

Freeze FC, tune GRU:

```powershell
python .\src\ablation.py --checkpoint model_checkpoint.pt freeze_finetune --freeze fc --finetune-epochs 3
```

If you also pass `--save`, the fine-tuned checkpoint is written to:

```text
artifacts/ablation_freeze_gru.pt
artifacts/ablation_freeze_fc.pt
```

### D) Loss-swap ablation (`loss_swap`)

Fine-tune the checkpoint with a different loss.

```powershell
python .\src\ablation.py --checkpoint model_checkpoint.pt loss_swap --loss energy --finetune-epochs 2
```

Supported losses:

```text
ensemble_nll, energy, kernel, energy_kernel
```

If you also pass `--save`, the fine-tuned checkpoint is written to:

```text
artifacts/ablation_loss_<loss>.pt
```

## 4) Common global options

These can be used with any mode:

```text
--mode quick|full
--eval-split test|val|all
--batch-size <int>
--max-origins <int>
--config <path>
--save
--checkpoint <value>
--metadata <path>
```

Example with eval controls:

```powershell
python .\src\ablation.py --checkpoint model_checkpoint.pt --mode quick --eval-split test --max-origins 100 feature_zero --zero-groups event
```

Example from `src` directory:

```powershell
python .\ablation.py --checkpoint model_checkpoint.pt --mode quick --eval-split test --max-origins 100 feature_zero --zero-groups event
```
