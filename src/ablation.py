"""Ablation studies using an existing model checkpoint.

Loads a trained checkpoint and evaluates modified versions of it so you
don't have to retrain from scratch every time.  Supported ablation modes:

  feature_zero   – Zero-out groups of calendar features at inference time
                   to measure each group's contribution.  No retraining.

  freeze_finetune – Freeze either the GRU or FC layers and fine-tune the
                    rest for a few epochs.

  loss_swap       – Fine-tune briefly with a different loss function.

  ensemble_size   – Re-evaluate using a different number of ensemble
                    draws (no retraining needed).

Examples:
  # See how much removing event features hurts (no retraining):
  python ablation.py feature_zero --zero-groups event

  # Restrict GRU layers while adapting forward fully-connected networks mapping.
  python ablation.py freeze_finetune --freeze gru --finetune-epochs 3

  # Fine-tune with kernel loss for 2 epochs:
  python ablation.py loss_swap --loss kernel --finetune-epochs 2

  # Project metrics under expanded ensemble simulation variations.
  python ablation.py ensemble_size --num-generations 50
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

import datapreprocessing as dp
import fullpredict as fp
from config import cli_or_config, load_config, resolve_path
from NN import ConditionalGenerativeModel, createGenerativeGRUNN
from datapreprocessing import NORMALIZATION_EPSILON, load_calendar_features
from train import TimeSeriesWindowDataset, compute_loss
from fullpredict import (
    _split_origin_range,
    compute_metrics,
    load_series_matrix,
    model_predict_mean,
    normalize_calendar_context,
    normalize_context_batch,
)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
METADATA_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
WINDOW_SIZE = dp.WINDOW_SIZE
DATA_PATH = Path(__file__).resolve().parent / 'sales_train_validation.csv'
CALENDAR_PATH = Path(__file__).resolve().parent / 'calendar.csv'
METADATA_FILE = Path(__file__).resolve().parent.parent / 'artifacts' / 'model_metadata.json'
MODEL_FILE = Path(__file__).resolve().parent.parent / 'artifacts' / 'model_checkpoint.pt'
CALENDAR_FEATURE_SET = 'rich'
MAX_DAY_COLUMNS = 365
MAX_PRODUCTS = 1000
TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.2
TRAIN_BATCH_SIZE = 81920
NUM_WORKERS = 8
PREFETCH_FACTOR = 2


def _create_data_loader(dataset, batch_size: int, shuffle: bool):
    """Instantiates a highly parallelized native PyTorch data loader for the dataset.

    Args:
        dataset: The collection of time series blocks to iterate over.
        batch_size: Fixed number of elements to process concurrently.
        shuffle: Toggles randomized sampling of elements.

    Returns:
        The initialized DataLoader object.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR if batch_size > 0 and NUM_WORKERS > 0 else None,
    )


def apply_runtime_config(config_path: str | None):
    """Reconciles internal parameter globals using cli flags or YAML configurations.

    Args:
        config_path: Pathline path directing to project configurations.

    Returns:
        None. Modifies global limits internally before program execution begins.
    """
    global WINDOW_SIZE, DATA_PATH, CALENDAR_PATH, METADATA_FILE, MODEL_FILE
    global CALENDAR_FEATURE_SET, MAX_DAY_COLUMNS, MAX_PRODUCTS, TRAIN_SPLIT, VAL_SPLIT
    global TRAIN_BATCH_SIZE, NUM_WORKERS, PREFETCH_FACTOR

    config = load_config(config_path)
    WINDOW_SIZE = int(cli_or_config(None, config, 'data', 'window_size'))
    dp.WINDOW_SIZE = WINDOW_SIZE
    fp.WINDOW_SIZE = WINDOW_SIZE

    DATA_PATH = resolve_path(config, 'data_path', DATA_PATH)
    CALENDAR_PATH = resolve_path(config, 'calendar_path', CALENDAR_PATH)
    METADATA_FILE = resolve_path(config, 'metadata_file', METADATA_FILE)
    MODEL_FILE = resolve_path(config, 'model_file', MODEL_FILE)

    CALENDAR_FEATURE_SET = str(cli_or_config(None, config, 'data', 'calendar_feature_set'))
    MAX_DAY_COLUMNS = int(cli_or_config(None, config, 'data', 'max_day_columns'))
    MAX_PRODUCTS = int(cli_or_config(None, config, 'data', 'max_products'))
    TRAIN_SPLIT = float(cli_or_config(None, config, 'training', 'train_split'))
    VAL_SPLIT = float(cli_or_config(None, config, 'training', 'val_split'))
    TRAIN_BATCH_SIZE = int(cli_or_config(None, config, 'training', 'batch_size'))
    NUM_WORKERS = int(cli_or_config(None, config, 'training', 'num_workers'))
    PREFETCH_FACTOR = int(cli_or_config(None, config, 'training', 'prefetch_factor'))

    fp.DATA_PATH = DATA_PATH
    fp.CALENDAR_PATH = CALENDAR_PATH
    fp.METADATA_FILE = METADATA_FILE
    fp.MODEL_FILE = MODEL_FILE
    fp.CALENDAR_FEATURE_SET = CALENDAR_FEATURE_SET
    fp.MAX_DAY_COLUMNS = MAX_DAY_COLUMNS
    fp.MAX_PRODUCTS = MAX_PRODUCTS
    fp.TRAIN_SPLIT = TRAIN_SPLIT
    fp.VAL_SPLIT = VAL_SPLIT


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_checkpoint_path(checkpoint_arg: str | None) -> Path:
    """Resolve checkpoint path from CLI input.

    Supported forms:
      - Full/relative path: ..\\artifacts\\model_checkpoint_x.pt
      - File name only: model_checkpoint_x.pt (looked up in cwd first, then artifacts)
      - Suffix token: _Energy_fullCalendar -> artifacts/model_checkpoint_Energy_fullCalendar.pt
    """
    if checkpoint_arg is None:
        return MODEL_FILE

    text = checkpoint_arg.strip()
    if not text:
        return MODEL_FILE

    if text.startswith("_"):
        candidate = ARTIFACTS_DIR / f"model_checkpoint{text}.pt"
        if candidate.exists():
            return candidate

    raw_path = Path(text)
    if raw_path.is_absolute() and raw_path.exists():
        return raw_path
    if raw_path.exists():
        return raw_path.resolve()

    artifacts_candidate = ARTIFACTS_DIR / text
    if artifacts_candidate.exists():
        return artifacts_candidate

    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_arg}. "
        f"Pass a valid .pt path, file name, or suffix like '_Energy_fullCalendar'."
    )


def _resolve_metadata_path(checkpoint_path: Path, metadata_arg: str | None) -> Path:
    """Resolve metadata path; infer matching metadata file from checkpoint when possible."""
    if metadata_arg:
        raw = Path(metadata_arg)
        if raw.is_absolute() and raw.exists():
            return raw
        if raw.exists():
            return raw.resolve()
        artifacts_candidate = ARTIFACTS_DIR / metadata_arg
        if artifacts_candidate.exists():
            return artifacts_candidate
        raise FileNotFoundError(f"Metadata file not found: {metadata_arg}")

    # Infer metadata file by matching the checkpoint's suffix.
    stem = checkpoint_path.stem
    if stem.startswith("model_checkpoint"):
        suffix = stem[len("model_checkpoint"):]
        inferred = ARTIFACTS_DIR / f"model_metadata{suffix}.json"
        if inferred.exists():
            return inferred

    if METADATA_FILE.exists():
        return METADATA_FILE

    raise FileNotFoundError(
        "Could not resolve metadata file. Pass --metadata explicitly."
    )


def _load_metadata(metadata_path: Path) -> dict:
    with metadata_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_checkpoint(checkpoint_path: Path, device: str = "cpu") -> dict:
    return torch.load(checkpoint_path, map_location=device)


def _load_calendar_matrix_for_checkpoint(day_cols: List[str], data_size: int, metadata_path: Path) -> torch.Tensor | None:
    """Load calendar matrix aligned to this checkpoint's expected metadata features."""
    if data_size == 1:
        return None

    if not CALENDAR_PATH.exists():
        raise FileNotFoundError(f"Calendar file not found: {CALENDAR_PATH}")

    calendar_values, calendar_feature_names = load_calendar_features(
        CALENDAR_PATH,
        day_cols,
        feature_set=CALENDAR_FEATURE_SET,
    )

    metadata = _load_metadata(metadata_path)
    expected_feature_names = metadata.get("calendar_feature_names", [])
    if isinstance(expected_feature_names, list) and len(expected_feature_names) > 0:
        feature_to_idx = {name: idx for idx, name in enumerate(calendar_feature_names)}
        aligned_columns = []
        for feature_name in expected_feature_names:
            feature_idx = feature_to_idx.get(feature_name)
            if feature_idx is None:
                aligned_columns.append(torch.zeros(len(day_cols), dtype=torch.float32).numpy())
            else:
                aligned_columns.append(calendar_values[:, feature_idx])

        calendar_values = torch.stack(
            [torch.tensor(column, dtype=torch.float32) for column in aligned_columns],
            dim=1,
        ).numpy()
        calendar_feature_names = expected_feature_names

    expected_calendar_features = data_size - 1
    if calendar_values.shape[1] != expected_calendar_features:
        raise ValueError(
            f"Checkpoint expects {expected_calendar_features} calendar features (data_size={data_size}), "
            f"but built {calendar_values.shape[1]} from calendar.csv using {metadata_path.name}."
        )

    return torch.tensor(calendar_values, dtype=torch.float32)


def _rebuild_model(checkpoint: dict, device: str) -> ConditionalGenerativeModel:
    """Build a fresh model from checkpoint hyper-params and load its weights."""
    net = createGenerativeGRUNN(
        data_size=checkpoint.get("data_size", 1),
        gru_hidden_size=checkpoint["gru_hidden_size"],
        noise_size=checkpoint["noise_size"],
        output_size=checkpoint["output_size"],
        hidden_sizes=checkpoint.get("fc_hidden_sizes"),
    )()
    model = ConditionalGenerativeModel(
        net=net,
        size_auxiliary_variable=checkpoint["noise_size"],
        number_generations_per_forward_call=checkpoint["number_generations_per_forward_call"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _rolling_eval(
    model: ConditionalGenerativeModel,
    series: torch.Tensor,
    calendar_matrix: torch.Tensor | None,
    device: str,
    batch_size: int,
    eval_split: str = "test",
    max_origins: int | None = None,
    feature_mask: torch.Tensor | None = None,
) -> Dict[str, float]:
    """Run the same rolling backtest as fullpredict but return only summary metrics.

    feature_mask – optional 1-D boolean tensor of shape (num_calendar_features,).
                   True = zero that feature column before feeding to the model.
    """
    model.eval()
    num_products, num_days = series.shape
    first_origin, last_origin = _split_origin_range(num_days, horizon=1, eval_split=eval_split)
    origins = list(range(first_origin, last_origin + 1))
    if not origins:
        raise ValueError(f"No rolling origins for eval_split='{eval_split}'.")
    if max_origins is not None and max_origins > 0:
        origins = origins[-max_origins:]

    all_true, all_pred = [], []

    for origin in origins:
        true_values = series[:, origin]

        normalized_calendar_window = None
        if calendar_matrix is not None:
            raw_cal = calendar_matrix[origin - WINDOW_SIZE:origin, :]
            if feature_mask is not None:
                raw_cal = raw_cal.clone()
                raw_cal[:, feature_mask] = 0.0
            normalized_calendar_window = normalize_calendar_context(raw_cal)

        pred_chunks = []
        for start in range(0, num_products, batch_size):
            end = min(start + batch_size, num_products)
            raw_ctx = series[start:end, origin - WINDOW_SIZE:origin]
            norm_ctx, ctx_mean, ctx_std = normalize_context_batch(raw_ctx)

            if normalized_calendar_window is not None:
                cal_batch = normalized_calendar_window.unsqueeze(0).repeat(end - start, 1, 1)
                model_ctx = torch.cat([norm_ctx.unsqueeze(-1), cal_batch], dim=2)
            else:
                model_ctx = norm_ctx.unsqueeze(-1)

            pred_mean = model_predict_mean(model, model_ctx, device)
            pred_chunks.append(pred_mean * ctx_std + ctx_mean)

        all_true.append(true_values)
        all_pred.append(torch.cat(pred_chunks, dim=0))

    return compute_metrics(torch.cat(all_true), torch.cat(all_pred))


def _print_metrics(label: str, metrics: Dict[str, float]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  MAE  : {metrics['mae']:.4f}")
    print(f"  RMSE : {metrics['rmse']:.4f}")
    print(f"  R2   : {metrics['r2']:.4f}")


# ---------------------------------------------------------------------------
# feature_zero – mask out calendar feature groups and re-evaluate
# ---------------------------------------------------------------------------

# Named groups for convenient ablation.  Keys map to substrings of
# calendar_feature_names in the metadata.
FEATURE_GROUPS = {
    "event":   "event_",
    "snap":    "snap_",
    "dow":     "dow_",
    "month":   "month_",
    "weekend": "is_weekend",
    "day_idx": "day_index",
}


def _resolve_feature_mask(group_name: str, calendar_feature_names: List[str]) -> torch.Tensor:
    """Return a boolean tensor – True for features to zero out."""
    prefix = FEATURE_GROUPS.get(group_name)
    if prefix is None:
        raise ValueError(
            f"Unknown feature group '{group_name}'. "
            f"Available: {', '.join(FEATURE_GROUPS)}"
        )
    mask = torch.tensor(
        [prefix in name for name in calendar_feature_names],
        dtype=torch.bool,
    )
    if not mask.any():
        print(f"Warning: no calendar features matched group '{group_name}'")
    return mask


def run_feature_zero(args: argparse.Namespace) -> None:
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    metadata_path = _resolve_metadata_path(checkpoint_path, args.metadata)
    metadata = _load_metadata(metadata_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = _load_checkpoint(checkpoint_path, device)
    model = _rebuild_model(checkpoint, device)
    model.eval()
    data_size = model.net.gru.input_size

    product_ids, day_cols, series = load_series_matrix(mode=args.mode)
    calendar_matrix = _load_calendar_matrix_for_checkpoint(day_cols=day_cols, data_size=data_size, metadata_path=metadata_path)
    calendar_names = metadata.get("calendar_feature_names", [])

    if calendar_matrix is None:
        print("Model was trained without calendar features – nothing to ablate.")
        return

    # Establish the unmasked baseline metric prior to ablation.
    baseline = _rolling_eval(
        model, series, calendar_matrix, device,
        batch_size=args.batch_size, eval_split=args.eval_split,
        max_origins=args.max_origins,
    )
    _print_metrics("Baseline (all features)", baseline)

    groups = args.zero_groups if args.zero_groups else list(FEATURE_GROUPS)
    results = [{"group": "baseline", **baseline}]

    for group in groups:
        mask = _resolve_feature_mask(group, calendar_names)
        n_masked = int(mask.sum())
        metrics = _rolling_eval(
            model, series, calendar_matrix, device,
            batch_size=args.batch_size, eval_split=args.eval_split,
            max_origins=args.max_origins, feature_mask=mask,
        )
        _print_metrics(f"Zeroed '{group}' ({n_masked} features)", metrics)
        results.append({"group": group, "features_masked": n_masked, **metrics})

    out_path = ARTIFACTS_DIR / "ablation_feature_zero.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


# ---------------------------------------------------------------------------
# freeze_finetune – freeze GRU or FC, fine-tune the rest
# ---------------------------------------------------------------------------

def _freeze_params(module: torch.nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def _unfreeze_params(module: torch.nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


def _get_finetune_loaders(args: argparse.Namespace, data_size: int, metadata_path: Path | None = None):
    """Build train + val DataLoaders for fine-tuning (reuses train.py logic)."""
    import gc
    import numpy as np

    header_columns = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()
    all_day_cols = sorted(
        [c for c in header_columns if c.startswith("d_")],
        key=lambda d: int(d.split("_", 1)[1]),
    )
    if args.mode == "quick":
        days_to_keep = max(WINDOW_SIZE + 1, MAX_DAY_COLUMNS)
        selected_day_cols = all_day_cols[-days_to_keep:]
    else:
        selected_day_cols = all_day_cols

    meta_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    usecols = meta_cols + selected_day_cols
    dataset = pd.read_csv(DATA_PATH, usecols=usecols)
    if args.mode == "quick":
        dataset = dataset.head(MAX_PRODUCTS)

    sales_matrix = dataset[selected_day_cols].to_numpy(dtype=np.float32, copy=True)

    metadata = _load_metadata(metadata_path) if metadata_path is not None else {}

    calendar_features = None
    if data_size > 1:
        feature_set = CALENDAR_FEATURE_SET
        if metadata_path is not None:
            feature_set = metadata.get("calendar_feature_set", CALENDAR_FEATURE_SET)

        calendar_features_arr, calendar_feature_names = load_calendar_features(
            CALENDAR_PATH, selected_day_cols, feature_set=feature_set,
        )

        expected_feature_names = metadata.get("calendar_feature_names", [])
        if isinstance(expected_feature_names, list) and len(expected_feature_names) > 0:
            feature_to_idx = {name: idx for idx, name in enumerate(calendar_feature_names)}
            aligned_columns = []
            for feature_name in expected_feature_names:
                feature_idx = feature_to_idx.get(feature_name)
                if feature_idx is None:
                    aligned_columns.append(np.zeros(len(selected_day_cols), dtype=np.float32))
                else:
                    aligned_columns.append(calendar_features_arr[:, feature_idx])

            calendar_features_arr = np.column_stack(aligned_columns).astype(np.float32)
            calendar_feature_names = expected_feature_names

        expected_calendar_features = data_size - 1
        if calendar_features_arr.shape[1] != expected_calendar_features:
            raise ValueError(
                f"Feature mismatch for fine-tune loader: checkpoint expects "
                f"{expected_calendar_features} calendar features (data_size={data_size}), "
                f"but loader produced {calendar_features_arr.shape[1]} using feature_set='{feature_set}'."
            )

        calendar_features = calendar_features_arr

    del dataset
    gc.collect()

    base = TimeSeriesWindowDataset(sales_matrix, WINDOW_SIZE, calendar_features)
    total_w = base.num_windows
    train_end = max(1, min(int(math.floor(total_w * TRAIN_SPLIT)), total_w - 2))
    val_end = max(train_end + 1, min(int(math.floor(total_w * (TRAIN_SPLIT + VAL_SPLIT))), total_w - 1))

    train_ds = TimeSeriesWindowDataset(sales_matrix, WINDOW_SIZE, calendar_features, 0, train_end)
    val_ds = TimeSeriesWindowDataset(sales_matrix, WINDOW_SIZE, calendar_features, train_end, val_end)

    train_loader = _create_data_loader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = _create_data_loader(val_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def _finetune(
    model: ConditionalGenerativeModel,
    train_loader,
    val_loader,
    epochs: int,
    loss_name: str,
    device: str,
    lr: float = 5e-4,
) -> ConditionalGenerativeModel:
    """Updates active parameters through back-propagation and partial training length.

    Args:
        model: Torch sequence estimation structure containing active gradients.
        train_loader: Sequence elements generator across subset.
        val_loader: Holdout chunk element loader.
        epochs: Repetition times around whole partial dataset.
        loss_name: Cost definition string.
        device: Current computing medium label.
        lr: Scaled descent rate parameter. Default is 5e-4.

    Returns:
        Conditioned network object loaded against minimized error point.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        print("Warning: no trainable parameters – skipping fine-tuning.")
        return model

    optimizer = torch.optim.Adam(trainable, lr=lr)
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0
        for ctx, tgt in train_loader:
            ctx, tgt = ctx.to(device), tgt.to(device)
            optimizer.zero_grad()
            preds = model(ctx)
            loss = compute_loss(preds, tgt, loss_name)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * ctx.size(0)
            n_samples += ctx.size(0)

        avg_train = running_loss / max(n_samples, 1)

        # Measure generalization error across the held-out validation segment.
        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for ctx, tgt in val_loader:
                ctx, tgt = ctx.to(device), tgt.to(device)
                preds = model(ctx)
                loss = compute_loss(preds, tgt, loss_name)
                val_loss += loss.item() * ctx.size(0)
                val_n += ctx.size(0)
        avg_val = val_loss / max(val_n, 1)

        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Fine-tune epoch {epoch}/{epochs}  train={avg_train:.6f}  val={avg_val:.6f}  best_val={best_val:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_freeze_finetune(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    metadata_path = _resolve_metadata_path(checkpoint_path, args.metadata)
    checkpoint = _load_checkpoint(checkpoint_path, device)
    model = _rebuild_model(checkpoint, device)
    data_size = checkpoint.get("data_size", 1)
    loss_name = checkpoint.get("loss_name", "ensemble_nll")

    # Apply targeted parameter immobilization based on runtime configuration.
    if args.freeze == "gru":
        _freeze_params(model.net.gru)
        label = "GRU frozen, FC fine-tuned"
    elif args.freeze == "fc":
        _freeze_params(model.net.fc_nn)
        label = "FC frozen, GRU fine-tuned"
    else:
        raise ValueError(f"--freeze must be 'gru' or 'fc', got '{args.freeze}'")

    print(f"Ablation: {label}")
    train_loader, val_loader = _get_finetune_loaders(args, data_size, metadata_path=metadata_path)
    model = _finetune(model, train_loader, val_loader, args.finetune_epochs, loss_name, device, lr=args.lr)

    # Compare model performance against the historical rolling window sequence.
    product_ids, day_cols, series = load_series_matrix(mode=args.mode)
    calendar_matrix = _load_calendar_matrix_for_checkpoint(day_cols=day_cols, data_size=data_size, metadata_path=metadata_path)
    metrics = _rolling_eval(
        model, series, calendar_matrix, device,
        batch_size=args.batch_size, eval_split=args.eval_split,
        max_origins=args.max_origins,
    )
    _print_metrics(label, metrics)

    if args.save:
        out_pt = ARTIFACTS_DIR / f"ablation_freeze_{args.freeze}.pt"
        torch.save({**checkpoint, "model_state_dict": model.state_dict()}, out_pt)
        print(f"Saved fine-tuned checkpoint: {out_pt}")


# ---------------------------------------------------------------------------
# loss_swap – fine-tune briefly with a different loss function
# ---------------------------------------------------------------------------

def run_loss_swap(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    metadata_path = _resolve_metadata_path(checkpoint_path, args.metadata)
    checkpoint = _load_checkpoint(checkpoint_path, device)
    model = _rebuild_model(checkpoint, device)
    data_size = checkpoint.get("data_size", 1)
    new_loss = args.loss

    print(f"Ablation: swapping loss from '{checkpoint.get('loss_name')}' to '{new_loss}'")
    train_loader, val_loader = _get_finetune_loaders(args, data_size)
    model = _finetune(model, train_loader, val_loader, args.finetune_epochs, new_loss, device, lr=args.lr)

    product_ids, day_cols, series = load_series_matrix(mode=args.mode)
    calendar_matrix = _load_calendar_matrix_for_checkpoint(day_cols=day_cols, data_size=data_size, metadata_path=metadata_path)
    metrics = _rolling_eval(
        model, series, calendar_matrix, device,
        batch_size=args.batch_size, eval_split=args.eval_split,
        max_origins=args.max_origins,
    )
    _print_metrics(f"Fine-tuned with '{new_loss}' loss", metrics)

    if args.save:
        out_pt = ARTIFACTS_DIR / f"ablation_loss_{new_loss}.pt"
        torch.save({**checkpoint, "model_state_dict": model.state_dict(), "loss_name": new_loss}, out_pt)
        print(f"Saved checkpoint: {out_pt}")


# ---------------------------------------------------------------------------
# ensemble_size – re-evaluate with different number of ensemble draws
# ---------------------------------------------------------------------------

def run_ensemble_size(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    metadata_path = _resolve_metadata_path(checkpoint_path, args.metadata)
    checkpoint = _load_checkpoint(checkpoint_path, device)
    model = _rebuild_model(checkpoint, device)
    model.eval()
    data_size = model.net.gru.input_size

    product_ids, day_cols, series = load_series_matrix(mode=args.mode)
    calendar_matrix = _load_calendar_matrix_for_checkpoint(day_cols=day_cols, data_size=data_size, metadata_path=metadata_path)

    sizes = args.num_generations
    results = []

    for n_gen in sizes:
        model.number_generations_per_forward_call = n_gen
        metrics = _rolling_eval(
            model, series, calendar_matrix, device,
            batch_size=args.batch_size, eval_split=args.eval_split,
            max_origins=args.max_origins,
        )
        _print_metrics(f"Ensemble draws = {n_gen}", metrics)
        results.append({"num_generations": n_gen, **metrics})

    out_path = ARTIFACTS_DIR / "ablation_ensemble_size.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ablation studies on a trained forecasting checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Shared flags
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument("--mode", choices=["quick", "full"], default=None)
    parser.add_argument("--eval-split", choices=["all", "val", "test"], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-origins", type=int, default=None)
    parser.add_argument("--save", action="store_true", help="Persist the modified checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint path, file name, or suffix token. "
            "Examples: ../artifacts/model_checkpoint.pt, "
            "model_checkpoint_Energy_fullCalendar.pt, _Energy_fullCalendar"
        ),
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help=(
            "Optional metadata json path/name. If omitted, inferred from checkpoint "
            "(model_checkpoint_SUFFIX.pt -> model_metadata_SUFFIX.json)."
        ),
    )

    sub = parser.add_subparsers(dest="ablation_mode", required=True)

    # feature_zero
    fz = sub.add_parser("feature_zero", help="Zero-out feature groups and re-evaluate.")
    fz.add_argument(
        "--zero-groups", nargs="+", default=None,
        help=f"Groups to ablate.  Available: {', '.join(FEATURE_GROUPS)}.  "
             "Omit to test all groups one-by-one.",
    )

    # Architecture limitation adjustments.
    ff = sub.add_parser("freeze_finetune", help="Freeze GRU or FC, fine-tune the rest.")
    ff.add_argument("--freeze", choices=["gru", "fc"], required=True)
    ff.add_argument("--finetune-epochs", type=int, default=None)
    ff.add_argument("--lr", type=float, default=None)

    # Alternate distribution criteria evaluation.
    ls = sub.add_parser("loss_swap", help="Fine-tune with a different loss function.")
    ls.add_argument("--loss", required=True, choices=["ensemble_nll", "energy", "kernel", "energy_kernel"])
    ls.add_argument("--finetune-epochs", type=int, default=None)
    ls.add_argument("--lr", type=float, default=None)

    # Generative count manipulation test setup.
    es = sub.add_parser("ensemble_size", help="Evaluate with different ensemble draw counts.")
    es.add_argument(
        "--num-generations", type=int, nargs="+", default=None,
        help="List of ensemble sizes to test.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()
    apply_runtime_config(args.config)

    runtime_config = load_config(args.config)
    args.mode = str(cli_or_config(args.mode, runtime_config, 'inference', 'mode'))
    args.eval_split = str(cli_or_config(args.eval_split, runtime_config, 'inference', 'eval_split'))
    args.batch_size = int(cli_or_config(args.batch_size, runtime_config, 'inference', 'batch_size'))
    if args.max_origins is None:
        args.max_origins = cli_or_config(None, runtime_config, 'inference', 'max_origins')

    if hasattr(args, 'finetune_epochs') and args.finetune_epochs is not None:
        pass
    elif hasattr(args, 'finetune_epochs'):
        args.finetune_epochs = int(cli_or_config(None, runtime_config, 'ablation', 'finetune_epochs'))

    if hasattr(args, 'lr') and args.lr is not None:
        pass
    elif hasattr(args, 'lr'):
        args.lr = float(cli_or_config(None, runtime_config, 'ablation', 'lr'))

    if hasattr(args, 'num_generations') and args.num_generations is None:
        args.num_generations = list(cli_or_config(None, runtime_config, 'ablation', 'num_generations'))

    dispatch = {
        "feature_zero":    run_feature_zero,
        "freeze_finetune": run_freeze_finetune,
        "loss_swap":       run_loss_swap,
        "ensemble_size":   run_ensemble_size,
    }

    dispatch[args.ablation_mode](args)


if __name__ == "__main__":
    main()
