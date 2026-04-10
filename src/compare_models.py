"""Compare a neural forecast model against classical time-series baselines.

This script runs a one-day-ahead rolling backtest on sales_train_validation.csv
using the same origin split strategy as fullpredict.py. It evaluates multiple
baseline models alongside the neural network to produce a leaderboard. The results
can then be plotted for visual comparison.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

import datapreprocessing as dp
import fullpredict as fp
from config import cli_or_config, load_config, resolve_path
from fullpredict import (
    _split_origin_range,
    compute_metrics,
    load_calendar_matrix,
    load_model,
    load_series_matrix,
    model_predict_mean,
    normalize_calendar_context,
    normalize_context_batch,
)


NEURAL_PROGRESS_INTERVAL = 25


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the comparison script.
    
    Returns:
        The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compare forecast models using rolling backtest metrics and presentation-ready plots."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument("--mode", choices=["quick", "full"], default=None)
    parser.add_argument("--eval-split", choices=["all", "val", "test"], default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-origins", type=int, default=None)
    parser.add_argument(
        "--max-products",
        type=int,
        default=None,
        help=(
            "Cap number of products for all models for runtime control. "
            "Use a large value for full panel comparison."
        ),
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to run.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    return parser.parse_args()


def apply_runtime_config(args: argparse.Namespace):
    """Apply combinations of CLI arguments and YAML configuration to global constants.
    
    This function overrides default configuration in the `dp` and `fp` modules 
    according to the specified priority (CLI overrides YAML, YAML overrides default).
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        A tuple containing runtime settings:
        (mode, eval_split, horizon, batch_size, max_origins, max_products, models, output_dir, device).
    """
    config = load_config(args.config)

    window_size = int(cli_or_config(None, config, 'data', 'window_size'))
    dp.WINDOW_SIZE = window_size
    fp.WINDOW_SIZE = window_size

    fp.DATA_PATH = resolve_path(config, 'data_path', fp.DATA_PATH)
    fp.CALENDAR_PATH = resolve_path(config, 'calendar_path', fp.CALENDAR_PATH)
    fp.METADATA_FILE = resolve_path(config, 'metadata_file', fp.METADATA_FILE)
    fp.MODEL_FILE = resolve_path(config, 'model_file', fp.MODEL_FILE)
    fp.CALENDAR_FEATURE_SET = str(cli_or_config(None, config, 'data', 'calendar_feature_set'))
    fp.MAX_DAY_COLUMNS = int(cli_or_config(None, config, 'data', 'max_day_columns'))
    fp.MAX_PRODUCTS = int(cli_or_config(None, config, 'data', 'max_products'))
    fp.TRAIN_SPLIT = float(cli_or_config(None, config, 'training', 'train_split'))
    fp.VAL_SPLIT = float(cli_or_config(None, config, 'training', 'val_split'))

    mode = str(cli_or_config(args.mode, config, 'inference', 'mode'))
    eval_split = str(cli_or_config(args.eval_split, config, 'inference', 'eval_split'))
    horizon = int(cli_or_config(args.horizon, config, 'inference', 'horizon'))
    batch_size = int(cli_or_config(args.batch_size, config, 'inference', 'batch_size'))
    max_origins = cli_or_config(args.max_origins, config, 'inference', 'max_origins')
    max_products = int(cli_or_config(args.max_products, config, 'comparison', 'max_products'))
    models = str(cli_or_config(args.models, config, 'comparison', 'models'))
    output_dir = str(cli_or_config(args.output_dir, config, 'comparison', 'output_dir'))
    device = str(cli_or_config(args.device, config, 'inference', 'device'))

    return mode, eval_split, horizon, batch_size, max_origins, max_products, models, output_dir, device


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def parse_model_list(raw_models: str) -> List[str]:
    valid = {
        "neural",
        "seasonal7",
        "seasonal28",
        "ma7",
        "ma28",
    }
    requested = [name.strip().lower() for name in raw_models.split(",") if name.strip()]
    unknown = [name for name in requested if name not in valid]
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}. Valid: {sorted(valid)}")
    return requested


def build_origins(num_days: int, horizon: int, eval_split: str, max_origins: int | None) -> List[int]:
    first_origin, last_origin = _split_origin_range(num_days, horizon, eval_split)
    origins = list(range(first_origin, last_origin + 1))
    if max_origins is not None and max_origins > 0:
        origins = origins[-max_origins:]
    return origins


def masked_metrics(y_true_values: np.ndarray, y_pred_values: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics for finite predicted and true values.
    
    This function excludes any `NaN` values resulting from unforecastable points
    and returns comprehensive accuracy data.
    
    Args:
        y_true_values: The ground-truth numeric values.
        y_pred_values: The model-predicted numeric values.
        
    Returns:
        A dictionary containing standard metrics (mae, rmse, r2, sse, sst) 
        plus the number of valid points evaluated (n).
    """
    valid_mask = np.isfinite(y_true_values) & np.isfinite(y_pred_values)
    valid_count = int(valid_mask.sum())
    if valid_count == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "sse": 0.0,
            "sst": 0.0,
            "n": 0,
        }

    y_true = torch.tensor(y_true_values[valid_mask], dtype=torch.float32)
    y_pred = torch.tensor(y_pred_values[valid_mask], dtype=torch.float32)
    metrics = compute_metrics(y_true, y_pred)
    metrics["n"] = valid_count
    return metrics


def compute_overall_and_origin_metrics(
    true_matrix: np.ndarray,
    pred_matrix: np.ndarray,
    origins: List[int],
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Calculate accuracy metrics both across all origins and for each origin individually.
    
    This allows a unified review of the predictions along with drill-down
    accuracy metrics by day.
    
    Args:
        true_matrix: True target values matching the prediction outputs.
        pred_matrix: Forecasted values matched against the targets.
        origins: Origin indices used for the forecasts.
        
    Returns:
        A tuple of (overall_metrics, per_origin_dataframe). The dataframe 
        contains row-level details indexed via the input `origins`.
    """
    overall = masked_metrics(true_matrix.reshape(-1), pred_matrix.reshape(-1))

    rows = []
    for origin_idx, origin in enumerate(origins):
        metrics = masked_metrics(true_matrix[:, origin_idx], pred_matrix[:, origin_idx])
        rows.append(
            {
                "origin_index": origin,
                "origin_number": origin_idx + 1,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "sse": metrics["sse"],
                "sst": metrics["sst"],
                "n_evaluated": metrics["n"],
            }
        )

    per_origin = pd.DataFrame(rows)
    return overall, per_origin


def run_neural_model(
    series: torch.Tensor,
    calendar_matrix: torch.Tensor | None,
    origins: List[int],
    batch_size: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    model = load_model(device=device)
    model.eval()

    num_products = series.shape[0]
    true_matrix = np.zeros((num_products, len(origins)), dtype=np.float32)
    pred_matrix = np.zeros((num_products, len(origins)), dtype=np.float32)

    for origin_number, origin in enumerate(origins, start=1):
        true_values = series[:, origin]
        true_matrix[:, origin_number - 1] = true_values.cpu().numpy()

        normalized_calendar_window = None
        if calendar_matrix is not None:
            raw_calendar_window = calendar_matrix[origin - dp.WINDOW_SIZE : origin, :]
            normalized_calendar_window = normalize_calendar_context(raw_calendar_window)

        pred_chunks = []
        for start in range(0, num_products, batch_size):
            end = min(start + batch_size, num_products)
            raw_context = series[start:end, origin - dp.WINDOW_SIZE : origin]
            normalized_context, context_mean, context_std = normalize_context_batch(raw_context)

            if normalized_calendar_window is not None:
                calendar_batch = normalized_calendar_window.unsqueeze(0).repeat(end - start, 1, 1)
                model_context = torch.cat([normalized_context.unsqueeze(-1), calendar_batch], dim=2)
            else:
                model_context = normalized_context.unsqueeze(-1)

            pred_mean = model_predict_mean(model, model_context, device)
            pred_chunks.append(pred_mean * context_std + context_mean)

        pred_values = torch.cat(pred_chunks, dim=0)
        pred_matrix[:, origin_number - 1] = pred_values.cpu().numpy()

        if origin_number % NEURAL_PROGRESS_INTERVAL == 0 or origin_number == len(origins):
            print(f"[neural] processed origin {origin_number}/{len(origins)}")

    return true_matrix, pred_matrix


def run_vector_baseline(series_np: np.ndarray, origins: List[int], model_key: str) -> Tuple[np.ndarray, np.ndarray]:
    num_products = series_np.shape[0]
    true_matrix = np.zeros((num_products, len(origins)), dtype=np.float32)
    pred_matrix = np.zeros((num_products, len(origins)), dtype=np.float32)

    for origin_idx, origin in enumerate(origins):
        true_matrix[:, origin_idx] = series_np[:, origin]
        if model_key == "seasonal7":
            pred = series_np[:, origin - 7]
        elif model_key == "seasonal28":
            pred = series_np[:, origin - 28]
        elif model_key == "ma7":
            pred = series_np[:, origin - 7 : origin].mean(axis=1)
        elif model_key == "ma28":
            pred = series_np[:, origin - 28 : origin].mean(axis=1)
        else:
            raise ValueError(f"Unsupported vector baseline: {model_key}")
        pred_matrix[:, origin_idx] = pred.astype(np.float32)

    return true_matrix, pred_matrix











def weighted_r2(per_origin: pd.DataFrame) -> float:
    total_sst = float(per_origin["sst"].sum(skipna=True))
    if total_sst <= 0:
        return float("nan")
    total_sse = float(per_origin["sse"].sum(skipna=True))
    return 1.0 - (total_sse / total_sst)


def save_comparison_results(leaderboard_df: pd.DataFrame, per_origin_df: pd.DataFrame, output_dir: Path) -> Tuple[Path, Path]:
    """Save the final ranking and per-origin metrics to disk.
    
    This function serializes the analysis tables so they can be reviewed outside
    of memory.
    
    Args:
        leaderboard_df: The aggregated metrics ranked across models.
        per_origin_df: Detailed rows for each prediction origin.
        output_dir: Directory where the target CSV files will be placed.
        
    Returns:
        A tuple of paths pointing to the leaderboard and per-origin CSVs respectively.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_csv = output_dir / "compare_models_metrics.csv"
    per_origin_csv = output_dir / "compare_models_per_origin.csv"

    leaderboard_df.to_csv(leaderboard_csv, index=False)
    per_origin_df.to_csv(per_origin_csv, index=False)
    
    return leaderboard_csv, per_origin_csv


def plot_outputs(leaderboard_df: pd.DataFrame, per_origin_df: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("Plotting requires matplotlib. Install with: pip install matplotlib") from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    ranked = leaderboard_df.sort_values("rmse", ascending=True).reset_index(drop=True)

    bar_colors = ["#F26430" if key == "neural" else "#A0AEC0" for key in ranked["model_key"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].barh(ranked["model_display"], ranked["rmse"], color=bar_colors)
    axes[0].set_title("Root Mean Square Error (RMSE) comparison")
    axes[0].set_xlabel("RMSE")
    axes[0].grid(axis="x", alpha=0.2)

    axes[1].barh(ranked["model_display"], ranked["weighted_r2"], color=bar_colors)
    axes[1].set_title("SST-weighted R² comparison")
    axes[1].set_xlabel("Weighted R²")
    axes[1].grid(axis="x", alpha=0.2)

    fig.suptitle("Forecast Model Performance Comparison", fontsize=15, fontweight="bold")
    fig.tight_layout()
    leaderboard_png = output_dir / "compare_models_leaderboard.png"
    fig.savefig(leaderboard_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for model_key, group in per_origin_df.groupby("model_key", sort=False):
        sorted_group = group.sort_values("origin_number")
        trend = sorted_group["rmse"].rolling(7, min_periods=1).mean()
        if model_key == "neural":
            ax.plot(
                sorted_group["origin_number"],
                trend,
                color="#F26430",
                linewidth=2.8,
                label="Neural forecast",
            )
        else:
            ax.plot(
                sorted_group["origin_number"],
                trend,
                linewidth=1.4,
                alpha=0.85,
                label=sorted_group["model_display"].iloc[0],
            )

    ax.set_title("Per-Origin RMSE Trend (7-origin moving average)")
    ax.set_xlabel("Origin number")
    ax.set_ylabel("RMSE")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    fig.tight_layout()
    trend_png = output_dir / "compare_models_per_origin_rmse.png"
    fig.savefig(trend_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {leaderboard_png}")
    print(f"Saved plot: {trend_png}")


def run_single_model(
    model_key: str,
    series: torch.Tensor,
    series_np: np.ndarray,
    origins: List[int],
    batch_size: int,
    device: str,
    calendar_matrix: torch.Tensor | None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    display_names = {
        "neural": "Neural forecast",
        "seasonal7": "Seasonal-7",
        "seasonal28": "Seasonal-28",
        "ma7": "MovingAvg-7",
        "ma28": "MovingAvg-28",
    }

    if model_key == "neural":
        true_matrix, pred_matrix = run_neural_model(
            series=series,
            calendar_matrix=calendar_matrix,
            origins=origins,
            batch_size=batch_size,
            device=device,
        )
    elif model_key in {"seasonal7", "seasonal28", "ma7", "ma28"}:
        true_matrix, pred_matrix = run_vector_baseline(series_np, origins, model_key)
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    return true_matrix, pred_matrix, display_names[model_key]


def main() -> None:
    args = parse_args()
    (
        mode,
        eval_split,
        horizon,
        batch_size,
        max_origins,
        max_products,
        models,
        output_dir_raw,
        device_name,
    ) = apply_runtime_config(args)

    if horizon != 1:
        raise ValueError("This script currently supports only --horizon 1.")

    requested_models = parse_model_list(models)
    device = choose_device(device_name)
    output_dir = Path(output_dir_raw)

    print(f"Device: {device}")
    print(f"Requested models: {requested_models}")

    product_ids, day_cols, series = load_series_matrix(mode=mode)

    if max_products is not None and max_products > 0:
        capped_products = min(max_products, series.shape[0])
        series = series[:capped_products, :]
        product_ids = product_ids[:capped_products]

    num_products, num_days = series.shape
    origins = build_origins(num_days, horizon, eval_split, max_origins)
    if not origins:
        raise ValueError("No rolling origins available for requested split and settings.")

    print(f"Products evaluated: {num_products}")
    print(f"Day columns in evaluation: {num_days}")
    print(f"Origins evaluated: {len(origins)}")

    calendar_matrix = None
    if "neural" in requested_models:
        model = load_model(device=device)
        model_data_size = model.net.gru.input_size
        del model
        calendar_matrix = load_calendar_matrix(day_cols=day_cols, data_size=model_data_size)

    series_np = series.cpu().numpy()

    leaderboard_rows = []
    per_origin_frames = []

    for model_key in requested_models:
        print("-" * 72)
        print(f"Running model: {model_key}")

        try:
            true_matrix, pred_matrix, display_name = run_single_model(
                model_key=model_key,
                series=series,
                series_np=series_np,
                origins=origins,
                batch_size=batch_size,
                device=device,
                calendar_matrix=calendar_matrix,
            )
        except RuntimeError as runtime_error:
            print(f"Skipping {model_key}: {runtime_error}")
            continue

        overall, per_origin = compute_overall_and_origin_metrics(true_matrix, pred_matrix, origins)
        valid_r2 = per_origin["r2"].replace([np.inf, -np.inf], np.nan).dropna()

        row = {
            "model_key": model_key,
            "model_display": display_name,
            "mae": overall["mae"],
            "rmse": overall["rmse"],
            "overall_r2": overall["r2"],
            "weighted_r2": weighted_r2(per_origin),
            "median_origin_r2": float(valid_r2.median()) if len(valid_r2) else float("nan"),
            "n_points": int(overall["n"]),
        }
        leaderboard_rows.append(row)

        per_origin = per_origin.copy()
        per_origin["model_key"] = model_key
        per_origin["model_display"] = display_name
        per_origin_frames.append(per_origin)

        print(
            f"{display_name}: "
            f"MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}, "
            f"overall R2={row['overall_r2']:.4f}, weighted R2={row['weighted_r2']:.4f}, "
            f"n={row['n_points']}"
        )

    if not leaderboard_rows:
        raise RuntimeError("No models completed. Install missing dependencies or reduce scope.")

    leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values("rmse", ascending=True)
    per_origin_df = pd.concat(per_origin_frames, ignore_index=True)

    leaderboard_csv, per_origin_csv = save_comparison_results(leaderboard_df, per_origin_df, output_dir)

    print("-" * 72)
    print("Leaderboard (sorted by RMSE):")
    print(
        leaderboard_df[
            [
                "model_display",
                "mae",
                "rmse",
                "overall_r2",
                "weighted_r2",
                "median_origin_r2",
                "n_points",
            ]
        ]
    )
    print(f"Saved table: {leaderboard_csv}")
    print(f"Saved table: {per_origin_csv}")

    plot_outputs(leaderboard_df, per_origin_df, output_dir)


if __name__ == "__main__":
    main()
