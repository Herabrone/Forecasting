"""Comprehensive rolling backtest for 1-day-ahead probabilistic sales forecasts.

This script evaluates a trained model by repeatedly forecasting the next day
from a moving context window (rolling forecast origins). It reports MAE, RMSE,
and R2 with a concise CLI summary.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List, Tuple

import pandas as pd
import torch

from NN import ConditionalGenerativeModel, createGenerativeGRUNN
from datapreprocessing import NORMALIZATION_EPSILON, WINDOW_SIZE, load_calendar_features
from train import (
	CALENDAR_FEATURE_SET,
	CALENDAR_PATH,
	DATA_PATH,
	MAX_DAY_COLUMNS,
	MAX_PRODUCTS,
	METADATA_FILE,
	MODEL_FILE,
	TRAIN_SPLIT,
	VAL_SPLIT,
)


METADATA_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run rolling backtest and report MAE/RMSE/R2 in the CLI."
	)
	parser.add_argument(
		"--mode",
		choices=["quick", "full"],
		default="quick",
		help="Evaluation data scope. quick uses fewer products/days; full uses all.",
	)
	parser.add_argument(
		"--horizon",
		type=int,
		default=1,
		help="Forecast horizon in days. Current implementation supports only 1.",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=2048,
		help="Number of products per inference batch during each forecast origin.",
	)
	parser.add_argument(
		"--max-origins",
		type=int,
		default=None,
		help="Optional cap on number of rolling origins (uses most recent origins).",
	)
	parser.add_argument(
		"--eval-split",
		choices=["all", "val", "test"],
		default="test",
		help=(
			"Temporal split to evaluate. "
			"'test' (default) uses the held-out last 20%%, "
			"'val' uses the middle 20%%, "
			"'all' evaluates every rolling origin."
		),
	)
	return parser.parse_args()


def load_model(checkpoint_path=MODEL_FILE, device: str = "cpu") -> ConditionalGenerativeModel:
	checkpoint = torch.load(checkpoint_path, map_location=device)
	fc_hidden_sizes = checkpoint.get("fc_hidden_sizes")
	data_size = checkpoint.get("data_size", 1)

	net = createGenerativeGRUNN(
		data_size=data_size,
		gru_hidden_size=checkpoint["gru_hidden_size"],
		noise_size=checkpoint["noise_size"],
		output_size=checkpoint["output_size"],
		hidden_sizes=fc_hidden_sizes,
	)()

	model = ConditionalGenerativeModel(
		net=net,
		size_auxiliary_variable=checkpoint["noise_size"],
		number_generations_per_forward_call=checkpoint["number_generations_per_forward_call"],
	).to(device)

	model.load_state_dict(checkpoint["model_state_dict"])
	model.eval()
	return model


def load_series_matrix(mode: str) -> Tuple[List[str], List[str], torch.Tensor]:
	dataset = pd.read_csv(DATA_PATH)

	if mode == "quick":
		dataset = dataset.head(MAX_PRODUCTS)

	day_cols = [col for col in dataset.columns if col not in METADATA_COLS]
	if mode == "quick":
		day_cols = day_cols[-max(WINDOW_SIZE + 1, MAX_DAY_COLUMNS):]

	product_ids = dataset["id"].astype(str).tolist()
	series = torch.tensor(dataset[day_cols].to_numpy(dtype="float32"), dtype=torch.float32)
	return product_ids, day_cols, series


def load_calendar_matrix(day_cols: List[str], data_size: int) -> torch.Tensor | None:
	if data_size == 1:
		return None

	if not CALENDAR_PATH.exists():
		raise FileNotFoundError(f"Calendar file not found: {CALENDAR_PATH}")

	calendar_values, calendar_feature_names = load_calendar_features(
		CALENDAR_PATH,
		day_cols,
		feature_set=CALENDAR_FEATURE_SET,
	)
	expected_calendar_features = data_size - 1

	if METADATA_FILE.exists():
		with METADATA_FILE.open("r", encoding="utf-8") as file_handle:
			metadata = json.load(file_handle)
		expected_feature_names = metadata.get("calendar_feature_names", [])
		if isinstance(expected_feature_names, list) and len(expected_feature_names) > 0:
			# Rebuild in training order and keep missing categories as zero columns.
			feature_to_idx = {name: idx for idx, name in enumerate(calendar_feature_names)}
			aligned_columns = []
			missing_names = []
			for feature_name in expected_feature_names:
				feature_idx = feature_to_idx.get(feature_name)
				if feature_idx is None:
					aligned_columns.append(torch.zeros(len(day_cols), dtype=torch.float32).numpy())
					missing_names.append(feature_name)
				else:
					aligned_columns.append(calendar_values[:, feature_idx])

			calendar_values = torch.stack(
				[torch.tensor(column, dtype=torch.float32) for column in aligned_columns],
				dim=1,
			).numpy()
			calendar_feature_names = expected_feature_names

			if missing_names:
				preview = ", ".join(missing_names[:5])
				print(
					"Warning: Missing calendar features from current slice were zero-filled: "
					f"{preview}"
				)

	if calendar_values.shape[1] != expected_calendar_features:
		raise ValueError(
			f"Checkpoint expects {expected_calendar_features} calendar features (data_size={data_size}), "
			f"but built {calendar_values.shape[1]} from calendar.csv. "
			f"If this checkpoint was trained with rich calendar features, ensure {METADATA_FILE} is present."
		)

	print(
		f"Loaded calendar covariates for backtest: {calendar_values.shape[1]} features "
		f"over {calendar_values.shape[0]} days"
	)
	print(f"Sample calendar columns: {calendar_feature_names[:5]}")
	return torch.tensor(calendar_values, dtype=torch.float32)


def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
	residuals = y_pred - y_true
	mae = residuals.abs().mean().item()
	rmse = math.sqrt((residuals.pow(2).mean().item()))

	true_mean = y_true.mean()
	sst = ((y_true - true_mean).pow(2)).sum().item()
	sse = (residuals.pow(2)).sum().item()
	r2 = float("nan") if sst == 0 else 1.0 - (sse / sst)

	return {"mae": mae, "rmse": rmse, "r2": r2, "sse": sse, "sst": sst}


def model_predict_mean(
	model: ConditionalGenerativeModel,
	context_batch: torch.Tensor,
	device: str,
) -> torch.Tensor:
	with torch.no_grad():
		ensemble_preds = model(context_batch.to(device, non_blocking=True))
	return ensemble_preds.squeeze(-1).mean(dim=1).to("cpu")


def normalize_context_batch(context_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Normalize each product context using only its own history."""

	context_mean = context_batch.mean(dim=1, keepdim=True)
	context_std = context_batch.std(dim=1, keepdim=True, unbiased=False)
	safe_std = torch.where(
		context_std > NORMALIZATION_EPSILON,
		context_std,
		torch.ones_like(context_std),
	)
	normalized_context = (context_batch - context_mean) / safe_std
	return normalized_context, context_mean.squeeze(1), safe_std.squeeze(1)


def normalize_calendar_context(calendar_context: torch.Tensor) -> torch.Tensor:
	"""Normalize calendar covariates for one rolling origin window."""

	feature_mean = calendar_context.mean(dim=0, keepdim=True)
	feature_std = calendar_context.std(dim=0, keepdim=True, unbiased=False)
	safe_std = torch.where(
		feature_std > NORMALIZATION_EPSILON,
		feature_std,
		torch.ones_like(feature_std),
	)
	return (calendar_context - feature_mean) / safe_std


def _split_origin_range(
	num_days: int,
	horizon: int,
	eval_split: str,
) -> Tuple[int, int]:
	"""Return (first_origin, last_origin) day indices for the requested split."""

	total_windows = num_days - WINDOW_SIZE
	train_end = int(math.floor(total_windows * TRAIN_SPLIT))
	val_end = int(math.floor(total_windows * (TRAIN_SPLIT + VAL_SPLIT)))
	train_end = max(1, min(train_end, total_windows - 2))
	val_end = max(train_end + 1, min(val_end, total_windows - 1))

	if eval_split == "val":
		# Val windows: [train_end, val_end)  →  origins: train_end+WINDOW_SIZE .. val_end+WINDOW_SIZE-1
		return WINDOW_SIZE + train_end, WINDOW_SIZE + val_end - 1
	if eval_split == "test":
		# Test windows: [val_end, total_windows)  →  origins: val_end+WINDOW_SIZE .. num_days-horizon
		return WINDOW_SIZE + val_end, num_days - horizon
	# "all"
	return WINDOW_SIZE, num_days - horizon


def run_rolling_backtest(
	model: ConditionalGenerativeModel,
	series: torch.Tensor,
	calendar_matrix: torch.Tensor | None,
	horizon: int,
	device: str,
	batch_size: int,
	max_origins: int | None,
	eval_split: str = "test",
) -> Tuple[Dict[str, float], pd.DataFrame, torch.Tensor, torch.Tensor]:
	if horizon != 1:
		raise ValueError("Only 1-day horizon is currently supported.")

	num_products, num_days = series.shape
	first_origin, last_origin = _split_origin_range(num_days, horizon, eval_split)
	origins = list(range(first_origin, last_origin + 1))

	if not origins:
		raise ValueError(
			f"No rolling origins available for eval_split='{eval_split}'. "
			"Check that the series has enough day columns for the requested split."
		)

	if max_origins is not None and max_origins > 0:
		origins = origins[-max_origins:]

	origin_rows = []
	all_true_batches = []
	all_pred_batches = []

	total_origins = len(origins)
	for origin_idx, origin in enumerate(origins, start=1):
		true_values = series[:, origin]

		normalized_calendar_window = None
		if calendar_matrix is not None:
			raw_calendar_window = calendar_matrix[origin - WINDOW_SIZE:origin, :]
			normalized_calendar_window = normalize_calendar_context(raw_calendar_window)

		pred_chunks = []
		for start in range(0, num_products, batch_size):
			end = min(start + batch_size, num_products)
			raw_context = series[start:end, origin - WINDOW_SIZE:origin]
			normalized_context, context_mean, context_std = normalize_context_batch(raw_context)

			if normalized_calendar_window is not None:
				calendar_batch = normalized_calendar_window.unsqueeze(0).repeat(end - start, 1, 1)
				model_context = torch.cat([normalized_context.unsqueeze(-1), calendar_batch], dim=2)
			else:
				model_context = normalized_context.unsqueeze(-1)

			pred_mean = model_predict_mean(model, model_context, device)
			pred_chunks.append(pred_mean * context_std + context_mean)

		pred_values = torch.cat(pred_chunks, dim=0)
		metrics = compute_metrics(true_values, pred_values)

		origin_rows.append(
			{
				"origin_index": origin,
				"origin_number": origin_idx,
				"mae": metrics["mae"],
				"rmse": metrics["rmse"],
				"r2": metrics["r2"],
				"sse": metrics["sse"],
				"sst": metrics["sst"],
			}
		)
		all_true_batches.append(true_values)
		all_pred_batches.append(pred_values)

		if origin_idx % 25 == 0 or origin_idx == total_origins:
			print(f"Processed rolling origin {origin_idx}/{total_origins}")

	all_true = torch.cat(all_true_batches, dim=0)
	all_pred = torch.cat(all_pred_batches, dim=0)
	overall_metrics = compute_metrics(all_true, all_pred)
	metrics_df = pd.DataFrame(origin_rows)

	return overall_metrics, metrics_df, all_true, all_pred


def print_summary(overall: Dict[str, float], per_origin: pd.DataFrame, mode: str, horizon: int, eval_split: str = "test") -> None:
	print("\nBacktest summary")
	print("-" * 72)
	print(f"Mode: {mode}")
	print(f"Eval split: {eval_split}")
	print(f"Horizon: {horizon} day")
	print(f"Origins evaluated: {len(per_origin)}")
	print("-" * 72)
	print(f"Overall MAE : {overall['mae']:.4f}")
	print(f"Overall RMSE: {overall['rmse']:.4f}")
	print(f"Overall R2  : {overall['r2']:.4f}")
	print("-" * 72)
	print("Per-origin metric means")
	print(f"MAE mean:  {per_origin['mae'].mean():.4f}")
	print(f"RMSE mean: {per_origin['rmse'].mean():.4f}")

	valid_r2 = per_origin["r2"].replace([float("inf"), float("-inf")], float("nan")).dropna()
	if len(valid_r2) > 0:
		print(f"R2 mean (raw):      {valid_r2.mean():.4f}")
		print(f"R2 median (robust): {valid_r2.median():.4f}")
	else:
		print("R2 mean/median: unavailable (no valid per-origin R2 values)")

	sst_total = per_origin["sst"].sum()
	if sst_total > 0:
		weighted_r2 = 1.0 - (per_origin["sse"].sum() / sst_total)
		print(f"R2 SST-weighted:    {weighted_r2:.4f}")

	low_variance_origins = int((per_origin["sst"] < 1e-6).sum())
	if low_variance_origins > 0:
		print(f"Low-variance origins (sst < 1e-6): {low_variance_origins}")


def main() -> None:
	args = parse_args()
	device = "cuda" if torch.cuda.is_available() else "cpu"

	print(f"Loading model from: {MODEL_FILE}")
	print(f"Device: {device}")

	model = load_model(device=device)
	model_data_size = model.net.gru.input_size
	product_ids, day_cols, series = load_series_matrix(mode=args.mode)
	calendar_matrix = load_calendar_matrix(day_cols=day_cols, data_size=model_data_size)
	print(f"Products: {len(product_ids)}")
	print(f"Day columns in evaluation: {series.shape[1]}")
	print(f"Model input size: {model_data_size}")

	print(f"Eval split: {args.eval_split}")

	overall, per_origin, _all_true, _all_pred = run_rolling_backtest(
		model=model,
		series=series,
		calendar_matrix=calendar_matrix,
		horizon=args.horizon,
		device=device,
		batch_size=args.batch_size,
		max_origins=args.max_origins,
		eval_split=args.eval_split,
	)

	print_summary(overall=overall, per_origin=per_origin, mode=args.mode, horizon=args.horizon, eval_split=args.eval_split)


if __name__ == "__main__":
	main()
