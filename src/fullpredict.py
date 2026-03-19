"""Comprehensive rolling backtest for 1-day-ahead probabilistic sales forecasts.

This script evaluates a trained model by repeatedly forecasting the next day
from a moving context window (rolling forecast origins). It reports MAE, RMSE,
and R2 with a concise CLI summary.
"""

from __future__ import annotations

import argparse
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
	MODEL_FILE,
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
	if calendar_values.shape[1] != expected_calendar_features:
		raise ValueError(
			f"Checkpoint expects {expected_calendar_features} calendar features (data_size={data_size}), "
			f"but built {calendar_values.shape[1]} from calendar.csv."
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


def run_rolling_backtest(
	model: ConditionalGenerativeModel,
	series: torch.Tensor,
	calendar_matrix: torch.Tensor | None,
	horizon: int,
	device: str,
	batch_size: int,
	max_origins: int | None,
) -> Tuple[Dict[str, float], pd.DataFrame, torch.Tensor, torch.Tensor]:
	if horizon != 1:
		raise ValueError("Only 1-day horizon is currently supported.")

	num_products, num_days = series.shape
	first_origin = WINDOW_SIZE
	last_origin = num_days - horizon
	origins = list(range(first_origin, last_origin + 1))

	if not origins:
		raise ValueError("Not enough day columns to build rolling windows.")

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


def print_summary(overall: Dict[str, float], per_origin: pd.DataFrame, mode: str, horizon: int) -> None:
	print("\nBacktest summary")
	print("-" * 72)
	print(f"Mode: {mode}")
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

	overall, per_origin, _all_true, _all_pred = run_rolling_backtest(
		model=model,
		series=series,
		calendar_matrix=calendar_matrix,
		horizon=args.horizon,
		device=device,
		batch_size=args.batch_size,
		max_origins=args.max_origins,
	)

	print_summary(overall=overall, per_origin=per_origin, mode=args.mode, horizon=args.horizon)


if __name__ == "__main__":
	main()
