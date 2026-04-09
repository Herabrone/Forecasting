"""Data preprocessing utilities for M5 sales forecasting.

This module builds model-ready sliding windows for sales history and optional
calendar covariates, with per-window normalization to avoid future leakage.
"""

import pandas as pd
import numpy as np

WINDOW_SIZE = 28
NORMALIZATION_EPSILON = 1e-8


def _parse_day_number(day_label):
    """Extract integer day index from labels like d_1, d_1913."""

    if not isinstance(day_label, str) or not day_label.startswith('d_'):
        raise ValueError(f"Invalid day label: {day_label}")
    try:
        return int(day_label.split('_', maxsplit=1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Invalid day label: {day_label}") from error


def _build_minimal_calendar_features(aligned_calendar):
    """Create compact calendar features from date columns."""

    features = []
    names = []

    if 'date' in aligned_calendar.columns:
        date_series = pd.to_datetime(aligned_calendar['date'], errors='coerce')
        if date_series.isna().any():
            raise ValueError('calendar.csv contains invalid date values.')

        day_of_week = date_series.dt.dayofweek.astype(np.float32).to_numpy()
        month = date_series.dt.month.astype(np.float32).to_numpy()

        features.extend(
            [
                np.sin(2.0 * np.pi * day_of_week / 7.0).astype(np.float32),
                np.cos(2.0 * np.pi * day_of_week / 7.0).astype(np.float32),
                np.sin(2.0 * np.pi * month / 12.0).astype(np.float32),
                np.cos(2.0 * np.pi * month / 12.0).astype(np.float32),
                (day_of_week >= 5).astype(np.float32),
            ]
        )
        names.extend(['dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'is_weekend'])
        return features, names

    if 'wday' in aligned_calendar.columns:
        wday = aligned_calendar['wday'].astype(np.float32).to_numpy() - 1.0
        features.extend(
            [
                np.sin(2.0 * np.pi * wday / 7.0).astype(np.float32),
                np.cos(2.0 * np.pi * wday / 7.0).astype(np.float32),
                (wday >= 5).astype(np.float32),
            ]
        )
        names.extend(['dow_sin', 'dow_cos', 'is_weekend'])
        return features, names

    raise ValueError("calendar.csv must include 'date' or 'wday' for date-derived features.")


def build_calendar_features(calendar_df, day_cols, feature_set='rich'):
    """Align calendar rows to selected sales days and build numeric feature matrix."""

    if not day_cols:
        raise ValueError('day_cols is empty; cannot align calendar features.')

    aligned_calendar = calendar_df.copy()
    if 'd' not in aligned_calendar.columns:
        raise ValueError("calendar.csv must include a 'd' column (for example: d_1, d_2).")

    aligned_calendar['d'] = aligned_calendar['d'].astype(str)
    calendar_by_day = aligned_calendar.set_index('d', drop=False)

    missing_days = [day for day in day_cols if day not in calendar_by_day.index]
    if missing_days:
        preview = ', '.join(missing_days[:5])
        raise ValueError(f'calendar.csv is missing required day labels: {preview}')

    aligned_calendar = calendar_by_day.loc[day_cols].copy()

    day_numbers = np.array([_parse_day_number(day) for day in day_cols], dtype=np.float32)
    feature_values = [day_numbers]
    feature_names = ['day_index']

    minimal_values, minimal_names = _build_minimal_calendar_features(aligned_calendar)
    feature_values.extend(minimal_values)
    feature_names.extend(minimal_names)

    if feature_set == 'rich':
        for snap_col in ('snap_CA', 'snap_TX', 'snap_WI'):
            if snap_col in aligned_calendar.columns:
                feature_values.append(aligned_calendar[snap_col].fillna(0).astype(np.float32).to_numpy())
                feature_names.append(snap_col)

        event_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
        for event_col in event_cols:
            if event_col not in aligned_calendar.columns:
                continue
            encoded = pd.get_dummies(
                aligned_calendar[event_col].fillna('none').astype(str),
                prefix=event_col,
                dtype=np.float32,
            )
            if encoded.shape[1] == 0:
                continue
            feature_values.extend([encoded[column].to_numpy(dtype=np.float32) for column in encoded.columns])
            feature_names.extend(encoded.columns.tolist())
    elif feature_set != 'minimal':
        raise ValueError(f"Unsupported calendar feature_set: {feature_set}. Use 'minimal' or 'rich'.")

    feature_matrix = np.column_stack(feature_values).astype(np.float32)
    return feature_matrix, feature_names


def load_calendar_features(calendar_path, day_cols, feature_set='rich'):
    """Load calendar CSV from disk and return aligned feature matrix."""

    calendar_df = pd.read_csv(calendar_path)
    feature_matrix, feature_names = build_calendar_features(calendar_df, day_cols, feature_set=feature_set)
    return feature_matrix, feature_names


def normalize_context_window(values):
    """Normalize a single context window using only its own history."""

    values = np.asarray(values, dtype=np.float32)
    mean = float(values.mean())
    std = float(values.std())
    safe_std = std if std > NORMALIZATION_EPSILON else 1.0
    normalized_values = (values - mean) / safe_std
    return normalized_values.astype(np.float32), mean, safe_std


def normalize_windows_and_targets(windows, targets):
    """Normalize each training sample from its context window to avoid future leakage."""

    context_mean = windows.mean(axis=2, keepdims=True)
    context_std = windows.std(axis=2, keepdims=True)
    safe_std = np.where(context_std > NORMALIZATION_EPSILON, context_std, 1.0)

    normalized_windows = (windows - context_mean) / safe_std
    normalized_targets = (targets - context_mean.squeeze(-1)) / safe_std.squeeze(-1)
    return normalized_windows.astype(np.float32), normalized_targets.astype(np.float32)


def normalize_calendar_windows(calendar_windows):
    """Normalize calendar covariates per window and feature."""

    feature_mean = calendar_windows.mean(axis=1, keepdims=True)
    feature_std = calendar_windows.std(axis=1, keepdims=True)
    safe_std = np.where(feature_std > NORMALIZATION_EPSILON, feature_std, 1.0)
    normalized = (calendar_windows - feature_mean) / safe_std
    return normalized.astype(np.float32)

def process(days, calendar_features=None):
    """Build model-ready windows from sales history."""

    # Convert sales counts to float32 before window construction.
    days_continuous = days.astype(np.float32).to_numpy()  # [num_products, num_days]

    if calendar_features is not None:
        calendar_features = np.asarray(calendar_features, dtype=np.float32)
        if calendar_features.ndim != 2:
            raise ValueError('calendar_features must be a 2D array shaped [num_days, num_features].')
        if calendar_features.shape[0] != days_continuous.shape[1]:
            raise ValueError(
                f'calendar_features day count ({calendar_features.shape[0]}) does not match '
                f'sales day count ({days_continuous.shape[1]}).'
            )

    # Build and normalize context-target pairs.
    windows, targets = sliding_window(days_continuous)    # windows: [num_windows, num_products, WINDOW_SIZE]
    windows, targets = normalize_windows_and_targets(windows, targets)

    # Reshape to per-sample layout: each (product, window) pair becomes one sample
    sales_X = np.transpose(windows, (1, 0, 2)).reshape(-1, WINDOW_SIZE, 1).astype(np.float32)

    if calendar_features is not None:
        num_products = days_continuous.shape[0]
        num_windows = windows.shape[0]

        calendar_windows = np.array(
            [calendar_features[i:i + WINDOW_SIZE] for i in range(num_windows)],
            dtype=np.float32,
        )
        calendar_windows = normalize_calendar_windows(calendar_windows)

        # Calendar values are day-level, so the same calendar window is repeated across products.
        calendar_X = np.repeat(calendar_windows[np.newaxis, ...], repeats=num_products, axis=0)
        calendar_X = calendar_X.reshape(-1, WINDOW_SIZE, calendar_windows.shape[-1]).astype(np.float32)
        X = np.concatenate([sales_X, calendar_X], axis=2).astype(np.float32)
    else:
        X = sales_X

    y = np.transpose(targets, (1, 0)).reshape(-1, 1).astype(np.float32)

    return X, y

def sliding_window(days):
    """Construct context windows and next-step targets for each product."""

    num_products, num_days = days.shape
    num_windows = num_days - WINDOW_SIZE

    windows = np.array([days[:, i:i + WINDOW_SIZE] for i in range(num_windows)])
    targets = np.array([days[:, i + WINDOW_SIZE] for i in range(num_windows)])

    return windows, targets

if __name__ == '__main__':

    # Full dataset
    dataset = pd.read_csv('sales_train_validation.csv')

    metadata_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    days_cols = [col for col in dataset.columns if col not in metadata_cols]

    days = dataset[days_cols]

    # test_arr = np.array([
    #     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    #     [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    # ])

    days_numpy = days.to_numpy()

    # Quick local check for array shapes.
    windows, targets = sliding_window(days_numpy)
    print(windows.shape)
    print(targets.shape)

    X, y = process(days) # Process the data
    print(X.shape)
    print(y.shape)