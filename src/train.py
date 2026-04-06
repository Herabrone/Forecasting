import sys
import os
import json
import time
import gc
import math
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datapreprocessing import WINDOW_SIZE, load_calendar_features
from src.NN import createGenerativeGRUNN, ConditionalGenerativeModel, ensemble_nll_loss
from src.scoringrules import energy, kernel, energy_kernel


DATA_PATH = Path(__file__).resolve().parent / 'sales_train_validation.csv'
CALENDAR_PATH = Path(__file__).resolve().parent / 'calendar.csv'
USE_CALENDAR_FEATURES = False  # Toggle calendar covariates on/off.
CALENDAR_FEATURE_SET = 'rich'  # Supported values: 'minimal' or 'rich'.
LOSS_NAME = 'ensemble_nll'  # Training objective to optimize. 'ensemble_nll' is usually the fastest and most stable.
                             # Other options ('energy', 'kernel', 'energy_kernel') are valid but can train slower.

QUICK_RUN = True  # If True, train on a subset for faster iteration. If False, use all products and all day columns.

MAX_PRODUCTS = 1000  # Number of product time series to keep when QUICK_RUN=True.
MAX_DAY_COLUMNS = 365  # Number of most recent day columns to keep when QUICK_RUN=True.
                       # Must be >= WINDOW_SIZE + 1 to form at least one input-target pair.

BATCH_SIZE = 1024  # Samples per optimizer step.
                   # Larger batches usually increase throughput on GPU but require more VRAM.
                   # If you hit CUDA OOM, lower this first (for example: 512, then 256).

EPOCHS = 10  # Full passes over the selected training subset.
            # More epochs can improve fit but increase runtime linearly.

# Temporal split ratios for train/val/test. The code will compute actual window boundaries based on the total number of windows.
TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2

# Explicit FC sizes prevent the default architecture from shrinking to a tiny last hidden layer.
FC_HIDDEN_SIZES = [128, 64, 32]

PROGRESS_EVERY = 50  # Print progress/ETA every N batches to show how close training is to completion.

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / 'artifacts'
MODEL_FILE = ARTIFACTS_DIR / 'model_checkpoint.pt'
METADATA_FILE = ARTIFACTS_DIR / 'model_metadata.json'
LOSS_PLOT_FILE = ARTIFACTS_DIR / 'loss_vs_epoch.png'


def _day_sort_key(day_col_name):
    return int(day_col_name.split('_', maxsplit=1)[1])


class TimeSeriesWindowDataset(Dataset):
    """Build training samples lazily to avoid materializing huge X/y tensors."""

    def __init__(self, sales_matrix, window_size, calendar_features=None, window_start=0, window_end=None):
        self.sales_matrix = np.asarray(sales_matrix, dtype=np.float32)
        self.window_size = window_size
        self.calendar_features = None if calendar_features is None else np.asarray(calendar_features, dtype=np.float32)

        if self.sales_matrix.ndim != 2:
            raise ValueError('sales_matrix must be 2D [num_products, num_days].')

        self.num_products, self.num_days = self.sales_matrix.shape
        self.num_windows = self.num_days - self.window_size
        if self.num_windows <= 0:
            raise ValueError('Not enough day columns to create at least one training sample.')

        if window_end is None:
            window_end = self.num_windows
        if window_start < 0 or window_end > self.num_windows or window_start >= window_end:
            raise ValueError(
                f'Invalid window range [{window_start}, {window_end}) for {self.num_windows} available windows.'
            )

        self.window_start = window_start
        self.window_end = window_end
        self.split_num_windows = self.window_end - self.window_start

        if self.calendar_features is not None:
            if self.calendar_features.ndim != 2:
                raise ValueError('calendar_features must be 2D [num_days, num_features].')
            if self.calendar_features.shape[0] != self.num_days:
                raise ValueError(
                    f'calendar_features day count ({self.calendar_features.shape[0]}) does not match '
                    f'sales day count ({self.num_days}).'
                )

        self.data_size = 1 if self.calendar_features is None else 1 + self.calendar_features.shape[1]

    def __len__(self):
        return self.num_products * self.split_num_windows

    def __getitem__(self, index):
        product_idx = index // self.split_num_windows
        local_window_start = index % self.split_num_windows
        window_start = self.window_start + local_window_start
        window_end = window_start + self.window_size

        sales_window = self.sales_matrix[product_idx, window_start:window_end]
        target_value = self.sales_matrix[product_idx, window_end]

        sales_mean = float(sales_window.mean())
        sales_std = float(sales_window.std())
        safe_std = sales_std if sales_std > 1e-8 else 1.0

        normalized_sales = ((sales_window - sales_mean) / safe_std).astype(np.float32)
        normalized_target = np.array([(target_value - sales_mean) / safe_std], dtype=np.float32)

        sample_features = normalized_sales.reshape(self.window_size, 1)

        if self.calendar_features is not None:
            calendar_window = self.calendar_features[window_start:window_end, :]
            cal_mean = calendar_window.mean(axis=0, keepdims=True)
            cal_std = calendar_window.std(axis=0, keepdims=True)
            safe_cal_std = np.where(cal_std > 1e-8, cal_std, 1.0)
            normalized_calendar = ((calendar_window - cal_mean) / safe_cal_std).astype(np.float32)
            sample_features = np.concatenate([sample_features, normalized_calendar], axis=1)

        return torch.from_numpy(sample_features), torch.from_numpy(normalized_target)


def score_rule_loss(ensemble_preds, targets, score_fn):
    """Average a scoring rule across the batch."""

    preds = ensemble_preds.squeeze(-1)
    targets = targets.squeeze(-1)

    return score_fn(preds, targets)


def compute_loss(ensemble_preds, targets, loss_name):
    """Select which training loss to use."""

    if loss_name == 'ensemble_nll':
        return ensemble_nll_loss(ensemble_preds, targets)
    if loss_name == 'energy':
        return score_rule_loss(ensemble_preds, targets, energy)
    if loss_name == 'kernel':
        return score_rule_loss(ensemble_preds, targets, kernel)
    if loss_name == 'energy_kernel':
        return score_rule_loss(ensemble_preds, targets, energy_kernel)

    raise ValueError(f"Unknown loss_name: {loss_name}")


def _compute_temporal_split_indices(total_windows):
    """Return chronological window boundaries for 60/20/20 split."""

    if total_windows < 5:
        raise ValueError('Need at least 5 windows to perform a 60/20/20 temporal split.')

    train_end = int(math.floor(total_windows * TRAIN_SPLIT))
    val_end = int(math.floor(total_windows * (TRAIN_SPLIT + VAL_SPLIT)))

    # Keep each split non-empty even for small debug runs.
    train_end = max(1, min(train_end, total_windows - 2))
    val_end = max(train_end + 1, min(val_end, total_windows - 1))

    return train_end, val_end


def load_and_preprocess():
    """Load selected columns and return lazy train/val/test datasets."""

    metadata_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

    header_columns = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()
    all_day_cols = sorted([col for col in header_columns if col.startswith('d_')], key=_day_sort_key)
    if not all_day_cols:
        raise ValueError('No day columns found in sales file.')

    if QUICK_RUN:
        days_to_keep = max(WINDOW_SIZE + 1, MAX_DAY_COLUMNS)
        selected_day_cols = all_day_cols[-days_to_keep:]
    else:
        selected_day_cols = all_day_cols

    usecols = metadata_cols + selected_day_cols
    dtype_map = {day_col: 'int16' for day_col in selected_day_cols}
    dataset = pd.read_csv(DATA_PATH, usecols=usecols, dtype=dtype_map)

    if QUICK_RUN:
        dataset = dataset.head(MAX_PRODUCTS)

    sales_matrix = dataset[selected_day_cols].to_numpy(dtype=np.float32, copy=True)

    calendar_features = None
    calendar_feature_names = []
    if USE_CALENDAR_FEATURES:
        if not CALENDAR_PATH.exists():
            raise FileNotFoundError(f'Calendar file not found: {CALENDAR_PATH}')
        calendar_features, calendar_feature_names = load_calendar_features(
            CALENDAR_PATH,
            selected_day_cols,
            feature_set=CALENDAR_FEATURE_SET,
        )
        print(
            f"Loaded calendar covariates: {calendar_features.shape[1]} features "
            f"for {calendar_features.shape[0]} days"
        )
        print(f"Calendar feature set: {CALENDAR_FEATURE_SET}")
        print(f"Sample calendar columns: {calendar_feature_names[:5]}")

    del dataset
    gc.collect()

    base_dataset = TimeSeriesWindowDataset(
        sales_matrix=sales_matrix,
        window_size=WINDOW_SIZE,
        calendar_features=calendar_features,
    )

    train_end, val_end = _compute_temporal_split_indices(base_dataset.num_windows)

    train_dataset = TimeSeriesWindowDataset(
        sales_matrix=sales_matrix,
        window_size=WINDOW_SIZE,
        calendar_features=calendar_features,
        window_start=0,
        window_end=train_end,
    )
    val_dataset = TimeSeriesWindowDataset(
        sales_matrix=sales_matrix,
        window_size=WINDOW_SIZE,
        calendar_features=calendar_features,
        window_start=train_end,
        window_end=val_end,
    )
    test_dataset = TimeSeriesWindowDataset(
        sales_matrix=sales_matrix,
        window_size=WINDOW_SIZE,
        calendar_features=calendar_features,
        window_start=val_end,
        window_end=base_dataset.num_windows,
    )

    split_info = {
        'total_windows': base_dataset.num_windows,
        'train_windows': train_end,
        'val_windows': val_end - train_end,
        'test_windows': base_dataset.num_windows - val_end,
    }

    return train_dataset, val_dataset, test_dataset, base_dataset.data_size, calendar_feature_names, split_info


def create_data_loader(dataset, batch_size=64, shuffle=False):
    """Wrap a lazy dataset in a DataLoader."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available()
    )


def save_artifacts(model, device, loss_name, final_loss, best_val_loss, test_loss, data_size, calendar_feature_names, split_info):
    """Persist model weights and minimal training metadata."""

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'loss_name': loss_name,
        'data_size': data_size,
        'noise_size': 8,
        'gru_hidden_size': 64,
        'output_size': 1,
        'fc_hidden_sizes': FC_HIDDEN_SIZES,
        'number_generations_per_forward_call': 20,
    }
    torch.save(checkpoint, MODEL_FILE)

    metadata = {
        'checkpoint_path': str(MODEL_FILE),
        'loss_name': loss_name,
        'device': str(device),
        'quick_run': QUICK_RUN,
        'max_products': MAX_PRODUCTS,
        'max_day_columns': MAX_DAY_COLUMNS,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'use_calendar_features': USE_CALENDAR_FEATURES,
        'calendar_feature_set': CALENDAR_FEATURE_SET,
        'calendar_feature_names': calendar_feature_names,
        'data_size': data_size,
        'final_epoch_loss': final_loss,
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'loss_plot_path': str(LOSS_PLOT_FILE),
        'split_strategy': 'temporal_windows',
        'split_ratios': {
            'train': TRAIN_SPLIT,
            'val': VAL_SPLIT,
            'test': TEST_SPLIT,
        },
        'split_windows': split_info,
    }
    with METADATA_FILE.open('w', encoding='utf-8') as file_handle:
        json.dump(metadata, file_handle, indent=2)


def save_loss_plot(train_losses, val_losses, output_path):
    """Plot training and validation losses across epochs and save as a PNG."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker='o', label='Train loss')
    plt.plot(epochs, val_losses, marker='o', label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def evaluate_model(model, data_loader, device, loss_name):
    """Compute average loss for a data split without gradient updates."""

    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for context_batch, target_batch in data_loader:
            context_batch = context_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            ensemble_preds = model(context_batch)
            loss = compute_loss(ensemble_preds, target_batch, loss_name)
            batch_size = context_batch.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size
    model.train()

    if total_count == 0:
        raise ValueError('DataLoader has no samples for evaluation.')

    return total_loss / total_count


def train_model(train_loader, val_loader, test_loader, data_size, calendar_feature_names, split_info, loss_name=LOSS_NAME):
    """Create NN model and minimal training setup."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    noise_size = 8 # Size of the noise vector for the generative model, can be chaned to see how it affects the results
    learning_rate = 1e-3
    epochs = EPOCHS
    
    net = createGenerativeGRUNN(
        data_size=data_size,
        gru_hidden_size=64,
        noise_size=noise_size,
        output_size=1,
        hidden_sizes=FC_HIDDEN_SIZES,
    )()
    
    model = ConditionalGenerativeModel(
        net=net,
        size_auxiliary_variable=noise_size,
        number_generations_per_forward_call=20
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    batches_per_epoch = len(train_loader)
    total_batches = batches_per_epoch * epochs
    training_start_time = time.time()

    print(f"Model ready on {device}. Batches per epoch: {batches_per_epoch}")
    print(f"Training setup - epochs: {epochs}, learning_rate: {learning_rate}, loss: {loss_name}, data_size: {data_size}")

    average_epoch_loss = None
    best_val_loss = float('inf')
    best_state_dict = None
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_index, (context_batch, target_batch) in enumerate(train_loader, start=1):
            context_batch = context_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            ensemble_preds = model(context_batch)
            loss = compute_loss(ensemble_preds, target_batch, loss_name)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * context_batch.size(0)

            #Adding this so I can see how roughly the training will take lol
            overall_batch_index = epoch * batches_per_epoch + batch_index
            if batch_index % PROGRESS_EVERY == 0 or batch_index == batches_per_epoch:
                elapsed_seconds = time.time() - training_start_time
                average_batch_seconds = elapsed_seconds / overall_batch_index
                remaining_batches = total_batches - overall_batch_index
                eta_seconds = average_batch_seconds * remaining_batches
                progress_percent = 100.0 * overall_batch_index / total_batches
                print(
                    f"Progress {progress_percent:.1f}% "
                    f"(epoch {epoch + 1}/{epochs}, batch {batch_index}/{batches_per_epoch}) - "
                    f"ETA: {eta_seconds / 60:.1f} min"
                )

        average_epoch_loss = epoch_loss / len(train_loader.dataset)
        val_loss = evaluate_model(model, val_loader, device, loss_name)

        train_losses.append(average_epoch_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"train_loss: {average_epoch_loss:.6f}, val_loss: {val_loss:.6f}, best_val: {best_val_loss:.6f}"
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss = evaluate_model(model, test_loader, device, loss_name)
    print(f"Test loss (best-val checkpoint): {test_loss:.6f}")

    save_loss_plot(train_losses, val_losses, LOSS_PLOT_FILE)
    print(f"Saved loss plot: {LOSS_PLOT_FILE}")

    save_artifacts(
        model,
        device,
        loss_name,
        average_epoch_loss,
        best_val_loss,
        test_loss,
        data_size,
        calendar_feature_names,
        split_info,
    )
    print(f"Saved model checkpoint: {MODEL_FILE}")
    print(f"Saved training metadata: {METADATA_FILE}")

    return model


if __name__ == '__main__':
    print(f'Calendar features enabled: {USE_CALENDAR_FEATURES}')
    print(f'Calendar path: {CALENDAR_PATH}')
    print(f'Calendar feature set: {CALENDAR_FEATURE_SET}')

    train_dataset, val_dataset, test_dataset, data_size, calendar_feature_names, split_info = load_and_preprocess()
    train_loader = create_data_loader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = create_data_loader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = create_data_loader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Model input features: {data_size}")
    print(f"Training batches: {len(train_loader)}")
    print(
        f"Temporal split windows - train: {split_info['train_windows']}, "
        f"val: {split_info['val_windows']}, test: {split_info['test_windows']} "
        f"(total: {split_info['total_windows']})"
    )
    if QUICK_RUN:
        print(f"Quick run enabled - products: {MAX_PRODUCTS}, day_columns: {max(WINDOW_SIZE + 1, MAX_DAY_COLUMNS)}, batch_size: {BATCH_SIZE}, epochs: {EPOCHS}")
    train_model(
        train_loader,
        val_loader,
        test_loader,
        data_size=data_size,
        calendar_feature_names=calendar_feature_names,
        split_info=split_info,
        loss_name=LOSS_NAME,
    )