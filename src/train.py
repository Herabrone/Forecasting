import sys
import os
import json
import time
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datapreprocessing import process, WINDOW_SIZE
from src.NN import createGenerativeGRUNN, ConditionalGenerativeModel, ensemble_nll_loss
from src.scoringrules import energy, kernel, energy_kernel


DATA_PATH = Path(__file__).resolve().parent / 'sales_train_validation.csv'
LOSS_NAME = 'ensemble_nll'  # Training objective to optimize. 'ensemble_nll' is usually the fastest and most stable.
                             # Other options ('energy', 'kernel', 'energy_kernel') are valid but can train slower.

QUICK_RUN = True  # If True, train on a subset for faster iteration. If False, use all products and all day columns.

MAX_PRODUCTS = 1000  # Number of product time series to keep when QUICK_RUN=True.
MAX_DAY_COLUMNS = 365  # Number of most recent day columns to keep when QUICK_RUN=True.
                       # Must be >= WINDOW_SIZE + 1 to form at least one input-target pair.

BATCH_SIZE = 1024  # Samples per optimizer step.
                   # Larger batches usually increase throughput on GPU but require more VRAM.
                   # If you hit CUDA OOM, lower this first (for example: 512, then 256).

EPOCHS = 3  # Full passes over the selected training subset.
            # More epochs can improve fit but increase runtime linearly.

PROGRESS_EVERY = 50  # Print progress/ETA every N batches to show how close training is to completion.

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / 'artifacts'
MODEL_FILE = ARTIFACTS_DIR / 'model_checkpoint.pt'
METADATA_FILE = ARTIFACTS_DIR / 'model_metadata.json'


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


def load_and_preprocess():
    """Load data from CSV, call process() from datapreprocessing, return X, y tensors ready for NN."""
    
    #This is just pulled from the datapreprocessing, ill clean this up later 
    dataset = pd.read_csv(DATA_PATH)

    if QUICK_RUN:
        dataset = dataset.head(MAX_PRODUCTS)
    
    metadata_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    days_cols = [col for col in dataset.columns if col not in metadata_cols]
    days = dataset[days_cols]

    if QUICK_RUN:
        days_to_keep = max(WINDOW_SIZE + 1, MAX_DAY_COLUMNS)
        days = days.iloc[:, -days_to_keep:]
    
    X, y = process(days)
    
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()
    
    return X_tensor, y_tensor


def create_train_loader(X, y, batch_size=64):
    """Wrap the full training tensors in a shuffled DataLoader."""

    dataset = TensorDataset(X, y)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )


def save_artifacts(model, device, loss_name, final_loss):
    """Persist model weights and minimal training metadata."""

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'loss_name': loss_name,
        'noise_size': 8,
        'gru_hidden_size': 64,
        'output_size': 1,
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
        'final_epoch_loss': final_loss,
    }
    with METADATA_FILE.open('w', encoding='utf-8') as file_handle:
        json.dump(metadata, file_handle, indent=2)


def train_model(train_loader, loss_name=LOSS_NAME):
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
        data_size=1,
        gru_hidden_size=64,
        noise_size=noise_size,
        output_size=1
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
    print(f"Training setup - epochs: {epochs}, learning_rate: {learning_rate}, loss: {loss_name}")

    average_epoch_loss = None

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
        print(f"Epoch {epoch + 1}/{epochs} - loss: {average_epoch_loss:.6f}")

    save_artifacts(model, device, loss_name, average_epoch_loss)
    print(f"Saved model checkpoint: {MODEL_FILE}")
    print(f"Saved training metadata: {METADATA_FILE}")

    return model


if __name__ == '__main__':
    X, y = load_and_preprocess()
    train_loader = create_train_loader(X, y, batch_size=BATCH_SIZE)
    print(f"Data shapes - X: {X.shape}, y: {y.shape}")
    print(f"Training batches: {len(train_loader)}")
    if QUICK_RUN:
        print(f"Quick run enabled - products: {MAX_PRODUCTS}, day_columns: {max(WINDOW_SIZE + 1, MAX_DAY_COLUMNS)}, batch_size: {BATCH_SIZE}, epochs: {EPOCHS}")
    train_model(train_loader, loss_name=LOSS_NAME)