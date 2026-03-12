import sys
import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datapreprocessing import process, WINDOW_SIZE
from src.NN import createGenerativeGRUNN, ConditionalGenerativeModel, ensemble_nll_loss
from src.scoringrules import energy, kernel, energy_kernel


DATA_PATH = Path(__file__).resolve().parent / 'sales_train_validation.csv'
LOSS_NAME = 'ensemble_nll' # Options: 'ensemble_nll', 'energy', 'kernel', 'energy_kernel'. Can be changed to see how different losses affect the results
QUICK_RUN = True
MAX_PRODUCTS = 60
MAX_DAY_COLUMNS = 80
BATCH_SIZE = 512
EPOCHS = 1


def score_rule_loss(ensemble_preds, targets, score_fn):
    """Average a scoring rule across the batch."""

    preds = ensemble_preds.squeeze(-1)
    targets = targets.squeeze(-1)

    batch_losses = [score_fn(preds[i], targets[i]) for i in range(preds.size(0))]
    return torch.stack(batch_losses).mean()


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
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_model(train_loader, loss_name=LOSS_NAME):
    """Create NN model and minimal training setup."""
    device = torch.device('cpu')

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

    print(f"Model ready on {device}. Batches per epoch: {len(train_loader)}")
    print(f"Training setup - epochs: {epochs}, learning_rate: {learning_rate}, loss: {loss_name}")

    for epoch in range(epochs):
        epoch_loss = 0.0

        for context_batch, target_batch in train_loader:
            context_batch = context_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            ensemble_preds = model(context_batch)
            loss = compute_loss(ensemble_preds, target_batch, loss_name)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * context_batch.size(0)

        average_epoch_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {average_epoch_loss:.6f}")

    return model


if __name__ == '__main__':
    X, y = load_and_preprocess()
    train_loader = create_train_loader(X, y, batch_size=BATCH_SIZE)
    print(f"Data shapes - X: {X.shape}, y: {y.shape}")
    print(f"Training batches: {len(train_loader)}")
    if QUICK_RUN:
        print(f"Quick run enabled - products: {MAX_PRODUCTS}, day_columns: {max(WINDOW_SIZE + 1, MAX_DAY_COLUMNS)}, batch_size: {BATCH_SIZE}, epochs: {EPOCHS}")
    train_model(train_loader, loss_name=LOSS_NAME)