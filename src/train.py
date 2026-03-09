import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datapreprocessing import process, sliding_window, WINDOW_SIZE
from src.NN import createGenerativeGRUNN, ConditionalGenerativeModel


def load_and_preprocess():
    """Load data from CSV, call process() from datapreprocessing, return X, y tensors ready for NN."""
    
    #This is just pulled from the datapreprocessing, ill clean this up later 
    dataset = pd.read_csv('./sales_train_validation.csv')
    
    metadata_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    days_cols = [col for col in dataset.columns if col not in metadata_cols]
    days = dataset[days_cols]
    
    X, y = process(days)
    
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()
    
    return X_tensor, y_tensor


def train_model(X, y):
    """Create NN model, create ConditionalGenerativeModel wrapper, initialize."""
    noise_size = 8 # Size of the noise vector for the generative model, can be chaned to see how it affects the results
    
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
    )
    
    # Test forward pass, this should be where we do the training loop (I just want to ensure the model works first lol)
    context_batch = X[:64]
    ensemble_preds = model(context_batch)
    print(f"Model initialized. Ensemble output shape: {ensemble_preds.shape}")


if __name__ == '__main__':
    X, y = load_and_preprocess()
    print(f"Data shapes - X: {X.shape}, y: {y.shape}")
    train_model(X, y)