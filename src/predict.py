"""Predict one chosen product and print predicted vs actual next-day sales."""

import pandas as pd
import torch

from NN import createGenerativeGRUNN, ConditionalGenerativeModel
from datapreprocessing import WINDOW_SIZE, normalize_context_window
from train import DATA_PATH, MODEL_FILE, QUICK_RUN, MAX_DAY_COLUMNS

TARGET_PRODUCT_ID = 'HOBBIES_1_008_CA_1_validation'


def load_product_latest_window(product_id):
    metadata_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

    dataset = pd.read_csv(DATA_PATH)
    product_row = dataset.loc[dataset['id'] == product_id]
    if product_row.empty:
        raise ValueError(f'Product id not found: {product_id}')

    product_row = product_row.iloc[0]
    days_cols = [column for column in dataset.columns if column not in metadata_cols]
    if QUICK_RUN:
        days_to_keep = max(WINDOW_SIZE + 1, MAX_DAY_COLUMNS)
        days_cols = days_cols[-days_to_keep:]

    series = product_row[days_cols].astype(float).to_numpy()
    if len(series) < WINDOW_SIZE + 1:
        raise ValueError('Not enough history to create a prediction window.')

    context_values = series[-(WINDOW_SIZE + 1):-1]
    actual_next = float(series[-1])
    normalized_context, context_mean, context_std = normalize_context_window(context_values)

    context_tensor = torch.tensor(normalized_context, dtype=torch.float32).view(1, WINDOW_SIZE, 1)
    return context_tensor, actual_next, context_mean, context_std


def load_model(checkpoint_path=MODEL_FILE, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    fc_hidden_sizes = checkpoint.get('fc_hidden_sizes')
    data_size = checkpoint.get('data_size', 1)

    if data_size != 1:
        raise NotImplementedError(
            'This predict script currently supports sales-only checkpoints (data_size=1). '
            'Train with USE_CALENDAR_FEATURES=False or extend predict.py to build calendar-aware inputs.'
        )

    net = createGenerativeGRUNN(
        data_size=data_size,
        gru_hidden_size=checkpoint['gru_hidden_size'],
        noise_size=checkpoint['noise_size'],
        output_size=checkpoint['output_size'],
        hidden_sizes=fc_hidden_sizes,
    )()

    model = ConditionalGenerativeModel(
        net=net,
        size_auxiliary_variable=checkpoint['noise_size'],
        number_generations_per_forward_call=checkpoint['number_generations_per_forward_call']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def main():
    device = 'cpu'
    model = load_model(device=device)

    context, target, context_mean, context_std = load_product_latest_window(TARGET_PRODUCT_ID)
    context = context.to(device)

    with torch.no_grad():
        ensemble_preds = model(context)

    preds = ensemble_preds.squeeze(0).squeeze(-1) * context_std + context_mean
    pred_mean = preds.mean().item()
    rounded_forecast = max(0, round(pred_mean))

    print(f'Loaded checkpoint: {MODEL_FILE}')
    print(f'Product id: {TARGET_PRODUCT_ID}')
    print(f'Actual next-day sales: {target:.4f}')
    print(f'Predicted mean sales: {pred_mean:.4f}')
    print(f'Rounded forecast (units): {rounded_forecast}')
    print(f'Ensemble draws: {[round(value, 4) for value in preds.tolist()]}')


if __name__ == '__main__':
    main()
