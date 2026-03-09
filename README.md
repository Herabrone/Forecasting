# MLForecasting

## Setup

Create and activate a virtual environment, then install dependencies:

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch pandas numpy einops torchtyping
```

## Running the training script

Place `sales_train_validation.csv` in `Forecasting/src/`, then run:

```
cd Forecasting/src
python train.py
```