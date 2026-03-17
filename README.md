# MLForecasting

## Setup

Create and activate a virtual environment, then install dependencies:

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch pandas numpy einops torchtyping
```

## for linux
python3 -m venv .venv
source .venv/bin/activate
pip install torch pandas numpy einops torchtyping
python train.py


## Running the training script

Place `sales_train_validation.csv` in `Forecasting/src/`, then run:

```
cd Forecasting/src
python train.py
```