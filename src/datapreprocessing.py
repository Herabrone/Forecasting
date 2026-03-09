# Notes:

# Need to do 'pip install pandas' if not installed

# I have read a lot of conflicting things but it seems that energy will work on our data just fine, but kernel (and by extension enery kernel) is meant for continuous values
# So, we can use kernel and turn our data into continuous values when we do, and if the results are bad it can just be one of the scoring rules we experimented with
# I also think we can use more scoring rules and it wouldn't count as changing the model?
# Also, I thought for continuous values we would need to scale them but I think literally just turning them into floats suffices

# For our methodology, we have 2 files: sales_train_validation and sales_train_evaluation
# The evaluation file is identical except it has 28 more days in the time series. In the competition, these 28 days were forecasted and the the evaluation file was used to judge
# how accurate they were, so a possible goal for us is trying to do the same thing: training based on sales_train_validation and then comparing our results to sales_train_evaluation
# using different scoring rules
# However, in the paper, they forecast 1 time step ahead which would be 1 day ahead for us, but if we try to forecast more days that shouldn't be an issue I think?
# Another thing is in the paper they do 60% training set, 20% validation, 20% test set, another thing to look into

# Sliding window: What this is is for training, you take k values and the target is the k + 1th value in the time step, you try to forecast the target based on the k values
# In the paper, their window size is k = 10, which I have set it to here but we can experiment with it, there may be a different value that works better for us
# This program gets a bunch of pairs in the form of 2 arrays, where each pair is k = 10 values, and their target
# So eg if we had [1 2 3 4 5 6 7 8 9 10 11 12] then the pairs would be ([1 2 3 4 5 6 7 8 9 10], 11) and ([2 3 4 5 6 7 8 9 10 11], 12)

import pandas as pd
import numpy as np

WINDOW_SIZE = 10

# Process data: get the timer series, convert to floats, build sliding windows for NN input
def process(days):

    # Convert to continuous
    days_continuous = days.astype(np.float32).to_numpy()  # [num_products, num_days]

    # Build sliding windows
    windows, targets = sliding_window(days_continuous)    # windows: [num_windows, num_products, WINDOW_SIZE]

    # Reshape to per-sample layout: each (product, window) pair becomes one sample
    X = np.transpose(windows, (1, 0, 2)).reshape(-1, WINDOW_SIZE, 1).astype(np.float32)  # [num_samples, WINDOW_SIZE, 1]
    y = np.transpose(targets, (1, 0)).reshape(-1, 1).astype(np.float32)   #This is predicting the immidiate next step [num_samples, num_steps_to_predict], would be interesting to see how it would work for multiple days in the future

    return X, y

# Sliding window: Create an array of windows and an array of their corresponding targets
def sliding_window(days):
    num_products, num_days = days.shape
    num_windows = num_days - WINDOW_SIZE

    windows = np.array([days[:, i:i + WINDOW_SIZE] for i in range(num_windows)]) # For every product get the windows of size k
    targets = np.array([days[:, i + WINDOW_SIZE] for i in range(num_windows)]) # For everty product get the targets

    return windows, targets

if __name__ == '__main__':

    # Full dataset
    dataset = pd.read_csv('sales_train_validation.csv')

    metadata_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    days_cols = [col for col in dataset.columns if col not in metadata_cols]

    metadata = dataset[metadata_cols] # Additional data
    days = dataset[days_cols] # Time series

    # test_arr = np.array([
    #     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    #     [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    # ])

    days_numpy = days.to_numpy() # Convert to numpy

    # Create sliding window pairs (THIS WILL USE A LOT OF MEMORY OMG)
    windows, targets = sliding_window(days_numpy)
    print(windows.shape)
    print(targets.shape)

    X, y = process(days) # Process the data
    print(X.shape)
    print(y.shape)