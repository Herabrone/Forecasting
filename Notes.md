Notes about what were working on etc.

# Darian

# Feb 26
We can look into how changing the way we predict the future can affect the final outcome. For ecxample, currently we are only predicting the immediate next day (or timestep), but what if we wanted to check on time step t + 30. I was thinking that there are two ways we can approach this, one we iterate one timestep at a time for the 30 days, taking each prediction as empirical that will be used for the following time step. Or we could also try having the network generate a sequence of X days into the future. It would be interesting to see what difference that makes.

# March 16
We may also could see how changing the noise affect the results as well

Initail training results:
Mode: quick
Horizon: 1 day
Origins evaluated: 355
------------------------------------------------------------------------
Overall MAE : 1.0170
Overall RMSE: 2.1009
Overall R2  : 0.3727
------------------------------------------------------------------------
Per-origin metric means
MAE mean:  1.0170
RMSE mean: 2.0723
R2 mean:   0.3527

Currenlty our model isnt performing as well, we are consistenly off by 1 product per day (MAE) and our model does not account for a large poriton of the variation (R2), I think this has to do with our data not being normalized, we have a lot of 0s and that means its hard to see products with higher selling rates. We dont account for those high predicitons either since we have a RMSE thats twice as our MAE.

I think we can also try increasing he window size from 10 to 28 (a bit of a early ablasion study) cuz i supect there is some 7 day cycle we might be missing with only 10 days.

# March 18

After applying the changes to the window size and normalizing the data set per product we are able to get an R2 of ~0.67 (Closed the tmux session with the actual output but believe me it is!) that still isnt neccesarily as good as I would have liked for the model though. 

I am starting to believe that we have several latent variables that is hindering the amount of variation that our model is able to represent. Looking back at the M5 forecasting dataset I think we can somwhat easily add in the calendar.csv covariates. I think that it should inherently capture things like seasnality and holidays and give more important infoartion to the model to learn. 

(Also small note that we should also see about how we would propagate a prediction to more than just the next day, as this would look pretty interesting on a graph if we had our predictions vs reality per month)

(Another small note I just thought of is if we could push that further and also compare to a naive classical statical approach and see if our method is statically sigificant for forecasting next months sales)

I also realize up to now I havent been doing to 60/20/20 split. 

# March 19
After Switching to the 60/20/20 split and running for a longer 20 epochs i got this

Epoch 20/20 - train_loss: -0.188688, val_loss: 0.118695, best_val: 0.118335

The results from the with 20% validation subset is:
Backtest summary
------------------------------------------------------------------------
Mode: quick
Eval split: val
Horizon: 1 day
Origins evaluated: 67
------------------------------------------------------------------------
Overall MAE : 0.9572
Overall RMSE: 2.0719
Overall R2  : 0.4128
------------------------------------------------------------------------
Per-origin metric means
MAE mean:  0.9572
RMSE mean: 2.0353
R2 mean (raw):      0.4104
R2 median (robust): 0.4056
R2 SST-weighted:    0.4068
Low-variance origins (sst < 1e-6): 1

Which is kinda unfourtunate since our R2 has actually decreased from .67, however I think that we got the .67 due to ovverfitting and allowing the training data to see the entire data set. We should consider this our actual baseline by just using the time series data. Ill try training with the calendar csv later today...

# more about the calendar preprocessing

The preprocessing for the calendar.csv starts quite intuitively, we simple just match the days to the day in our time series data one to one (also I dropped the d_ prefix so its just numerical data).

The actual features we can capture are a little more interesting.

For days of the week and months of the year we map numerical time values (v) into angles by angle = 2pi * (v/period) with the period either being 7 for days of the week, or 12 for months. I did this becuase it better represents the cyclical nature of the temporal data. For example, Sunday is close to Monday but in the data it would look as if its 6 time steps away. Similarly for December/January. My main justicifation was this aritcle I was reading: (https://medium.com/%40indradeep.chatterjee/cyclic-encoding-in-machine-learning-a-practical-guide-with-code-and-intuition-049846419ccb)

Also have a simple binary weekend flag

There is also info for when food stamps become available, im not too convinced that this has a significant impact (maybe for some of the food items at walmart) but it is also processed as a binary flag for each state respectively.

This is the same idea for when holidays occur, just convert the categorical data to nummeric




# Rey


# Shiv