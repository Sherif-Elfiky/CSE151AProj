Dataset: https://www.kaggle.com/datasets/dharun4772/doordash-eta-prediction
Click the link to download

**Data preprocessing:**

For the dataset we will add feature `time_to_deliver` - the time it takes for the order to be complete in a duration datetime timestamp. We will do regression on `time_to_deliver`.
To do this, we had to first convert `actual_delivery_time` and `created_at` from strings to datetimes. Then, we subtracted `created_at` from `actual_delivery_time` to get `time_to_deliver`

Then, we noticed that all our data was filled except that some observations had `NaN` values for `store_primary_category`. Checking the dataset on kaggle, these `NaN` values were supposed 
to be `N/A`. So, we replaced all `NaN` values with `N/A` for feature `store_primary_category`.

Besides that, there were no further preprocessing needed so we looked decided to do some data exploration and plot our data and the correspondance of the features.

**Milestone 3 Updates:**

We did more preprocessing, including min-max scaling our numerical data and removing outliers. After taking a look at our data, we found that there weren't any clear patterns between any numerical feature and `time_to_deliver`. However, for simplicity, we chose to start off with a linear regression model using `total_outstanding_orders` to predict `time_to_deliver`. Then, we compared our predictions to the ground truth for both training and testing and then inspecting our MSE.

**Conclusion:**

We don't think there's anything we can do to improve a linear regression model since it already performs OLS to achieve the most optimal line. The MSE for both training and testing is very high and makes sense because all of the data is cluttered around in a non-linear fashion. But since the testing MSE is less than the training MSE, the model underfits.
