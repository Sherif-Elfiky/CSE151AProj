** Analysis of attributes in Door Dash wait time **

Dataset: https://www.kaggle.com/datasets/dharun4772/doordash-eta-prediction
Click the link to download

**Data preprocessing:**

For the dataset we will add feature `time_to_deliver` - the time it takes for the order to be complete in a duration datetime timestamp. We will do regression on `time_to_deliver`.
To do this, we had to first convert `actual_delivery_time` and `created_at` from strings to datetimes. Then, we subtracted `created_at` from `actual_delivery_time` to get `time_to_deliver`

Then, we noticed that all our data was filled except that some observations had `NaN` values for `store_primary_category`. Checking the dataset on kaggle, these `NaN` values were supposed 
to be `N/A`. So, we replaced all `NaN` values with `N/A` for feature `store_primary_category`.

Besides that, there were no further preprocessing needed so we looked decided to do some data exploration and plot our data and the correspondance of the features.
