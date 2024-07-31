# CSE 151A Group Project

## Abstract

## Introduction
Doordash is one of the most popular food delivery apps, allowing for a wide variety of choices on demand. Whether its a big catering event or whether its a really hungry individual, customers need accurate ETAs. Despite its popularity, DoorDash's ETA feature is sometimes unreliable. Motivated by this, in this project, our group decided to give ETA regression a try.

## Figures
Figures 1.1, 1.2, and 1.3 all present the data in a visual way, helping us understand the relationships between certain features.

### Figure 1.1:
Pairplot showing the relationship between individual features with each other. The diagonal represents the distribution for each feature. Each observation is color-coded by `store_primary_category` - the type of food or service the resturant provides (e.g. Italian, Chinese, Fast Food)
![image](https://github.com/user-attachments/assets/61f78eea-e4df-4488-80af-efdd535a857c)

### Figure 1.2:
Pairplot after Data Preprocessing. Shows relationships between features and the distribution for each feature after removing outliers and normalization. 
![image](https://github.com/user-attachments/assets/11343030-fff4-494e-baa4-7da7640aab58)

### Figure 1.3
Heatmap showing the correlation between each feature. Greater values imply greater correlation.
![image](https://github.com/user-attachments/assets/6240da21-5b30-443e-902e-91c6e3a46454)

### Figure 1.4
Epoch vs. Training Loss plot for our baseline deep neural network.
![image](https://github.com/user-attachments/assets/812d3757-7ff5-41a0-b8d7-5550b65f57e0)

### Figure 1.5
Epoch vs. Training Loss plot for our deep neural network with optimized hyperparameters.

## Methods
Here are all the steps we took in our attempt of ETA regression. Here is the [dataset](https://www.kaggle.com/datasets/dharun4772/doordash-eta-prediction) we used.

### Data Exploration and Preprocessing - [Milestone 2](https://github.com/Sherif-Elfiky/CSE151AProj/tree/milestone2/151AProject.ipynb)
The data has 197428 observations and 16 features. Features `created_at` and `actual_delivery_time` are timestamps for when the order was made and when the order was completed. Initially, these were `string` values, but we converted them to `datetime`.
```
data['created_at'] = pd.to_datetime(data['created_at'])
data['actual_delivery_time'] = pd.to_datetime(data['actual_delivery_time'])
```

Then, since we wanted to do a regression on the duration of the order, we created a new feature `time_to_deliver` to represent this in seconds. 
```
data['time_to_deliver'] = (pd.to_datetime(data['actual_delivery_time']) - pd.to_datetime(data['created_at'])).dt.total_seconds()
```

Many observations were missing `store_primary_category`, looking at the dataset on kaggle, we realized that restaurants with category N/A were converted to `NaN`. So, we replaced these `NaN` values with N/A.
```
data.fillna({'store_primary_category': 'N/A'})
```

Our data had very extreme outliers towards the greater end for a most of our numerical features. So, we dropped those outliers. See [Figure 1.1](#figure-1.1) for pre-dropped data.
```
data = data[data['time_to_deliver'] <= 6000]
data = data[data['total_items'] <= 8]
data = data[data['subtotal'] < 10000]
data = data[data['max_item_price'] <= 2500]
data = data[data['total_onshift_dashers'] < 90]
data = data[data['total_busy_dashers'] < 90]
data = data[data['total_outstanding_orders'] <= 100]
```

We also dropped our categorical features since they were mostly irrelevant to predicting `time_to_deliver`. Its important to note that `estimated_order_place_duration` which represents how long it takes for the restaurant to receive the order is indeed categorical despite its description implying continuity. Anyways, it would not have much impact on `time_to_deliver` since the process happens in milliseconds. 
```
dropped_columns = ['created_at', 'actual_delivery_time', 'store_primary_category', 'market_id', 'store_id', 'order_protocol', 'estimated_order_place_duration']
numerical_data = data.drop(dropped_columns, axis=1)
```

With just numerical features left, we min-max scaled our data except for `time_to_deliver`. See [Figure 1.2](#figure-1.2) for data after preprocessing.
```
mms = MinMaxScaler()
numerical_data = pd.DataFrame(mms.fit_transform(numerical_data), columns = numerical_data.columns)
numerical_data['time_to_deliver'] = data['time_to_deliver']
```

### Model 1 - [Milestone 3](https://github.com/Sherif-Elfiky/CSE151AProj/tree/milestone3/151AProject.ipynb)
We began with a linear regression model using a single feature to predict `time_to_deliver` as a baseline model. After plotting the correlations between each feature and `time_to_deliver` on a heatmap (see [Figure 1.3](#figure-1.3)), we decided to use `estimated_store_to_consumer_driving_duration` to predict `time_to_deliver`.

So, we made our X `estimated_store_to_consumer_driving_duration` and our y `time_to_deliver`. 
```
# we use estimated_store_to_consumer_driving_duration to predict time_to_deliver
X = numerical_data['estimated_store_to_consumer_driving_duration']

# we want to predict time_to_deliver
y = numerical_data['time_to_deliver']
```

Then, we split our train/test 80:20.
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

Then, we used `LinearRegression` from sci-kit learn to generate our model. We then fit it on our training set.
```
reg = LinearRegression()
regmodel = reg.fit(X_train.values.reshape(-1,1), y_train.values.reshape(-1,1))
```

We used mean squared error as our loss function. To calculate mean squared error, we generated predictions for both train/test.
```
# get training predictions
yhat_train = regmodel.predict(X_train.values.reshape(-1,1))
# get testing predictions
yhat_test = regmodel.predict(X_test.values.reshape(-1,1))
```

We were then able to see our mean squared error to evaluate how good our model did using the predictions we calculated.
```
print('Training MSE: %.2f' % mean_squared_error(y_train, yhat_train))
print('\nTesting MSE: %.2f' % mean_squared_error(y_test, yhat_test))
```

### Model 2 - [Milestone 3](https://github.com/Sherif-Elfiky/CSE151AProj/tree/milestone3/151AProject.ipynb)
We then tried a more complex linear regression, using all other features besides `time_to_deliver` instead of just `estimated_store_to_consumer_driving_duration` to predict `time_to_deliver`. Accordingly, we had to redefine X to include all our features. We also kept the same 80:20 train/test split.
```
X = numerical_data.drop(['time_to_deliver'], axis=1)
y = numerical_data['time_to_deliver']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

From there, our code basically looked identical to the previous model. First, we defined our model using sci-kit learn's `LinearRegression` and fit it on the training set.
```
reg2 = LinearRegression()
regmodel2 = reg2.fit(X_train, y_train)
```

Then, we generated our predictions for both the training and test sets to calculate our mean squared error for both sets.
```
yhat_train = regmodel2.predict(X_train)
yhat_test = regmodel2.predict(X_test)
```

Finally, we reported our mean squared error.
```
print('Training MSE: %.2f' % mean_squared_error(y_train, yhat_train))
print('\nTesting MSE: %.2f' % mean_squared_error(y_test, yhat_test))
```

### Model 3 - [Milestone 4](https://github.com/Sherif-Elfiky/CSE151AProj/blob/main/151AProject.ipynb)
Our third model was a dense neural network. This model served as a baseline neural network to compare with for our fourth. We used `Sequential` from keras for our model and each layer waas a `Dense` layer from keras. 

For our baseline, we had two hidden layers with 32 nodes per hidden layer. Our activation for each layer was ReLU.
```
model = Sequential()
model.add(Dense(units = 32, activation = 'relu', input_dim = X_train.shape[1]))
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dense(units = 1, activation = 'relu'))
```

Our optimizer was Adam and we again used mean squared error for our loss function.
```
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
```

Before we fit the model, we used early stopping settings to stop potential overfitting.
```
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    min_delta=0,
    patience=1,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0
)
```

We then fit our model using 10 epochs and our aforementioned early stopping settings.
```
history = model.fit(X_train, y_train, epochs = 10, verbose = 1, callbacks = [early_stopping])
```




### Model 4 - [Milestone 4](https://github.com/Sherif-Elfiky/CSE151AProj/blob/main/151AProject.ipynb)

## Results

## Discussion

## Conclusion

## Statement of Collaboration
