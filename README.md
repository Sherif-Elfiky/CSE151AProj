# CSE 151A Group Project

Many students order food on DoorDash when they're in a hurry or looking for something new to eat. However, the ETA on the app is sometimes unreliable. Our goal is to accurately predict the ETA of an order based on attributes/features like the type of cuisine, the number of on-shift dashers, the time the order was placed, etc. Our dataset consists of 197k observations/orders and 16 features that include those mentioned above. We will do regression on the time it takes for the order to be complete. To these ends, we tried a variety of models ranging from linear regression to deep neural networks. Below are the results of our models, our code and methods, and our thought process.

## Introduction
Doordash is one of the most popular food delivery apps, allowing for a wide variety of choices on demand. Whether its a big catering event or whether its a really hungry individual, customers need accurate ETAs. Despite its popularity, DoorDash's ETA feature is sometimes unreliable. Motivated by this, in this project, our group decided to give ETA regression a try.

## Figures
Figures 1.1, 1.2, and 1.3 all present the data in a visual way, helping us understand the relationships between certain features. Figures 1.4 and 1.5 help us visualize the loss of our models over time.

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
![image](https://github.com/user-attachments/assets/e9ed2301-5183-4d90-83da-647edc6d1288)

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

After we fit our model, we evaluated our model on the testing set to see how well it generalized.
```
model.evaluate(X_test, y_test)
```

### Model 4 - [Milestone 4](https://github.com/Sherif-Elfiky/CSE151AProj/blob/main/151AProject.ipynb)
Our fourth model was a another dense neural network, however, we did hyperparameter tuning. 

Before performing a gridsearch, we first needed to specify our hyperparameters. In our gridsearch, we wanted to vary our models in nodes per layer, number of hidden layers, activation each layer, and the learning rate.
```
def build_model(hp):
  # our hyperparameters
  units = hp.Choice('units', values = [12, 32, 64])
  hidden_layers = hp.Int('layers', min_value = 1, max_value = 3, step = 1)
  activation = hp.Choice('activation', values = ['relu', 'leaky_relu', 'elu'])
  learning_rate = hp.Float('lr', min_value = 1e-3, max_value = 1e-2, step = 3, sampling = 'log')

  model = Sequential()

  # input layer
  model.add(Dense(units = units, activation = activation, input_dim = X_train.shape[1]))

  # tune number of hidden layers
  for i in range(hidden_layers):
    model.add(Dense(units = units, activation = activation))

  # output layer
  model.add(Dense(units = 1, activation = activation))

  model.compile(optimizer = 'adam', loss = 'mean_squared_error')

  return model
```

Then, we used `keras_tuner` to specify the settings for our gridsearch.
```
tuner = keras_tuner.GridSearch(
    hypermodel=build_model,
    objective='loss',
    seed=15,
    executions_per_trial=1,
    tune_new_entries=True,
    allow_new_entries=True,
    max_consecutive_failed_trials=3,
    overwrite=True
)
```

After a gridsearch, we found that the model that had the best loss had four hidden layers with 32 nodes per hidden layer. The activation each layer was ELU.
```
Trial 0053 summary
Hyperparameters:
units: 32
layers: 3
activation: elu
lr: 0.009000000000000001
Score: 873135.9375
```

Again, we used `Sequential` from keras for our model and each layer waas a `Dense` layer from keras. 
```
best_model = Sequential()
best_model.add(Dense(units = 32, activation = 'elu', input_dim = X_train.shape[1]))
best_model.add(Dense(units = 32, activation = 'elu'))
best_model.add(Dense(units = 32, activation = 'elu'))
best_model.add(Dense(units = 32, activation = 'elu'))
best_model.add(Dense(units = 1, activation = 'elu'))
best_model.learning_rate = 0.009
```

We didn't do a gridsearch on different optimizers nor loss functions and just kept to Adam and mean squared error.
```
best_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
```

Then, we fit our model on the training set.
```
best_history = best_model.fit(X_train, y_train, epochs = 10, verbose = 1, callbacks = [early_stopping])
```

Finally, we evaluated our model on the test set.
```
best_model.evaluate(X_test, y_test)
```

## Results
### Model 1
Our model yielded really really high loss in the 800000-900000 range for both train and test.
```
Training MSE: 871983.11
Testing MSE: 863516.33
```
### Model 2
Our model yielded similar loss to our first linear regression model for both train and test.
```
Training MSE: 871735.45
Testing MSE: 864404.64
```

### Model 3
While fitting our model, the loss per epoch was output.
```
Epoch 1/10
2194/2194 [==============================] - 5s 2ms/step - loss: 3463438.2500
Epoch 2/10
2194/2194 [==============================] - 5s 2ms/step - loss: 1002810.7500
Epoch 3/10
2194/2194 [==============================] - 7s 3ms/step - loss: 916028.3125
Epoch 4/10
2194/2194 [==============================] - 4s 2ms/step - loss: 888758.3125
Epoch 5/10
2194/2194 [==============================] - 4s 2ms/step - loss: 879416.1250
Epoch 6/10
2194/2194 [==============================] - 7s 3ms/step - loss: 875893.0625
Epoch 7/10
2194/2194 [==============================] - 6s 3ms/step - loss: 874398.8750
Epoch 8/10
2194/2194 [==============================] - 4s 2ms/step - loss: 873716.1250
Epoch 9/10
2194/2194 [==============================] - 5s 2ms/step - loss: 873246.7500
Epoch 10/10
2194/2194 [==============================] - 4s 2ms/step - loss: 872971.5000
```
Then, we plotted our loss per epoch to visualize what was happening with our loss over time. Below is [Figure 1.4](#figure-1.4).
![image](https://github.com/user-attachments/assets/e9ed2301-5183-4d90-83da-647edc6d1288)

Finally, here's the loss while we evaluated our model on the testing set.
```
549/549 [==============================] - 1s 2ms/step - loss: 865206.4375
865206.4375
```
### Model 4
These are the top three models with the best loss performance after the gridsearch/
```
Results summary
Results in ./untitled_project
Showing 3 best trials
Objective(name="loss", direction="min")

Trial 0053 summary
Hyperparameters:
units: 32
layers: 3
activation: elu
lr: 0.009000000000000001
Score: 873135.9375

Trial 0063 summary
Hyperparameters:
units: 64
layers: 2
activation: relu
lr: 0.001
Score: 873173.5

Trial 0071 summary
Hyperparameters:
units: 64
layers: 2
activation: elu
lr: 0.009000000000000001
Score: 873274.8125
```

Of course, we took the best model `Trial 0053`. When we fit our model, we were able to output the loss per epoch. It is important to note that this model ran on the same early stopping settings, iterating through only seven epochs before stopping.
```
Epoch 1/10
2194/2194 [==============================] - 17s 7ms/step - loss: 1391516.0000
Epoch 2/10
2194/2194 [==============================] - 13s 6ms/step - loss: 879407.1250
Epoch 3/10
2194/2194 [==============================] - 5s 2ms/step - loss: 875330.9375
Epoch 4/10
2194/2194 [==============================] - 7s 3ms/step - loss: 874432.1875
Epoch 5/10
2194/2194 [==============================] - 5s 2ms/step - loss: 874171.3125
Epoch 6/10
2194/2194 [==============================] - 6s 3ms/step - loss: 874147.3750
Epoch 7/10
2194/2194 [==============================] - 6s 3ms/step - loss: 874235.1250
```

Then, we plotted our loss per epoch to visualize what was happening with our loss over time. Below is [Figure 1.5](#figure-1.5).
![image](https://github.com/user-attachments/assets/812d3757-7ff5-41a0-b8d7-5550b65f57e0)

Finally, here's the loss while we evaluated our model on the testing set.
```
549/549 [==============================] - 1s 2ms/step - loss: 865334.6875
865334.6875
```

## Discussion
### Data Exploration and Preprocessing
Initially, we were tempted to do a regression on `actual_delivery_time` since its a timestamp in UTC, matching the ETA DoorDash would provide. However, we realized that this wouldn't be very sustainable for not visualizing our data. Additionally, instead of being a simpler regression where a model outputs a single numerical value, the problem would be more complex outputting a specific timestamp. 

So, we decided to create new feature `time_to_deliver` which represents the amount of seconds the order takes. To calculate this, we would need to subtract `created_at` from `actual_delivery_time` and then convert the resulting timestamp into a float.

After that, we were able to visualize our data via a pairplot [Figure 1.1](#figure-1.1). Evaluating our data, we saw many outliers towards the larger side. This is likely due to extra large orders or special conditions that might affect some of our features. For example, if the dasher encountered a traffic accident, `estimated_store_to_consumer_driving_duration` would be abnormally high. So, to ensure we were able to regress on average cases, we got rid of abnormally high outliers. We decided when to cut our outliers off by eye.

Additionally, we noticed that our categorical features wouldn't really affect ETA. Features like `market_id` and `store_id` are irrelevant to our goal. Although its true that some restuarants might take longer than others because of how they personally operate, we are limited by the fact that we wouldn't know which restaurants have a certain `store_id`. Other features like `order_protocol` and `estimated_order_place_duration` do have some impact, however, very very small. For instance, `estimated_order_place_duration` represents how long the store receives the order from DoorDash, which happens to not take long at all.

In the end, we were left with our numerical features. From [Figure 1.1](#figure-1.1), we saw that some categories of `store_primary_category` were normally distributed but skewed from the massive outliers. However, this was not respresentative for all features. Therefore, we chose not to standardize our data and instead chose to min-max scale our data to make computation faster. 

### Model 1
After we preprocessed our data, we visualized our data again with another pairplot [Figure 1.2](#figure-1.2). Unable to find any clear correlations between the features and `time_to_deliver`, we generated a heatmap [Figure 1.3](#figure-1.3) plotting all the correlations between our features to see the numerical impact. Again, we weren't able to see any strong correlations with any of the features with `time_to_deliver`. 

However, we chose to do a linear regression as a baseline model. Out of all the features, `estimated_store_to_consumer_driving_duration` had the greatest correlation with `time_to_deliver`. Therefore, we chose `estimated_store_to_consumer_driving_duration` as our feature to predict `time_to_deliver`. 

We ended up with a smaller testing MSE than training MSE, implying that our model underfit. This makes sense because according to [Figure 1.2](#figure-1.2), there isn't really a clear defined pattern between the two features. In other words, the data seems too scattered for a linear model to coherently fit it. Since we underfit, for our next model, we planned for it to be more complex in order to better fit the data. Additionally, since there weren't any clear patterns between any specific feature and `time_to_deliver`, we also thought that a model that would take in all of the features to predict `time_to_deliver` would have more information to work with to accurately predict.

### Model 2
Our first instinct was to try a linear regression but using all other features to predict `time_to_deliver`. 

For this model, we again used mean squared error for our loss and `LinearRegression` from scikit-learn, which uses OLS to optimize our weights. 

Our mean squared error for our train and our test sets did have similar values to those of the first model, falling into the 800000-900000 range. And again, our model produced greater training error than test, implying underfitting. 

We did notice that this model performed better on the training set and worse on the test set than the previous, implying Model 2 is more fitted to the data than Model 1. This makes sense because we did increase the complexity for Model 2, incorporating all features this time.

We thought that maybe this was a step in the right direction and thus decided our next model be something more complex.

### Model 3
For our third model, we chose a deep neural network. Since we didn't recognize any of the patterns in the data at all, we thought that maybe a neural network would be helpful to find some of the more complex patterns we couldn't decipher. 

We used this model as a baseline model for our fourth since we didn't know how complex we needed the neural network needed to be. We arbitrarily chose an architechture consisting of two hidden layers and 32 nodes per hidden layer. For each layer, from experience and research, we found that ReLU generally performed the best and thus used it for each layer. To optimize speed while also prioritizing performance, we chose the Adam optimizer. 

Again, since we didn't know how the model would perform, to be safe, we incorporated early stopping to help prevent potential overfitting. Most of the settings are the same as the default settings. Then, we fit our data with 10 epochs as a baseline. 

We saw that our loss quickly converged to a similar value to those of the other two models after a single epoch. After we fit our data, we evaluated our model on the testing set and reported the loss. The testing loss was greater than our training loss, implying towards overfitting. However, its worth to note that there is a lot of randomness in the data and that the optimizer might have trouble actually finding better local minima as a result.

Taking a look at our predictions, some are within five or so minutes of the ground truth, however, some are almost an hour. As this was a baseline deep neural network, we thought maybe our results would be different with some hyperparameter tuning.

### Model 4
For our hyperparameter tuning, we ran a gridsearch with settings that were specialized to time. For example, we would only do one execution per trial. Additionally, we would only run three epochs per trial, however, its somewhat justified given the previous model's very early convergence. 

For the hyperparameters we wanted to tune, we chose the number of hidden layers, nodes per layer, activation function, and learning rate. We wanted our number of hidden layers to be between two and four given and our number of nodes per layer to between 12 and 64 given how our previous model could have potentially shown signs of overfitting. Wanting to keep a ReLU type of activation function, we experimented ELU and Leaky ReLU since they were modified versions. 

Running only three epochs was surprisingly fine since all the models seemed to converge early during the gridsearch. In the end, the model with the best performance was a deep neural network with four hidden layers, 32 nodes per layer, ELU for each activation, and a learning rate of 0.009.

Our model had similar results to the baseline deep neural network, achieving similar mean squared error and also performing better on train than test. Given how this model was relatively much more complex than the baseline given the greater number of hidden layers and performed almost identical to the baseline, we reasoned that maybe the patterns in our loss might just be attributed to the randomness of the data. 

## Conclusion
In the end, with our different models and methods even after pruning out our very extreme outliers, we couldn't seem to achieve good mean squared error. Our predictions ended up being very off, sometimes being almost up to an hour off. 

We believe that maybe we couldn't accurately predict ETA given how many important factors we were missing. For example, the weather matters, the restaurant itself matters since it may have different procedures, maybe how crowded the restaurant is, etc. Therefore, even though these features on paper seem like the main deciders of what the ETA would be, any of the factors that were missing could change the ETA entirely. 

Furthermore, even without taking these factors into account, our methods probably needed even greater complexity to predict the ETA. One idea that comes to mind is using ensemble methods that also take our potentially signficant categorical features that may affect the ETA into account like `store_primary_category`. Maybe some other methods that are more advanced that we haven't considered like transformers may be able to perform better.

However, we wish that maybe we could've chosen a better dataset to do our project on given how dynamic DoorDash ETA can be. After some research on DoorDash's actual ETA predictor, they incorporate many components that we lack, utilizing complex probabilistic models to aide their data preprocessing and model performance.

## Statement of Collaboration
Eric Wang - Project Manager, Worked on data preprocessing, Model 3, Model 4, and the writeup.

Sherif Elfiky - Data preprocessing, Model 1, reviewed everything

Yudong Chen - Model 3, Model 4, reviewed everything

Alexander Yang - Model 1, Model 2, reviewed everything

Tianlin Zhao - Model 1, Model 2, reviewed everything
