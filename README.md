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

Our data had very extreme outliers towards the greater end for a most of our numerical features. So, we dropped those outliers. 
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

With just numerical features left, we min-max scaled our data except for `time_to_deliver`
```
mms = MinMaxScaler()
numerical_data = pd.DataFrame(mms.fit_transform(numerical_data), columns = numerical_data.columns)
numerical_data['time_to_deliver'] = data['time_to_deliver']
```

### Model 1 - [Milestone 3](https://github.com/Sherif-Elfiky/CSE151AProj/tree/milestone3/151AProject.ipynb)


### Model 2 - [Milestone 3](https://github.com/Sherif-Elfiky/CSE151AProj/tree/milestone3/151AProject.ipynb)

### Model 3 - [Milestone 4](https://github.com/Sherif-Elfiky/CSE151AProj/blob/main/151AProject.ipynb)

### Model 4 - [Milestone 4](https://github.com/Sherif-Elfiky/CSE151AProj/blob/main/151AProject.ipynb)

## Results

## Discussion

## Conclusion

## Statement of Collaboration
