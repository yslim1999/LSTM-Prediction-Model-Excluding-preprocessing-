import pandas as pd
import numpy as np
from matplotlib import pyplot
import plotly.graph_objs as go
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from numpy import concatenate
import sys
from sklearn.metrics import mean_squared_error

pd.set_option('display.max_columns', None)
data_set = pd.read_csv("No_promo_cluster4_train.csv")
test_set = pd.read_csv("No_promo_cluster4_validation.csv")
groups = [2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14]
data_set['Date'] =  pd.to_datetime(data_set['Date'], format='%m/%d/%Y')
test_set['Date'] =  pd.to_datetime(test_set['Date'], format='%m/%d/%Y')
data_set['Date'] = (data_set['Date'] - data_set['Date'].min())  / np.timedelta64(1,'D')
test_set['Date'] =   (test_set['Date'] - test_set['Date'].min())  / np.timedelta64(1,'D')
print(data_set['Date'].dtype)
data_set = data_set.drop(['Store', 'Type_assortment', 'CompetitionDistance', 'Customers','SchoolHoliday'], axis = 1)
test_set = test_set.drop(['Store', 'Type_assortment', 'CompetitionDistance', 'Customers','SchoolHoliday'], axis = 1)
print(data_set)

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

#integer encoding
encoder = LabelEncoder()
values = data_set.values
test = test_set.values
values[:,5] = encoder.fit_transform(values[:,5])
test[:,5] = encoder.fit_transform(test[:,5])

#all data are float
values = values.astype('float32')
test = test.astype('float32')
#normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
test_scaled = scaler.transform(test)

# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
reframed.drop(reframed.columns[[6,7,9,10,11]], axis=1, inplace=True)

reframed_test = series_to_supervised(test_scaled, 1, 1)
reframed_test.drop(reframed_test.columns[[6,7,9,10,11]], axis=1, inplace=True)

# split into train and test sets
values = reframed.values
test = reframed_test.values
train_days = int(data_set.shape[0])
test_days = int(test_set.shape[0])
train = values[:train_days, :]
test = values[test_days:, :]

#split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

train_test_X = train_X
train_test_Y = train_y
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# Design LSTM network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=5, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
np.set_printoptions(threshold=sys.maxsize)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
test_X = np.delete(test_X, 2, 1)
inv_yhat = np.append(test_X, yhat, axis=1)
inv_yhat[:, [5, 2]] = inv_yhat[:, [2, 5]]
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,2]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
print(test_y.shape)
inv_y = np.delete(test_X, 2, 1)
inv_y = np.append(test_X, test_y, axis=1)
inv_y[:, [5, 2]] = inv_y[:, [2, 5]]
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,2]
print(inv_y)
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
rmspe = np.sqrt(np.mean(np.square((inv_y - inv_yhat) / inv_y)))
print('Test RMSPE: %.3f' % rmspe)

#Trainig RMSE and RMSPE
# make a prediction
yhat = model.predict(train_test_X)
np.set_printoptions(threshold=sys.maxsize)
train_test_X = train_test_X.reshape((train_test_X.shape[0], train_test_X.shape[2]))
# invert scaling for forecast
train_test_X = np.delete(train_test_X, 2, 1)
inv_yhat = np.append(train_test_X, yhat, axis=1)
inv_yhat[:, [5, 2]] = inv_yhat[:, [2, 5]]
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,2]
# invert scaling for actual
train_test_Y = train_test_Y.reshape((len(train_test_Y), 1))
inv_y = np.delete(train_test_X, 2, 1)
inv_y = np.append(train_test_X, train_test_Y, axis=1)
inv_y[:, [5, 2]] = inv_y[:, [2, 5]]
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,2]
print(inv_y)
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Train RMSE: %.3f' % rmse)
rmspe = np.sqrt(np.mean(np.square((inv_y - inv_yhat) / inv_y)))
print('Train RMSPE: %.3f' % rmspe)

data_set_thing = data_set[:-1]
data_set_thing['Sales'] = inv_yhat
data_set_thing.to_csv(r'D:\Documents\dataAnalytics\predicted_CLUSTER_4_TRAIN.csv', index = False, header=True)

fig, ax = pyplot.subplots()
dates = data_set['Date'].to_numpy()
dates = dates[:-1]
ax.set_title('Cluster 4: Predictions vs Historical')
pyplot.xlabel("Date")
pyplot.ylabel("Sales")
ax.legend(['Historical','Forecast'])
pyplot.plot(dates, inv_y,color ="green", label='Historical')
pyplot.plot(dates, inv_yhat,color ="blue", label='Forecast')
pyplot.legend()
pyplot.show()

#load test data
for_this_cluster_csv = pd.read_csv("CLUSTER_4_TEST&STORE.csv")
for_this_cluster = for_this_cluster_csv
print(data_set.columns)
for_this_cluster['Sales'] = for_this_cluster['Sales'].fillna(0)
for_this_cluster = for_this_cluster.drop(['Store', 'Customers','SchoolHoliday'], axis = 1)
for_this_cluster['Date'] =  pd.to_datetime(for_this_cluster['Date'], format='%d/%m/%Y')
for_this_cluster['Date'] = (for_this_cluster['Date'] - for_this_cluster['Date'].min())  / np.timedelta64(1,'D')
for_this_cluster = for_this_cluster[['DayOfWeek', 'Date', 'Sales','Open', 'Promo','StateHoliday']]
#add padding
for_this_cluster.loc[len(for_this_cluster)] = 0
print(for_this_cluster)

test_values = for_this_cluster.values
test_values = test_values.astype('float32')
#normalize features
scaled_1 = scaler.transform(test_values)
reframed = series_to_supervised(scaled_1, 1, 1)
reframed.drop(reframed.columns[[6,7,9,10,11]], axis=1, inplace=True)
print(reframed)

values = reframed.values
print(values)
value_X, value_y = values[:, :-1], values[:, -1]
value_X = value_X.reshape((value_X.shape[0], 1, value_X.shape[1]))
six_week_predictions = model.predict(value_X)
print(six_week_predictions)
value_X = value_X.reshape((value_X.shape[0], value_X.shape[2]))
new_X = np.delete(value_X, 2, 1)
new_X = np.append(new_X, six_week_predictions, axis=1)
new_X[:, [5, 2]] = new_X[:, [2, 5]]
print(new_X)
new_X_1 = scaler.inverse_transform(new_X)
new_X_1 = new_X_1[:,2]
print(new_X_1)
for_this_cluster_csv['Sales'] = new_X_1
print(for_this_cluster_csv)
for_this_cluster_csv.to_csv(r'D:\Documents\dataAnalytics\predicted_CLUSTER_4_TEST&STORE.csv', index = False, header=True)

#load test data
for_this_cluster_csv = pd.read_csv("No_promo_cluster4_validation.csv")
for_this_cluster = for_this_cluster_csv
print(data_set.columns)
for_this_cluster['Sales'] = for_this_cluster['Sales'].fillna(0)
for_this_cluster = for_this_cluster.drop(['Store', 'Customers','SchoolHoliday'], axis = 1)
for_this_cluster['Date'] =  pd.to_datetime(for_this_cluster['Date'], format='%m/%d/%Y')
for_this_cluster['Date'] = (for_this_cluster['Date'] - for_this_cluster['Date'].min())  / np.timedelta64(1,'D')
for_this_cluster = for_this_cluster[['DayOfWeek', 'Date', 'Sales','Open', 'Promo','StateHoliday']]
#add padding
for_this_cluster.loc[len(for_this_cluster)] = 0
print(for_this_cluster)

test_values = for_this_cluster.values
test_values = test_values.astype('float32')
#normalize features
scaled_1 = scaler.transform(test_values)
reframed = series_to_supervised(scaled_1, 1, 1)
reframed.drop(reframed.columns[[6,7,9,10,11]], axis=1, inplace=True)
print(reframed)

values = reframed.values
print(values)
value_X, value_y = values[:, :-1], values[:, -1]
value_X = value_X.reshape((value_X.shape[0], 1, value_X.shape[1]))
six_week_predictions = model.predict(value_X)
print(six_week_predictions)
value_X = value_X.reshape((value_X.shape[0], value_X.shape[2]))
new_X = np.delete(value_X, 2, 1)
new_X = np.append(new_X, six_week_predictions, axis=1)
new_X[:, [5, 2]] = new_X[:, [2, 5]]
print(new_X)
new_X_1 = scaler.inverse_transform(new_X)
new_X_1 = new_X_1[:,2]
print(new_X_1)
for_this_cluster_csv['Sales'] = new_X_1
print(for_this_cluster_csv)
for_this_cluster_csv.to_csv(r'D:\Documents\dataAnalytics\predicted_CLUSTER_4_VALIDATION.csv', index = False, header=True)






