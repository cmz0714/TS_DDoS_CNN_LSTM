import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error,mean_absolute_error

y1=np.load("../data/number_ddos.npy")
y2=np.load("../data/number_normal.npy")

# Apply SES methods to forcast number of DDoS attacks in next hour
X=y1
size = int(len(X) * 0.80)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
# fit model
for t in range(len(test)):
	model = SimpleExpSmoothing(history)
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
# calcul of MSE and MAE
mse = np.sqrt(mean_squared_error(test, predictions))
print('Test MSE: %.2f' % mse)
mae = np.sqrt(mean_absolute_error(test, predictions))
print('Test MAE: %.2f' % mae)

# plot results of predictions including real values
x1=np.arange(0,len(X))
x2=np.arange(size,len(X))
plt.figure(figsize=(12,5))
plt.title('Trend of DDoS flows with SES')
plt.xlabel('Steps')
plt.ylabel('Number of DDoS flows')
plt.plot(x1,X)
plt.plot(x2,predictions, color='red')
plt.show()
########################################################################

# Apply SES methods to forcast number of Normal flows in next hour
X = y2
size = int(len(X) * 0.80)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = SimpleExpSmoothing(history)
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
#######################################################################
# calcul of MSE and MAE
mse = np.sqrt(mean_squared_error(test, predictions))
print('Test MSE: %.2f' % mse)
mae = np.sqrt(mean_absolute_error(test, predictions))
print('Test MAE: %.2f' % mae)

# plot results of predictions including real values
x1=np.arange(0,len(X))
x2=np.arange(size,len(X))
plt.figure(figsize=(12,5))
plt.title('Trend of Normal flows with SES')
plt.xlabel('Steps')
plt.ylabel('Number of Normal flows')
plt.plot(x1,X)
plt.plot(x2,predictions, color='red')
plt.show()
########################################################################