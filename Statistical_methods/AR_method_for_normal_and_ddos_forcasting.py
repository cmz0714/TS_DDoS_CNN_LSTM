import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error,mean_absolute_error

y1=np.load("../data/number_ddos.npy")
y2=np.load("../data/number_normal.npy")
# Apply AR method to forcast number of DDoS flows in next hour
X = y1
size=int((0.80*len(X)))
train, test = X[:size], X[size:]
# train autoregression
model = AR(train)
model_fit = model.fit()
window = model_fit.k_ar
coef = model_fit.params
# walk forward over time steps in test
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
	length = len(history)
	lag = [history[i] for i in range(length-window,length)]
	yhat = coef[0]
	for d in range(window):
		yhat += coef[d+1] * lag[window-d-1]
	obs = test[t]
	predictions.append(np.abs(yhat))
	history.append(obs)
###############################
# calcul of MSE and MAE
mse = np.sqrt(mean_squared_error(test, predictions))
print('Test MSE: %.2f' % mse)
mae = np.sqrt(mean_absolute_error(test, predictions))
print('Test MAE: %.2f' % mae)
##################################
# plot results of predictions including real values
x1=np.arange(0,len(X))
x2=np.arange(size,len(X))
plt.figure(figsize=(12,5))
plt.title('Trend of Normal flows with AR')
plt.xlabel('Steps')
plt.ylabel('Number of Normal flows')
plt.plot(x1,X)
plt.plot(x2,predictions, color='red')
plt.show()
#############################################################
# Apply AR method to forcast number of Normal flows in next hour
X = y2
size=int((0.80*len(X)))
train, test = X[:size], X[size:]
# train autoregression
model = AR(train)
model_fit = model.fit()
window = model_fit.k_ar
coef = model_fit.params
# walk forward over time steps in test
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
	length = len(history)
	lag = [history[i] for i in range(length-window,length)]
	yhat = coef[0]
	for d in range(window):
		yhat += coef[d+1] * lag[window-d-1]
	obs = test[t]
	predictions.append(np.abs(yhat))
	history.append(obs)
###############################
# calcul of MSE and MAE
mse = np.sqrt(mean_squared_error(test, predictions))
print('Test MSE: %.2f' % mse)
mae = np.sqrt(mean_absolute_error(test, predictions))
print('Test MAE: %.2f' % mae)
##################################
# plot results of predictions including real values
x1=np.arange(0,len(X))
x2=np.arange(size,len(X))
plt.figure(figsize=(12,5))
plt.title('Trend of Normal flows with AR')
plt.xlabel('Steps')
plt.ylabel('Number of Normal flows')
plt.plot(x1,X)
plt.plot(x2,predictions, color='red')
plt.show()
#############################################################