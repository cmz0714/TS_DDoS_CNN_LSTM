import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adadelta, Adam, rmsprop
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn import metrics

# Load train, validation and test data
X_train=pd.read_csv('./data/x_train_data.csv')
Y_train=pd.read_csv('./data/y_train_data.csv')

X_val=pd.read_csv('./data/x_val_data.csv')
Y_val=pd.read_csv('./data/y_val_data.csv')

X_test=pd.read_csv('./data/x_test_data.csv')
Y_test=pd.read_csv('./data/y_test_data.csv')

# Define function devoted to apply sliding window method for forcasting
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# Create arrays with window of 60 flows
time_steps = 60
X_train, y_train = create_dataset(X_train, Y_train, time_steps)
X_val, y_val= create_dataset(X_val, Y_val, time_steps)
X_test, y_test = create_dataset(X_test, Y_test, time_steps)

# Reshape arrays to use them in CNN and LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val=X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test=X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define CNN architecture
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(60,12)))
model.add(tf.keras.layers.MaxPooling1D(pool_size=1))
model.add(tf.keras.layers.Dropout(0.52))
#model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
#model.add(tf.keras.layers.MaxPooling1D(pool_size=1))
#model.add(tf.keras.layers.  Dropout(0.29))
#model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
#model.add(tf.keras.layers.MaxPooling1D(pool_size=1))
#model.add(tf.keras.layers.Dropout(0.33))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
h=model.fit(X_train,y_train,epochs=8,batch_size=256,validation_data=(X_val,y_val),callbacks=[earlystopping])

# Test model
yhat = model.predict(X_test, verbose=2)
yhatt= (yhat>=0.5)

# Show results :Classification report, Confusion matrix, Accuracy, Precision, Recall and F1-score
print(classification_report(yhatt,y_test))
print(confusion_matrix(yhatt,y_test))
print('accuracy:  %.4f'%accuracy_score(yhatt,y_test))
print('P:  %.4f'%metrics.precision_score(yhatt,y_test,average=None).mean())
print('R:  %.4f'%metrics.recall_score(yhatt,y_test,average=None).mean())
print('F1: %.4f'%metrics.f1_score(yhatt,y_test,average=None).mean())

# Define LSTM architecture
model=tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(64, return_sequences=True,
               input_shape=(60, 12))) 
#model.add(tf.keras.layers.LSTM(64, return_sequences=True))  
#model.add(tf.keras.layers.LSTM(64)) 
model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# Train LSTM Model
earlystopping=tf.keras.callbacks.EarlyStopping()
h=model.fit(X_train,y_train,epochs=6,batch_size=128,validation_data=(X_val,y_val),callbacks=[earlystopping])

# Test model
yhat = model.predict(X_test, verbose=2)
yhatt= (yhat>=0.5)

# Show results :Classification report, Confusion matrix, Accuracy, Precision, Recall and F1-score
print(classification_report(yhatt,y_test))
print(confusion_matrix(yhatt,y_test))
print('accuracy:  %.4f'%accuracy_score(yhatt,y_test))
print('P:  %.4f'%metrics.precision_score(yhatt,y_test,average=None).mean())
print('R:  %.4f'%metrics.recall_score(yhatt,y_test,average=None).mean())
print('F1: %.4f'%metrics.f1_score(yhatt,y_test,average=None).mean())

# Define GRU architecture
model=tf.keras.Sequential()
model.add(tf.keras.layers.GRU(64, return_sequences=True,
               input_shape=(60, 12))) 
model.add(tf.keras.layers.GRU(64, return_sequences=True))  
model.add(tf.keras.layers.GRU(64)) 
model.add(tf.keras.layers.Dense(32))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# Train GRU Model
earlystopping=tf.keras.callbacks.EarlyStopping()
h=model.fit(X_train,y_train,epochs=6,batch_size=128,validation_data=(X_val,y_val),callbacks=[earlystopping])

# Test model
yhat = model.predict(X_test, verbose=2)
yhatt= (yhat>=0.5)

# Show results :Classification report, Confusion matrix, Accuracy, Precision, Recall and F1-score
print(classification_report(yhatt,y_test))
print(confusion_matrix(yhatt,y_test))
print('accuracy:  %.4f'%accuracy_score(yhatt,y_test))
print('P:  %.4f'%metrics.precision_score(yhatt,y_test,average=None).mean())
print('R:  %.4f'%metrics.recall_score(yhatt,y_test,average=None).mean())
print('F1: %.4f'%metrics.f1_score(yhatt,y_test,average=None).mean())

# Count number of predicted DDoS flows and normal flows based on 3600s time window to compare results with statistical methods
ind=np.load('./data/indices.npy')
p1,p2,yhat,yhatt = [],[],[],[]
test_size=int(len(ind)*0.8)
j=ind[test_size]
j=int(j)
for i in ind[test_size+1:]:
    yhatt=[]
    i=int(i)
    xx=dataset[j:i]
    xx=pd.DataFrame(xx)
    y_test=targets[j:i]
    x_input, y_test = create_dataset(xx, y_test, 60)    
    x_input = x_input.reshape(xx.shape[0], xx.shape[1],1)
    yhat = model.predict(x_input, verbose=2)
    for k in yhat:
        if k>=0.5:
            yhatt= np.append(yhatt,1)
        else:
            yhatt= np.append(yhatt,0)
    print(confusion_matrix(yhatt,y_test))
    if(confusion_matrix(yhatt,y_test).shape==(2,2)):
        p1=np.append(p1,confusion_matrix(yhatt,y_test)[0][0])
        p2=np.append(p2,confusion_matrix(yhatt,y_test)[1][1])
    else:
        p1=np.append(p1,0)
        p2=np.append(p2,confusion_matrix(yhatt,y_test)[0][0])
    j=i

# Load real number of DDoS and normal flows to validate DNN
y1=np.load("./data/number_ddos.npy")
y2=np.load("./data/number_normal.npy")

y11=y1[134:]
y21=y1[134:]
# show results of DNN based on MSE and MAE to compare our model with statistical methods
mse = np.sqrt(mean_squared_error(p1, y11))
print('Test MSE: %.2f' % mse)
mae = np.sqrt(mean_absolute_error(p2, y21))
print('Test MAE: %.2f' % mae)

