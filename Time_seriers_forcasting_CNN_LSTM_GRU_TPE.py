import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adadelta, Adam, rmsprop

# Load train and validation
X_train=pd.read_csv('./data/x_train_data.csv')
Y_train=pd.read_csv('./data/y_train_data.csv')

X_val=pd.read_csv('./data/x_val_data.csv')
Y_val=pd.read_csv('./data/y_val_data.csv')

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


# Define space of hyper paramters for CNN

space = {
            'filters1': hp.choice('filters1',[32,64,128,256]),
            'kernels': hp.choice('kernels',[1,3,5,7,9]),
            'units1': hp.choice('units1', [32,64,128,256,512]),

            'dropout1': hp.uniform('dropout1', .25,.75),
            'dropout2': hp.uniform('dropout2',  .25,.75),
            'dropout3': hp.uniform('dropout3',  .25,.75),

            'batch_size' : hp.choice('batch_size', [32,64,128,256,512,1024]),

            'nb_epochs' : hp.choice('nb_epochs',[2,4,6,8,10,12]),
            'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
            'activation': 'relu'
        }
 
# Define function for hyper paramters tuning fo CNN
def f_nn(params):   
    model = tf.keras.Sequential()
    print ('Params testing: ', params)
    model.add(tf.keras.layers.Conv1D(filters=params['filters1'], kernel_size=params['kernels'], activation='relu', input_shape=(60,12)))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=1))
    model.add(tf.keras.layers.Dropout(params['dropout1']))
    model.add(tf.keras.layers.Conv1D(filters=params['filters1'], kernel_size=params['kernels'], activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=1))
    model.add(tf.keras.layers.Dropout(params['dropout2']))
    model.add(tf.keras.layers.Conv1D(filters=params['filters1'], kernel_size=params['kernels'], activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=1))
    model.add(tf.keras.layers.Dropout(params['dropout3']))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=params['units1'], activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=params['optimizer'],metrics=['accuracy'])

    model.fit(X_train, y_train, nb_epoch=params['nb_epochs'], batch_size=params['batch_size'],verbose=2)

    pred_auc =model.predict_proba(X_val, batch_size = 128, verbose = 2)
    acc = roc_auc_score(y_val, pred_auc)
    print('AUC:', acc)
    return {'loss': -acc, 'status': STATUS_OK}

# Lunch training and validation with hyperparamter tuning
trials = Trials()
best = fmin(f_nn, space, algo=tpe.suggest, max_evals=20, trials=trials)
print ('best: ',best)

# Define space for LSTM and GRU
space = {
            'units1': hp.choice('units1', [32,64,128,256]),
            'units2': hp.choice('units2', [32,64,128,256]),
            'units3': hp.choice('units3', [32,64,128,256]),
            'units4': hp.choice('units4', [32,64,128,256]),
            'batch_size' : hp.choice('batch_size', [32,64,128,256,512,1024]),
            'nb_epochs' : hp.choice('nb_epochs',[2,4,6,8,10,12]),
            'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
            'activation': 'relu'
        }
# Define function for hyper paramters tuning fo LSTM
def f_nn(params):   
    model = tf.keras.Sequential()
    print ('Params testing: ', params)
    model.add(tf.keras.layers.LSTM(units=params['units1'], return_sequences=True,
               input_shape=(60, 12))) 
    model.add(tf.keras.layers.LSTM(units=params['units2'], return_sequences=True))  
    model.add(tf.keras.layers.LSTM(units=params['units3'])) 
    model.add(tf.keras.layers.Dense(units=params['units4'], activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=params['optimizer'],metrics=['accuracy'])

    model.fit(X_train, _train, nb_epoch=params['nb_epochs'], batch_size=params['batch_size'],verbose=2)

    pred_auc =model.predict_proba(X_val, batch_size = 128, verbose = 0)
    acc = roc_auc_score(y_val, pred_auc)
    print('AUC:', acc)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f_nn, space, algo=tpe.suggest, max_evals=20, trials=trials)
print ('best: ',best)

# Define function for hyper paramters tuning fo GRU
def f_nn(params):   
    model = tf.keras.Sequential()
    print ('Params testing: ', params)
    model.add(tf.keras.layers.GRU(units=params['units1'], return_sequences=True,
               input_shape=(60, 12))) 
    model.add(tf.keras.layers.GRU(units=params['units2'], return_sequences=True))  
    model.add(tf.keras.layers.GRU(units=params['units3'])) 
    model.add(tf.keras.layers.Dense(units=params['units4'], activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=params['optimizer'],metrics=['accuracy'])

    model.fit(X_train, _train, nb_epoch=params['nb_epochs'], batch_size=params['batch_size'],verbose=2)

    pred_auc =model.predict_proba(X_val, batch_size = 128, verbose = 0)
    acc = roc_auc_score(y_val, pred_auc)
    print('AUC:', acc)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f_nn, space, algo=tpe.suggest, max_evals=20, trials=trials)
print ('best: ',best)

