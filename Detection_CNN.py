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
# Reshape arrays for CNN
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val=X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define CNN architecture for detection
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(12,1)))
model.add(tf.keras.layers.MaxPooling1D(pool_size=1))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train model
earlystopping=tf.keras.callbacks.EarlyStopping()
h=model.fit(X_train,Y_train,
          epochs=10,
          batch_size=512,
          validation_data=(X_val,Y_val),
          callbacks=[earlystopping]
         )
# Show results :Classification report, Confusion matrix, Accuracy, Precision, Recall and F1-score
print(classification_report(yhatt,y_test))
print(confusion_matrix(yhatt,y_test))
print('accuracy:  %.4f'%accuracy_score(yhatt,y_test))
print('P:  %.4f'%metrics.precision_score(yhatt,y_test,average=None).mean())
print('R:  %.4f'%metrics.recall_score(yhatt,y_test,average=None).mean())
print('F1: %.4f'%metrics.f1_score(yhatt,y_test,average=None).mean())
