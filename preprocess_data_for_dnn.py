import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#load data
data= pd.read_csv("./data/data_dnn.csv")
data=data.drop('Unnamed: 0',axis=1)
# Remove time features from dataset
features=data.drop(['Date first seen','date2','date_sec'],axis=1)
targets=features['label']
features=features.drop('label',axis=1)
# Scale data with standard scaler
dataset=preprocessing.scale(features)

# Load indices used for train test split
ind=np.load('./data/indices.npy')

# Split data into train, validation and test parts
end_train=int(ind[int(len(ind)*0.6)])
end_val=int(ind[int(len(ind)*0.8)])
end_test=int(ind[len(ind)-1])

X_train=dataset[0:end_train]
Y_train=targets[0:end_train]

X_val=dataset[end_train:end_val]
Y_val=targets[end_train:end_val]

X_test=dataset[end_val:end_test]
Y_test=targets[end_val:end_test]

# Convert data from numpy array to dataframe for save them to csv output
X_train=pd.DataFrame(X_train)
Y_train=pd.DataFrame(Y_train)

X_val=pd.DataFrame(X_val)
Y_val=pd.DataFrame(Y_val)

X_test=pd.DataFrame(X_test)
Y_test=pd.DataFrame(Y_test)
# Save data 
X_train.to_csv('./data/x_train_data.csv')
Y_train.to_csv('./data/y_train_data.csv')

X_val.to_csv('./data/x_val_data.csv')
Y_val.to_csv('./data/y_val_data.csv')

X_test.to_csv('./data/x_test_data.csv')
Y_test.to_csv('./data/y_test_data.csv')

