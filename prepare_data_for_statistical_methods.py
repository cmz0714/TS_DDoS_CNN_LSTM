import numpy as np
import pandas as pd
import dateutil.parser

# data lead 
data= pd.read_csv("data_for_ts.csv")
data=data.drop('Unnamed: 0',axis=1)

# Convert feature date from string to date
data['date2']=data['Date first seen'].apply(lambda x:dateutil.parser.parse(x))
# drop duplicate data based on timestamp
data=data.drop_duplicates(subset='date2')
# sort data based on timestamp
data=data.sort_values(by='date2')
# reset index 
data=data.reset_index()
# drop index because data is sroted based on timestamp
data=data.drop('index',axis=1)
# convert timestamp to microsecond is used for calcul and create windwos for time series anlysis
data['date_sec']=data['date2'].apply(lambda x: x.replace(microsecond=0).isoformat(' '))
data['date_sec']=data['date_sec'].apply(lambda x:dateutil.parser.parse(x))

# calcul difference of seconds betwen first flow and all others flows
list_sec2=[]
for i in range(1,754965):
    date2=data['date_sec'][i]-data['date_sec'][0]
    date2=date2.total_seconds()
    list_sec2=np.append(list_sec2,date2)

# create array contains indexes of first flow of each window
ind=[]
win=3600
number_of_iters=int(data.shape[0]/win)
for c in range(win,(number_of_iters+1)*win,win):
    ind=np.append(ind,(np.abs(list_sec2-c)).argmin())
######################################################
# calcul number of DDoS attacks and normal flows in each window
steps=ind.shape[0]
j=0
y1=[]
y2=[]
for i in ind:
    i=int(i)
    if(data[j:i].label.value_counts().shape[0]==2):
        y1=np.append(y1,data[j:i].label.value_counts()[0])
        y2=np.append(y2,data[j:i].label.value_counts()[1])
    else:
        y1=np.append(y1,0)
        y2=np.append(y2,data[j:i].label.value_counts()[1])
    j=i
######################################################
# save results of calcul number of attack flows and normal flows in each window time for apply time series forcasting methods
np.save('./data/number_ddos.npy', y1)
np.save('./data/number_normal.npy', y2)
######################################################
# save indices for data preprossing for CNN and LSTM
np.save('indices.npy',ind)


