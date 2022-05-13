import os.path
import pickle
import time
import warnings 
warnings.filterwarnings("ignore")
from datetime import timedelta
import pandas as pd
import statsmodels.api as ar
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from numpy.random import default_rng
from sklearn.preprocessing import MinMaxScaler
from numpy.random import default_rng
rng = default_rng(12345) #random generator seed
import numpy as np


# Calculating the expected average waiting time function
def average_waiting_time(forecast,interval,capacity):
    #how many passengers will come to station every "interval" minutes in last 1 hour
    for k in range(len(forecast)):
        if forecast[k] < 0: forecast[k] = 0
        
    fdf = pd.DataFrame(forecast.astype(int))    
    # fdf = fdf[fdf.index > fdf.index.max() - pd.Timedelta(hours=1)]


    
    # prepare data
    values = fdf.values
    values = values.reshape((len(values), 1))

    # estimate how many seats available based on how many passengers in the station
    scaler = MinMaxScaler(feature_range=(0, capacity))
    scaler = scaler.fit(values)
    num_passengers_in_metro = scaler.transform(values)
    fdf["In_Metro"] = num_passengers_in_metro.astype(int)
    fdf["In_Station_Oneway"] = (fdf["Forecast"]/2).astype(int) #assuming half of the people in station will que up for one of the two ways 
    fdf["Seats_Available"] = capacity-fdf["In_Metro"]

    # batch sequence: all passengers arriving druing an interval is one batch
    batch_seq = []
    
    for p in range(len(fdf)):
        num_passenger_oneway = fdf["In_Station_Oneway"][p]
        lam = num_passenger_oneway/interval
        
        # the times at which passenger arraive during the interval between 2 metro trains.
        # assign the arrival time for every passenger in the batch
        # based on poisson distribution 
        passenger_arrival_times = rng.poisson(lam=lam, size=num_passenger_oneway)%interval 
        passenger_arrival_times.sort()
        batch_seq.append(list(passenger_arrival_times))
        
    # now calculate waiting times
    average_batch_waiting_times = []
    remaining = []
    for q in range(len(fdf)):
        seats_available=fdf["Seats_Available"][q]

        batch_seq[q][0:0] = remaining # add the remaining passengers from previous batch to the que
        batch_waiting_times = interval-np.array(batch_seq[q])  # waiting time calculation
        

        if seats_available < len(batch_seq[q]):
            average_batch_waiting_times.append(batch_waiting_times[0:seats_available].mean()) 
            remaining = batch_seq[q][seats_available:]
            remaining = list(np.array(remaining) * -1)  # waiting time calculation for the passengers from previous batch
            del batch_seq[q][0:seats_available]
        else:
            average_batch_waiting_times.append(batch_waiting_times.mean())
            remaining = []

    # calculate the expected avg WT for all passnegrs
    f1_average_waiting_time = np.nan_to_num(np.array(average_batch_waiting_times)).mean()        
    
    return f1_average_waiting_time

# Setting the API
token = "*****" 
org = "******"
bucket = "_monitoring"
client = InfluxDBClient(url="https://westeurope-1.azure.cloud2.influxdata.com", token=token, org=org, verify_ssl=False) 

write_api = client.write_api(write_options=SYNCHRONOUS)


# reading the raw data
if os.path.exists('***'):
    df=pd.read_csv('*****')
    df.columns=["Date","Station"]
    df['Date'] = df['Date'].astype('datetime64[ns]')
    
    # Grouping by the data every 10min for every station 
    groupby_df1 = df.groupby([pd.Grouper(key='Date',freq='10T'),
                              df[df['Station'] == 'Burj Khalifa/ Dubai Mall Metro Station'].Station]).size().reset_index(name='Passengers')
    groupby_df1.set_index('Date', inplace=True)
    groupby_df1.drop('Station', inplace=True, axis=1)
    # groupby_df1.to_csv("Burj Khalifa Metro Station.csv")
    
    groupby_df2 = df.groupby([pd.Grouper(key='Date',freq='10T'),
                              df[df['Station'] == 'Mall of the Emirates Metro Station'].Station]).size().reset_index(name='Passengers')
    groupby_df2.set_index('Date', inplace=True)
    groupby_df2.drop('Station', inplace=True, axis=1)                    
    # groupby_df2.to_csv("Mall of the Emirates Metro Station.csv")

    groupby_df3 = df.groupby([pd.Grouper(key='Date',freq='10T'),
                              df[df['Station'] == 'BurJuman Metro Station'].Station]).size().reset_index(name='Passengers')
    groupby_df3.set_index('Date', inplace=True)
    groupby_df3.drop('Station', inplace=True, axis=1)                    
    # groupby_df3.to_csv("BurJuman Metro Station.csv")

    groupby_df4 = df.groupby([pd.Grouper(key='Date',freq='10T'),
                              df[df['Station'] == 'Union  Metro Station'].Station]).size().reset_index(name='Passengers')
    groupby_df4.set_index('Date', inplace=True)
    groupby_df4.drop('Station', inplace=True, axis=1)                    
    # groupby_df4.to_csv("Union  Metro Station.csv")

    groupby_df5 = df.groupby([pd.Grouper(key='Date',freq='10T'),
                              df[df['Station'] == 'ADCB Metro Station'].Station]).size().reset_index(name='Passengers')
    groupby_df5.set_index('Date', inplace=True)
    groupby_df5.drop('Station', inplace=True, axis=1)                    
    # groupby_df5.to_csv("ADCB Metro Station.csv")

    groupby_df6 = df.groupby([pd.Grouper(key='Date',freq='10T'),
                              df[df['Station'] == 'Sharaf DG Metro Station'].Station]).size().reset_index(name='Passengers')
    groupby_df6.set_index('Date', inplace=True)
    groupby_df6.drop('Station', inplace=True, axis=1)                    
    # groupby_df6.to_csv("Sharaf DG Metro Station.csv")

    # Splitting the training data and the test data
    df_train1 = groupby_df1[groupby_df1.index < '2021-11-17 09:00:00'] 
    df_test1 = groupby_df1[groupby_df1.index >= '2021-11-17 09:00:00']

    df_train2 = groupby_df2[groupby_df2.index < '2021-11-17 09:00:00'] 
    df_test2 = groupby_df2[groupby_df2.index >= '2021-11-17 09:00:00']

    df_train3 = groupby_df3[groupby_df3.index < '2021-11-17 09:00:00'] 
    df_test3 = groupby_df3[groupby_df3.index >= '2021-11-17 09:00:00']

    df_train4 = groupby_df4[groupby_df4.index < '2021-11-17 09:00:00'] 
    df_test4 = groupby_df4[groupby_df4.index >= '2021-11-17 09:00:00']

    df_train5 = groupby_df5[groupby_df5.index < '2021-11-17 09:00:00'] 
    df_test5 = groupby_df5[groupby_df5.index >= '2021-11-17 09:00:00']

    df_train6 = groupby_df6[groupby_df6.index < '2021-11-17 09:00:00'] 
    df_test6 = groupby_df6[groupby_df6.index >= '2021-11-17 09:00:00']

  
else:
    # If there is no dataset exit
    print(('/n There is no dataset to be found /n'))
    exit()


# Creating a model for each station with their own unique ARIMA order
# then we save the model on the  disk
model=ar.tsa.arima.ARIMA(df_train1,order=(4,0,4))
model_fit = model.fit()
with open('model1.pkl', 'wb') as pkl:
    pickle.dump(model_fit, pkl)   
 
model=ar.tsa.arima.ARIMA(df_train2,order=(2,0,4))
model_fit = model.fit()
with open('model2.pkl', 'wb') as pkl:
    pickle.dump(model_fit, pkl)   
 
model=ar.tsa.arima.ARIMA(df_train3,order=(2,0,7))
model_fit = model.fit()
with open('model3.pkl', 'wb') as pkl:
    pickle.dump(model_fit, pkl)   
 
model=ar.tsa.arima.ARIMA(df_train4,order=(2,0,3))
model_fit = model.fit()
with open('model4.pkl', 'wb') as pkl:
    pickle.dump(model_fit, pkl)   
 
model=ar.tsa.arima.ARIMA(df_train5,order=(2,0,3))
model_fit = model.fit()
with open('model5.pkl', 'wb') as pkl:
    pickle.dump(model_fit, pkl)   
 
model=ar.tsa.arima.ARIMA(df_train6,order=(2,0,3))
model_fit = model.fit()
with open('model6.pkl', 'wb') as pkl:
    pickle.dump(model_fit, pkl)   
 


# empty dataframes  
df1 = pd.DataFrame()     
df2 = pd.DataFrame()     
df3 = pd.DataFrame()     
df4 = pd.DataFrame()     
df5 = pd.DataFrame()     
df6 = pd.DataFrame()     


i=0
c=1

# start the simulation
while i in range(len(df_test1)):
    
    # append the test data to the empty df for forecasting
    # and also append the test data to the train data for retraining the models
    df1=df1.append(df_test1.iloc[i])
    df_train1=df_train1.append(df1.iloc[i])

    df2=df2.append(df_test2.iloc[i])
    df_train2=df_train2.append(df2.iloc[i])

    df3=df3.append(df_test3.iloc[i])
    df_train3=df_train3.append(df3.iloc[i])

    df4=df4.append(df_test4.iloc[i])
    df_train4=df_train4.append(df4.iloc[i])

    df5=df5.append(df_test5.iloc[i])
    df_train5=df_train5.append(df5.iloc[i])

    df6=df6.append(df_test6.iloc[i])
    df_train6=df_train6.append(df6.iloc[i])

   
    # load the models then forecast for the next 1 hour and a half
    with open('model1.pkl', 'rb') as pkl:
       model_fit = pickle.load(pkl)
    forecast1 = model_fit.predict(len(df1), (len(df1)-1) + 9,
                                  typ = 'levels').rename('Forecast')                 
    
    with open('model2.pkl', 'rb') as pkl:
       model_fit = pickle.load(pkl)
    forecast2 = model_fit.predict(len(df2), (len(df2)-1) + 9,
                                  typ = 'levels').rename('Forecast')                 
    with open('model3.pkl', 'rb') as pkl:
       model_fit = pickle.load(pkl)
    forecast3 = model_fit.predict(len(df3), (len(df3)-1) + 9,
                                  typ = 'levels').rename('Forecast')                 
  
    with open('model4.pkl', 'rb') as pkl:
       model_fit = pickle.load(pkl)
    forecast4 = model_fit.predict(len(df4), (len(df4)-1) + 9,
                                  typ = 'levels').rename('Forecast')                 

    with open('model5.pkl', 'rb') as pkl:
       model_fit = pickle.load(pkl)
    forecast5 = model_fit.predict(len(df5), (len(df5)-1) + 9,
                                  typ = 'levels').rename('Forecast')                 
  
    with open('model6.pkl', 'rb') as pkl:
       model_fit = pickle.load(pkl)
    forecast6 = model_fit.predict(len(df6), (len(df6)-1) + 9,
                                  typ = 'levels').rename('Forecast')                 


    # retrain the models every hour
    if c==6:
        model=ar.tsa.arima.ARIMA(df_train1,order=(4,0,4))
        model_fit = model.fit()
        with open('model1.pkl', 'wb') as pkl:
            pickle.dump(model_fit, pkl)   
            
        model=ar.tsa.arima.ARIMA(df_train2,order=(2,0,4))
        model_fit = model.fit()
        with open('model2.pkl', 'wb') as pkl:
            pickle.dump(model_fit, pkl)   
        
        model=ar.tsa.arima.ARIMA(df_train3,order=(2,0,7))
        model_fit = model.fit()
        with open('model3.pkl', 'wb') as pkl:
            pickle.dump(model_fit, pkl)   
        
        model=ar.tsa.arima.ARIMA(df_train4,order=(2,0,3))
        model_fit = model.fit()
        with open('model4.pkl', 'wb') as pkl:
            pickle.dump(model_fit, pkl)   
        
        model=ar.tsa.arima.ARIMA(df_train5,order=(2,0,3))
        model_fit = model.fit()
        with open('model5.pkl', 'wb') as pkl:
            pickle.dump(model_fit, pkl)   
        
        model=ar.tsa.arima.ARIMA(df_train6,order=(2,0,3))
        model_fit = model.fit()
        with open('model6.pkl', 'wb') as pkl:
            pickle.dump(model_fit, pkl)   

    
        c=0    
    c+=1
    
    
    # API code -  Sending The actual values and prediction values
    point = Point("Burj Khalifa/ Dubai Mall Metro Station1") \
    .tag("host", "host1") \
    .field("Actual-001", df1['Passengers'][-1]) \
    .time((df1.index[-1]).strftime("%m/%d/%Y, %H:%M:%S"), WritePrecision.NS)    
    write_api.write(bucket, org, point)
    
    point = Point("Mall of the Emirates Metro Station1") \
    .tag("host", "host1") \
    .field("Actual-002", df2['Passengers'][-1]) \
    .time((df2.index[-1]).strftime("%m/%d/%Y, %H:%M:%S"), WritePrecision.NS)    
    write_api.write(bucket, org, point)
            
    point = Point("BurJuman Metro Station1") \
    .tag("host", "host1") \
    .field("Actual-003", df3['Passengers'][-1]) \
    .time((df3.index[-1]).strftime("%m/%d/%Y, %H:%M:%S"), WritePrecision.NS)    
    write_api.write(bucket, org, point)
            
    point = Point("Union Metro Station1") \
    .tag("host", "host1") \
    .field("Actual-004", df4['Passengers'][-1]) \
    .time((df4.index[-1]).strftime("%m/%d/%Y, %H:%M:%S"), WritePrecision.NS)    
    write_api.write(bucket, org, point)
            
    point = Point("ADCB Metro Station1") \
    .tag("host", "host1") \
    .field("Actual-005", df5['Passengers'][-1]) \
    .time((df5.index[-1]).strftime("%m/%d/%Y, %H:%M:%S"), WritePrecision.NS)    
    write_api.write(bucket, org, point)
            
    point = Point("Sharaf DG Metro Station1") \
    .tag("host", "host1") \
    .field("Actual-006", df6['Passengers'][-1]) \
    .time((df6.index[-1]).strftime("%m/%d/%Y, %H:%M:%S"), WritePrecision.NS)    
    write_api.write(bucket, org, point)
    
    print("The Actual :", df1['Passengers'][-1], 'The date :', df1.index[-1])
    
    for x in range(9):             
        if forecast1.iloc[x] < 0 : forecast1.iloc[x] = 0
        point = Point("Burj Khalifa/ Dubai Mall Metro Station1") \
        .tag("host", "host1") \
        .field("Prediction001", round(forecast1.iloc[x])) \
        .time((df1.index[-1]+timedelta(minutes=10*(x+1))).strftime("%m/%d/%Y, %H:%M:%S"), WritePrecision.NS)    
        write_api.write(bucket, org, point)

        if forecast2.iloc[x] < 0 : forecast2.iloc[x] = 0
        point = Point("Mall of the Emirates Metro Station1") \
        .tag("host", "host1") \
        .field("Prediction002", round(forecast2.iloc[x])) \
        .time((df2.index[-1]+timedelta(minutes=10*(x+1))).strftime("%m/%d/%Y, %H:%M:%S"), WritePrecision.NS)    
        write_api.write(bucket, org, point)

        if forecast3.iloc[x] < 0 : forecast3.iloc[x] = 0
        point = Point("BurJuman Metro Station1") \
        .tag("host", "host1") \
        .field("Prediction003", round(forecast3.iloc[x])) \
        .time((df3.index[-1]+timedelta(minutes=10*(x+1))).strftime("%m/%d/%Y, %H:%M:%S"), WritePrecision.NS)    
        write_api.write(bucket, org, point)

        if forecast4.iloc[x] < 0 : forecast4.iloc[x] = 0
        point = Point("Union Metro Station1") \
        .tag("host", "host1") \
        .field("Prediction004", round(forecast4.iloc[x])) \
        .time((df4.index[-1]+timedelta(minutes=10*(x+1))).strftime("%m/%d/%Y, %H:%M:%S"), WritePrecision.NS)    
        write_api.write(bucket, org, point)

        if forecast5.iloc[x] < 0 : forecast5.iloc[x] = 0
        point = Point("ADCB Metro Station1") \
        .tag("host", "host1") \
        .field("Prediction005", round(forecast5.iloc[x])) \
        .time((df5.index[-1]+timedelta(minutes=10*(x+1))).strftime("%m/%d/%Y, %H:%M:%S"), WritePrecision.NS)    
        write_api.write(bucket, org, point)

        if forecast6.iloc[x] < 0 : forecast6.iloc[x] = 0
        point = Point("Sharaf DG Metro Station1") \
        .tag("host", "host1") \
        .field("Prediction006", round(forecast6.iloc[x])) \
        .time((df6.index[-1]+timedelta(minutes=10*((x+1)))).strftime("%m/%d/%Y, %H:%M:%S"), WritePrecision.NS)    
        write_api.write(bucket, org, point)

    print('Prediction: ', round(forecast1.iloc[0]),' date: ', df1.index[-1]+timedelta(minutes=10))
    

    # read number of metro parameter from file
    if os.path.exists('***'):
        df_params=pd.read_csv('***')
        df_params.columns=["num_metro","interval","capacity"] 
    else:
        print('parameter file doesnt exist')
        exit()
        
    old_num_metro = 5  # number of metros in service on the line 
    new_num_metro = df_params["num_metro"][0] 
    #  to check if the input is negative
    if new_num_metro < 0:
        new_num_metro = old_num_metro
        
    interval = df_params["interval"][0] # metro interval
    if old_num_metro != new_num_metro:  # then update the interval
        interval = (old_num_metro * interval)/new_num_metro
        old_num_metro = new_num_metro
    capacity = df_params["capacity"][0] # metro passenger capacity


    f1_average_waiting_time = average_waiting_time(forecast=forecast1,interval=interval,capacity=capacity)
    f2_average_waiting_time = average_waiting_time(forecast=forecast2,interval=interval,capacity=capacity)
    f3_average_waiting_time = average_waiting_time(forecast=forecast3,interval=interval,capacity=capacity)
    f4_average_waiting_time = average_waiting_time(forecast=forecast4,interval=interval,capacity=capacity)
    f5_average_waiting_time = average_waiting_time(forecast=forecast5,interval=interval,capacity=capacity)
    f6_average_waiting_time = average_waiting_time(forecast=forecast6,interval=interval,capacity=capacity)
    
    
    # Api code - sending the lon, lat and Avg WT
    point = Point("Burj Khalifa/ Dubai Mall Metro Station1") \
    .tag("host", "host1") \
    .field("lat", 25.20139461387934) \
    .field("lon", 55.26945716854178) \
    .field("WT001", round(f1_average_waiting_time,1) ) \
    .time((df1.index[-1]+timedelta(minutes=10*(x+1))).strftime("%m/%d/%Y, %H:%M:%S"), WritePrecision.NS)    
    write_api.write(bucket, org, point)

    point = Point("Mall of the Emirates Metro Station1") \
    .tag("host", "host1") \
    .field("lat", 25.120194240063377) \
    .field("lon", 55.19998459827798625) \
    .field("WT002", round(f2_average_waiting_time,1) ) \
    .time((df2.index[-1]+timedelta(minutes=10*(x+1))).strftime("%m/%d/%Y, %H:%M:%S"), WritePrecision.NS)    
    write_api.write(bucket, org, point)

    point = Point("BurJuman Metro Station1") \
    .tag("host", "host1") \
    .field("lat", 25.254838747236235) \
    .field("lon", 55.30422384807238) \
    .field("WT003", round(f3_average_waiting_time,1) ) \
    .time((df3.index[-1]+timedelta(minutes=10*(x+1))).strftime("%m/%d/%Y, %H:%M:%S"), WritePrecision.NS)    
    write_api.write(bucket, org, point)

    point = Point("Union Metro Station1") \
    .tag("host", "host1") \
    .field("lat", 25.266430803460665) \
    .field("lon", 55.313672173335924) \
    .field("WT004", round(f4_average_waiting_time,1) ) \
    .time((df4.index[-1]+timedelta(minutes=10*(x+1))).strftime("%m/%d/%Y, %H:%M:%S"), WritePrecision.NS)    
    write_api.write(bucket, org, point)

    point = Point("ADCB Metro Station1") \
    .tag("host", "host1") \
    .field("lat", 25.244457157825533) \
    .field("lon", 55.298184442949974) \
    .field("WT005", round(f5_average_waiting_time,1) ) \
    .time((df5.index[-1]+timedelta(minutes=10*(x+1))).strftime("%m/%d/%Y, %H:%M:%S"), WritePrecision.NS)    
    write_api.write(bucket, org, point)

    point = Point("Sharaf DG Metro Station1") \
    .tag("host", "host1") \
    .field("lat", 25.25843624556656) \
    .field("lon", 55.29754669674732) \
    .field("WT006", round(f6_average_waiting_time,1) ) \
    .time((df6.index[-1]+timedelta(minutes=10*(x+1))).strftime("%m/%d/%Y, %H:%M:%S"), WritePrecision.NS)    
    write_api.write(bucket, org, point)
    
    print('AWTs : ', round(f1_average_waiting_time,2))
    print(" - - - - - - - -")
    time.sleep(5)
    
    i+=1
